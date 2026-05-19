"""
Dashboard Streamlit — Análise de petições iniciais (Itaú).

Duas abas:
  Tab 1 "📊 Escopo & Regex": triagem de escopo em 3 níveis (regex) + gráficos
                              sobre os casos in-scope usando apenas colunas
                              confiáveis do regex + painel para disparar a IA.
  Tab 2 "🤖 Análise IA":     análise completa com campos extraídos pelo LLM.
                              Vazia até o usuário rodar a IA na Tab 1.

Sidebar único, filtros globais (afetam as duas abas).
"""

import io
import json
import os
import unicodedata
from datetime import datetime
from pathlib import Path

import hashlib
import re

import nest_asyncio
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.formula.api as smf
import streamlit as st
from dotenv import load_dotenv
from scipy.stats import chi2_contingency

import ia_pipeline
from regex_otimizado import classificar_escopo_regex, processar_batch

# Raiz do projeto = pai da pasta dashboard/. Usada para localizar dataset, geojson e .env.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DASHBOARD_DIR = Path(__file__).resolve().parent
CSV_FILENAME = "dataset_clinica_20261.csv"
# Locais onde procuramos o CSV automaticamente, em ordem de preferência.
CSV_CANDIDATE_PATHS = [
    DASHBOARD_DIR / CSV_FILENAME,
    PROJECT_ROOT / CSV_FILENAME,
]

nest_asyncio.apply()
load_dotenv(PROJECT_ROOT / ".env")

# Ponte para Streamlit Community Cloud: `st.secrets` não popula `os.environ` automaticamente.
# Sem isso, `ia_pipeline._cliente()` falharia mesmo com OPENAI_API_KEY definido nos secrets do app.
try:
    if "OPENAI_API_KEY" in st.secrets and not os.getenv("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
except Exception:
    pass

st.set_page_config(page_title="Dashboard Jurídico — Itaú", layout="wide")


# ════════════════════════════════════════════════════════════════════════════
# UTILITÁRIOS DE DADOS
# ════════════════════════════════════════════════════════════════════════════

def corrigir_mojibake(valor):
    if isinstance(valor, str) and ("Ã" in valor or "Â" in valor):
        try:
            return valor.encode("latin1").decode("utf-8")
        except UnicodeError:
            return valor
    return valor


@st.cache_data(show_spinner=False)
def load_data(csv_bytes: bytes, _origem: str):
    """Carrega o CSV a partir de bytes. `_origem` (path local ou nome do upload) entra na
    chave de cache para invalidar quando a fonte muda — o conteúdo (`csv_bytes`) já é
    suficiente, mas o rótulo também é hasheado e ajuda no debugging."""
    df = pd.read_csv(io.BytesIO(csv_bytes), encoding="utf-8")
    colunas_texto = df.select_dtypes(include="object").columns
    for col in colunas_texto:
        df[col] = df[col].apply(corrigir_mojibake)
    if "magistrado" in df.columns:
        df["magistrado"] = df["magistrado"].astype(str).str.strip().str.upper()

    decisoes = df["decisao"].fillna("").tolist()

    resultados_regex = processar_batch(decisoes)
    for col, valores in resultados_regex.items():
        df[col] = valores

    classificacoes = [classificar_escopo_regex(t) for t in decisoes]
    df["nivel_confianca_regex"] = [c["nivel"] for c in classificacoes]
    df["motivo_escopo_regex"] = [c["motivo"] for c in classificacoes]
    df["fora_do_escopo_regex"] = [c["resultado"] for c in classificacoes]

    return df


def _localizar_csv_local() -> Path | None:
    for p in CSV_CANDIDATE_PATHS:
        if p.exists():
            return p
    return None


def _ler_bytes_csv(arquivo) -> bytes:
    """Lê bytes do CSV — aceita Path ou UploadedFile do Streamlit."""
    if isinstance(arquivo, Path):
        return arquivo.read_bytes()
    return arquivo.getvalue()


@st.cache_data(show_spinner=False)
def load_geojson():
    with open(PROJECT_ROOT / "geodata" / "SP.json", "r", encoding="utf-8") as f:
        return json.load(f)


def _normalizar_nome(nome):
    if pd.isna(nome):
        return ""
    t = unicodedata.normalize("NFKD", str(nome)).encode("ascii", "ignore").decode("ascii")
    return t.strip().lower()


@st.cache_data(show_spinner=False)
def build_comarca_to_geocodigo(geojson_data):
    mapping = {}
    for feature in geojson_data["features"]:
        mapping[_normalizar_nome(feature["properties"]["NOME"])] = feature["properties"]["GEOCODIGO"]
    return mapping


def _recortar_para_boxplot(df, col, percentil=99):
    """Remove valores acima do percentil dado para o boxplot não achatar com 1 outlier gigante.
    Se houver menos de 10 pontos, devolve o df inteiro (não faz sentido cortar)."""
    serie = df[col].astype(float)
    if len(serie) < 10:
        return df
    limite = serie.quantile(percentil / 100)
    if pd.isna(limite) or limite <= 0:
        return df
    return df[serie <= limite]


def padronizar_canal(c):
    c = str(c).lower().strip()
    if "whatsapp" in c or "whats" in c:
        return "whatsapp"
    if "e-mail" in c or "email" in c:
        return "e-mail"
    if "telefone" in c or "sac" in c or "0800" in c:
        return "telefone/sac"
    if "procon" in c:
        return "procon"
    if "consumidor.gov" in c:
        return "consumidor.gov"
    if "reclame aqui" in c or "reclameaqui" in c:
        return "reclame aqui"
    if "agência" in c or "agencia" in c or "presencial" in c:
        return "agência presencial"
    if c in ["não identificado", "nan", "não aplicável", "none"]:
        return "não identificado"
    return "outro"


# ════════════════════════════════════════════════════════════════════════════
# MAPA SP (preservado do dashboard anterior — agora aceita config dinâmica)
# ════════════════════════════════════════════════════════════════════════════

def render_mapa_sp(df_mapa, geojson_data, metrica_cfg, key_suffix=""):
    df_plot = df_mapa[df_mapa["geocodigo"].notna()].copy()
    if df_plot.empty:
        st.warning("Nenhuma comarca pôde ser mapeada para os municípios de SP.")
        return

    cfg = metrica_cfg
    col = cfg["col"]
    serie = df_plot[df_plot[col] > 0][col]
    zmax = serie.quantile(cfg.get("zmax_quantile", 0.85)) if not serie.empty else 100

    all_geocodigos = [f["properties"]["GEOCODIGO"] for f in geojson_data["features"]]
    all_nomes = [f["properties"]["NOME"] for f in geojson_data["features"]]

    base = go.Choropleth(
        geojson=geojson_data,
        locations=all_geocodigos,
        featureidkey="properties.GEOCODIGO",
        z=[0] * len(all_geocodigos),
        text=all_nomes,
        colorscale=[[0, "#1a1a2e"], [1, "#1a1a2e"]],
        showscale=False,
        marker_line_color="#374151",
        marker_line_width=0.4,
        hoverinfo="text",
    )

    overlay = go.Choropleth(
        geojson=geojson_data,
        locations=df_plot["geocodigo"].astype(str),
        z=df_plot[col],
        text=df_plot["comarca"],
        featureidkey="properties.GEOCODIGO",
        colorscale=cfg["scale"],
        zmin=0,
        zmax=zmax,
        colorbar=dict(title=cfg["label"]),
        marker_line_color="#4b5563",
        marker_line_width=0.4,
        hovertemplate=("<b>%{text}</b><br>" + cfg["label"] + ": %{z:" + cfg["fmt"] + "}<extra></extra>"),
    )

    fig = go.Figure(data=[base, overlay])
    fig.update_geos(fitbounds="geojson", visible=False, projection_type="mercator", bgcolor="rgba(0,0,0,0)")
    fig.update_layout(
        title=dict(text=cfg["title"], x=0.02, xanchor="left"),
        margin={"r": 0, "t": 50, "l": 0, "b": 0},
        height=620,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True, key=f"mapa_{cfg['col']}_{key_suffix}")


def agregar_mapa_ia(df, comarca_map):
    """Agrega métricas IA por comarca (valor médio danos morais + taxa procedência)."""
    agg = df.groupby("comarca").agg(
        total_processos=("id_processo", "count"),
        valor_medio_morais=("valor_danos_morais_ia", "mean"),
    ).reset_index()

    def _calc_taxa(group):
        total = len(group)
        if total == 0:
            return 0.0
        favoraveis = group["resultado_julgamento_regex"].isin(
            ["procedente", "parcialmente procedente"]
        ).sum()
        return favoraveis / total * 100

    taxa = df.groupby("comarca").apply(_calc_taxa, include_groups=False).reset_index()
    taxa.columns = ["comarca", "taxa_procedencia"]

    agg = agg.merge(taxa, on="comarca", how="left")
    agg["comarca_norm"] = agg["comarca"].apply(_normalizar_nome)
    agg["geocodigo"] = agg["comarca_norm"].map(comarca_map)
    agg["valor_medio_morais"] = agg["valor_medio_morais"].round(2)
    agg["taxa_procedencia"] = agg["taxa_procedencia"].round(1)
    return agg


def agregar_mapa_regex(df, comarca_map):
    """Agrega por comarca apenas o total de processos."""
    agg = df.groupby("comarca").agg(
        total_processos=("id_processo", "count"),
    ).reset_index()
    agg["comarca_norm"] = agg["comarca"].apply(_normalizar_nome)
    agg["geocodigo"] = agg["comarca_norm"].map(comarca_map)
    return agg


# ════════════════════════════════════════════════════════════════════════════
# CONFIGS DE MAPA
# ════════════════════════════════════════════════════════════════════════════

MAPA_REGEX_CONFIG = {
    "col": "total_processos",
    "title": "Total de processos por comarca",
    "label": "Total",
    "scale": [[0.0, "#1e1b4b"], [0.3, "#4338ca"], [0.7, "#818cf8"], [1.0, "#eef2ff"]],
    "fmt": ",.0f",
    "zmax_quantile": 0.85,
}

MAPA_IA_CONFIGS = {
    "valor_morais": {
        "col": "valor_medio_morais",
        "title": "Valor Médio de Danos Morais por Comarca",
        "label": "Valor Médio (R$)",
        "scale": [[0.0, "#2e1065"], [0.3, "#7e22ce"], [0.7, "#d946ef"], [1.0, "#fdf4ff"]],
        "fmt": ",.2f",
        "zmax_quantile": 0.80,
    },
    "taxa_procedencia": {
        "col": "taxa_procedencia",
        "title": "Taxa de Procedência por Comarca",
        "label": "Procedência (%)",
        "scale": [[0.0, "#022c22"], [0.3, "#059669"], [0.7, "#34d399"], [1.0, "#f0fdf4"]],
        "fmt": ".1f",
        "zmax_quantile": 0.85,
    },
}


# ════════════════════════════════════════════════════════════════════════════
# ABA 1 — ESCOPO & REGEX
# ════════════════════════════════════════════════════════════════════════════

def render_distribuicao_escopo(df_filtered):
    """3 metric cards + donut chart com a distribuição L1/L2/L3."""
    n_total = len(df_filtered)
    n_dentro = int((df_filtered["nivel_confianca_regex"] == 1).sum())
    n_incerto = int((df_filtered["nivel_confianca_regex"] == 2).sum())
    n_fora = int((df_filtered["nivel_confianca_regex"] == 3).sum())

    st.markdown("### Distribuição de Escopo (Regex em 3 níveis)")
    c1, c2, c3, c4 = st.columns([1, 1, 1, 2])
    c1.metric("✅ Dentro (L1)", f"{n_dentro:,}".replace(",", "."), f"{n_dentro/n_total*100:.1f}%" if n_total else "—")
    c2.metric("❓ Incerto (L2)", f"{n_incerto:,}".replace(",", "."), f"{n_incerto/n_total*100:.1f}%" if n_total else "—")
    c3.metric("❌ Fora (L3)", f"{n_fora:,}".replace(",", "."), f"{n_fora/n_total*100:.1f}%" if n_total else "—")

    with c4:
        if n_total > 0:
            df_donut = pd.DataFrame({
                "Nível": ["Dentro (L1)", "Incerto (L2)", "Fora (L3)"],
                "Casos": [n_dentro, n_incerto, n_fora],
            })
            fig = px.pie(
                df_donut, values="Casos", names="Nível", hole=0.55,
                color="Nível",
                color_discrete_map={
                    "Dentro (L1)": "#10b981",
                    "Incerto (L2)": "#f59e0b",
                    "Fora (L3)": "#ef4444",
                },
            )
            fig.update_traces(textposition="inside", textinfo="percent")
            fig.update_layout(showlegend=True, height=180, margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(fig, use_container_width=True)


def render_graficos_regex_dentro(df_dentro, geojson_data, comarca_map):
    """Gráficos apenas com colunas regex confiáveis sobre os casos in-scope (L1)."""
    if df_dentro.empty:
        st.info("Nenhum caso L1 (dentro do escopo) nos filtros atuais.")
        return

    st.markdown(f"### Análise dos {len(df_dentro):,} casos DENTRO do escopo".replace(",", "."))

    col1, col2 = st.columns(2)
    with col1:
        df_res = df_dentro[df_dentro["resultado_julgamento_regex"] != "não identificado"]
        if not df_res.empty:
            counts = df_res["resultado_julgamento_regex"].value_counts().reset_index()
            counts.columns = ["Resultado", "Quantidade"]
            fig = px.bar(
                counts, x="Quantidade", y="Resultado", orientation="h",
                title="Resultado do Julgamento", color="Resultado",
                color_discrete_sequence=px.colors.qualitative.Pastel,
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        df_rito = df_dentro[df_dentro["rito_processual_regex"] != "não identificado"]
        if not df_rito.empty:
            counts = df_rito["rito_processual_regex"].value_counts().reset_index()
            counts.columns = ["Rito", "Quantidade"]
            fig = px.bar(
                counts, x="Quantidade", y="Rito", orientation="h",
                title="Rito Processual", color="Rito",
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    col3, col4, col5 = st.columns(3)
    for col, campo, titulo, mapa_cores in [
        (col3, "justica_gratuita_regex", "Justiça Gratuita", {"sim": "#10b981", "não": "#94a3b8"}),
        (col4, "boletim_de_ocorrencia_regex", "Boletim de Ocorrência", {"sim": "#3b82f6", "não": "#94a3b8"}),
        (col5, "mencao_reclame_aqui_regex", "Menção a Reclame Aqui", {"sim": "#f97316", "não": "#94a3b8"}),
    ]:
        with col:
            if campo in df_dentro.columns:
                counts = df_dentro[campo].value_counts().reset_index()
                counts.columns = ["Valor", "Quantidade"]
                fig = px.pie(
                    counts, values="Quantidade", names="Valor", title=titulo,
                    color="Valor", color_discrete_map=mapa_cores,
                )
                fig.update_traces(textposition="inside", textinfo="percent+label")
                fig.update_layout(height=320, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

    if "repeticao_indebito_regex" in df_dentro.columns:
        counts = df_dentro["repeticao_indebito_regex"].value_counts().reset_index()
        counts.columns = ["Valor", "Quantidade"]
        fig = px.bar(
            counts, x="Valor", y="Quantidade", title="Repetição de Indébito",
            color="Valor", color_discrete_sequence=px.colors.qualitative.Pastel1,
        )
        fig.update_layout(showlegend=False, height=320)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Mapa de SP — Total de processos por comarca")
    df_mapa = agregar_mapa_regex(df_dentro, comarca_map)
    render_mapa_sp(df_mapa, geojson_data, MAPA_REGEX_CONFIG, key_suffix="regex")


def render_aviso_danos_regex(df_dentro):
    """Mostra danos extraídos por regex com banner de aproximação."""
    if df_dentro.empty or "valor_danos_morais_regex" not in df_dentro.columns:
        return
    st.markdown("### Valores monetários (extração regex)")
    st.warning("⚠️ Extração aproximada — use a aba **🤖 Análise IA** para valores precisos.")

    morais_pos = df_dentro[df_dentro["valor_danos_morais_regex"] > 0]
    materiais_pos = df_dentro[df_dentro["valor_danos_materiais_regex"] > 0]

    col1, col2 = st.columns(2)
    with col1:
        if not morais_pos.empty:
            df_plot = _recortar_para_boxplot(morais_pos, "valor_danos_morais_regex")
            fig = px.box(
                df_plot, y="valor_danos_morais_regex", points="outliers",
                title="Danos morais (regex, >R$0, sem top 1%)",
                labels={"valor_danos_morais_regex": "Valor (R$)"},
                color_discrete_sequence=["#9333ea"],
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    with col2:
        if not materiais_pos.empty:
            df_plot = _recortar_para_boxplot(materiais_pos, "valor_danos_materiais_regex")
            fig = px.box(
                df_plot, y="valor_danos_materiais_regex", points="outliers",
                title="Danos materiais (regex, >R$0, sem top 1%)",
                labels={"valor_danos_materiais_regex": "Valor (R$)"},
                color_discrete_sequence=["#0ea5e9"],
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)


def render_painel_pipeline_ia(df_filtered):
    """Controles para rodar a IA: modo, N de amostra, preview de custo, confirmação."""
    st.divider()
    st.markdown("## 🚀 Rodar análise com IA")
    st.markdown(
        "Selecione um subconjunto dos casos filtrados acima e dispare a extração de campos semânticos. "
        "Cache local evita chamadas duplicadas — sempre é mostrado o preview de custo antes."
    )

    api_key_presente = bool(os.getenv("OPENAI_API_KEY"))
    if not api_key_presente:
        st.error(
            "❌ `OPENAI_API_KEY` ausente. Adicione ao arquivo `.env` na raiz do projeto "
            "para habilitar a análise com IA. A aba Regex segue funcional sem isso."
        )
        return

    col_a, col_b = st.columns([1, 2])
    with col_a:
        modo_label = st.radio(
            "Modo de análise:",
            [
                "Classificar incertos com IA (L1 + L2 → análise completa)",
                "Apenas casos já confirmados dentro pelo regex (L1)",
            ],
            index=0,
            help=(
                "**Classificar incertos com IA**: a IA primeiro decide se cada caso L2 (incerto) "
                "está dentro ou fora do escopo, depois extrai os campos semânticos dos que ficaram "
                "dentro. Não existe estado 'continua incerto' no fim — todo L2 vira dentro ou fora.\n\n"
                "**Apenas L1**: ignora completamente os L2. A IA só extrai campos dos casos "
                "que o regex já confirmou como dentro do escopo (Nível 1)."
            ),
        )
        modo = "l1_plus_l2" if modo_label.startswith("Classificar") else "l1_only"

    with col_b:
        if modo == "l1_only":
            n_disponivel = int((df_filtered["nivel_confianca_regex"] == 1).sum())
        else:
            n_disponivel = int(df_filtered["nivel_confianca_regex"].isin([1, 2]).sum())
        if n_disponivel == 0:
            st.info(f"Nenhum caso elegível ({modo_label}) nos filtros atuais.")
            return
        n_amostra = st.slider(
            f"Casos a analisar (há {n_disponivel:,} elegíveis):".replace(",", "."),
            min_value=1, max_value=n_disponivel,
            value=min(50, n_disponivel),
            help=(
                "Amostragem aleatória com seed fixa (random_state=42), reproduzível entre "
                "execuções. Rodadas grandes podem levar vários minutos — use o preview de custo "
                "para dimensionar antes de confirmar."
            ),
        )

    if st.button("🔍 Pré-visualizar custo", type="secondary"):
        niveis_elegiveis = [1] if modo == "l1_only" else [1, 2]
        df_eleg = df_filtered[df_filtered["nivel_confianca_regex"].isin(niveis_elegiveis)]
        df_amostra = df_eleg.sample(n=min(n_amostra, len(df_eleg)), random_state=42)
        decisoes = df_amostra["decisao"].fillna("").tolist()
        niveis = df_amostra["nivel_confianca_regex"].tolist()
        preview = ia_pipeline.estimar_custo(decisoes, niveis, modo)
        st.session_state["preview_pendente"] = {
            "preview": preview,
            "modo": modo,
            "indices": df_amostra.index.tolist(),
        }

    preview_pendente = st.session_state.get("preview_pendente")
    if preview_pendente:
        p = preview_pendente["preview"]
        st.markdown("#### Preview da rodada")
        col1, col2, col3 = st.columns(3)
        col1.metric("Amostra", f"{p['linhas_amostra']:,}".replace(",", "."))
        col1.caption(f"In-scope estimado: {p['linhas_in_scope_estimado']:,}".replace(",", "."))

        col2.metric("Cache hits", f"{p['cache_hits_escopo'] + p['cache_hits_extracao']:,}".replace(",", "."))
        col2.caption(f"escopo: {p['cache_hits_escopo']} · extração: {p['cache_hits_extracao']}")

        col3.metric("Chamadas API", f"{p['calls_escopo'] + p['calls_extracao']:,}".replace(",", "."))
        col3.caption(f"escopo: {p['calls_escopo']} · extração: {p['calls_extracao']}")

        st.markdown(
            f"**Custo estimado**: ≈ **US$ {p['custo_usd_estimado']:.4f}**  ·  "
            f"modelo: `{p['modelo']}`  ·  concorrência: {p['concorrencia']}"
        )

        cb1, cb2 = st.columns(2)
        if cb1.button("✅ Confirmar e rodar", type="primary"):
            _executar_rodada_ia(df_filtered, preview_pendente["indices"], preview_pendente["modo"])
            st.session_state.pop("preview_pendente", None)
            st.rerun()
        if cb2.button("✕ Cancelar"):
            st.session_state.pop("preview_pendente", None)
            st.rerun()


def _executar_rodada_ia(df_filtered, indices_amostra, modo):
    df_amostra = df_filtered.loc[indices_amostra]
    decisoes = df_amostra["decisao"].fillna("").tolist()
    niveis = df_amostra["nivel_confianca_regex"].tolist()

    progress = st.progress(0.0, text="Iniciando…")
    status = st.empty()

    def on_progress(etapa: str, feitos: int, total: int):
        if total == 0:
            return
        progress.progress(feitos / total, text=f"{etapa.capitalize()}: {feitos}/{total}")
        status.text(f"Etapa: {etapa} · processados {feitos} de {total}")

    try:
        escopo_por_idx, extracao_por_idx, stats = ia_pipeline.processar_lote(
            decisoes, niveis, modo, on_progress=on_progress,
        )
    except RuntimeError as e:
        st.error(str(e))
        return

    progress.empty()
    status.empty()

    df_ia = df_amostra.reset_index(drop=False).copy()
    df_ia = df_ia.rename(columns={"index": "_idx_orig"})

    for campo in ia_pipeline._campos_padrao_ia:
        df_ia[f"{campo}_ia"] = None
    df_ia["status_extracao_ia"] = None
    df_ia["fora_do_escopo_final"] = None

    for pos_amostra in range(len(df_ia)):
        fora = escopo_por_idx.get(pos_amostra)
        df_ia.at[df_ia.index[pos_amostra], "fora_do_escopo_final"] = fora
        ext = extracao_por_idx.get(pos_amostra)
        if ext:
            for campo, valor in ext.items():
                if campo == "status_extracao":
                    df_ia.at[df_ia.index[pos_amostra], "status_extracao_ia"] = valor
                else:
                    df_ia.at[df_ia.index[pos_amostra], f"{campo}_ia"] = valor

    st.session_state["ia_results"] = df_ia
    st.session_state["ia_stats"] = stats
    st.session_state["ia_timestamp"] = datetime.now()
    st.session_state["ia_modo"] = modo


# ════════════════════════════════════════════════════════════════════════════
# ABA 2 — ANÁLISE IA
# ════════════════════════════════════════════════════════════════════════════

def render_aba_ia(geojson_data, comarca_map):
    df_ia = st.session_state.get("ia_results")
    if df_ia is None:
        st.info(
            "🤖 Nenhuma análise IA rodada nesta sessão.  \n"
            "Vá para a aba **📊 Escopo & Regex**, role até o painel **🚀 Rodar análise com IA**, "
            "ajuste a amostra e clique em **Pré-visualizar custo**."
        )
        return

    stats = st.session_state.get("ia_stats", {})
    ts = st.session_state.get("ia_timestamp")
    modo = st.session_state.get("ia_modo", "?")
    ts_str = ts.strftime("%d/%m/%Y %H:%M:%S") if ts else "—"

    n_total = len(df_ia)
    n_in_scope = int((df_ia["fora_do_escopo_final"] == False).sum())
    n_ok = int((df_ia["status_extracao_ia"] == "ok").sum())
    n_falhas = int((df_ia["status_extracao_ia"] == "falha_api").sum())

    st.success(
        f"✅ Análise rodada em **{ts_str}** · modo **{modo}** · "
        f"{n_total} casos · {n_in_scope} dentro do escopo · "
        f"{n_ok} extraídos · {n_falhas} falhas · "
        f"≈ US$ {stats.get('custo_usd_real_estimado', 0):.4f}"
    )

    df_dentro = df_ia[df_ia["fora_do_escopo_final"] == False].copy()
    if df_dentro.empty:
        st.warning("Nenhum caso confirmado dentro do escopo após a análise.")
        return

    st.divider()
    st.markdown("### Filtros adicionais (campos IA)")
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        tipos = sorted([x for x in df_dentro["tipo_acao_ia"].dropna().unique()])
        sel_tipo = st.multiselect("Tipo de ação (IA)", tipos)
    with col_f2:
        canais = sorted([x for x in df_dentro["canal_contato_ia"].dropna().unique()])
        sel_canal = st.multiselect("Canal de contato (IA)", canais)
    with col_f3:
        culpas = sorted([x for x in df_dentro["culpa_atribuida_ia"].dropna().unique()])
        sel_culpa = st.multiselect("Culpa atribuída (IA)", culpas)

    df_view = df_dentro
    if sel_tipo:
        df_view = df_view[df_view["tipo_acao_ia"].isin(sel_tipo)]
    if sel_canal:
        df_view = df_view[df_view["canal_contato_ia"].isin(sel_canal)]
    if sel_culpa:
        df_view = df_view[df_view["culpa_atribuida_ia"].isin(sel_culpa)]

    if df_view.empty:
        st.info("Nenhum caso após filtros IA.")
        return

    df_view = df_view.copy()
    df_view["canal_contato_ia"] = df_view["canal_contato_ia"].apply(padronizar_canal)

    # Recorta outliers em danos morais (IQR, igual ao dashboard antigo)
    valores_positivos = df_view[df_view["valor_danos_morais_ia"].astype(float) > 0]["valor_danos_morais_ia"].astype(float)
    if not valores_positivos.empty:
        q1 = valores_positivos.quantile(0.25)
        q3 = valores_positivos.quantile(0.75)
        iqr = q3 - q1
        limite = q3 + 1.5 * iqr
        if pd.isna(limite):
            limite = 50000
    else:
        limite = 0
    df_view["valor_danos_morais_ia"] = df_view["valor_danos_morais_ia"].astype(float)
    df_clean = df_view[df_view["valor_danos_morais_ia"] <= limite].copy()

    # ─── Parte 1: visão geral ───────────────────────────────────────────────
    st.header("Parte 1: Visão Geral (IA)")
    colA, colB = st.columns(2)
    with colA:
        counts = df_clean["tipo_acao_ia"].value_counts().reset_index()
        counts.columns = ["Tipo", "Quantidade"]
        fig = px.bar(
            counts, x="Quantidade", y="Tipo", orientation="h",
            title="Tipo de ação", color="Tipo",
            color_discrete_sequence=px.colors.qualitative.Bold,
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    with colB:
        morais_pos = df_clean[df_clean["valor_danos_morais_ia"] > 0]
        materiais_pos = df_clean[df_clean["valor_danos_materiais_ia"].astype(float) > 0]
        sub1, sub2 = st.columns(2)
        with sub1:
            if not morais_pos.empty:
                df_plot = _recortar_para_boxplot(morais_pos, "valor_danos_morais_ia")
                fig = px.box(
                    df_plot, y="valor_danos_morais_ia", points="outliers",
                    title="Danos morais (IA, >R$0, sem top 1%)",
                    labels={"valor_danos_morais_ia": "Valor (R$)"},
                    color_discrete_sequence=["#9333ea"],
                )
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
        with sub2:
            if not materiais_pos.empty:
                df_plot = _recortar_para_boxplot(materiais_pos, "valor_danos_materiais_ia")
                fig = px.box(
                    df_plot, y="valor_danos_materiais_ia", points="outliers",
                    title="Danos materiais (IA, >R$0, sem top 1%)",
                    labels={"valor_danos_materiais_ia": "Valor (R$)"},
                    color_discrete_sequence=["#0ea5e9"],
                )
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Mapas Coropléticos — Estado de SP")
    metrica = st.selectbox(
        "Métrica:",
        list(MAPA_IA_CONFIGS.keys()),
        format_func=lambda x: MAPA_IA_CONFIGS[x]["title"],
        key="metrica_mapa_ia",
    )
    df_mapa = agregar_mapa_ia(df_clean, comarca_map)
    render_mapa_sp(df_mapa, geojson_data, MAPA_IA_CONFIGS[metrica], key_suffix="ia")

    # ─── Parte 2: contato prévio ────────────────────────────────────────────
    st.header("Parte 2: Impacto do contato prévio (IA)")
    col1, col2 = st.columns([1, 2])
    with col1:
        counts = df_clean["contato_previo_banco_ia"].value_counts().reset_index()
        counts.columns = ["Contato?", "Quantidade"]
        fig = px.pie(
            counts, values="Quantidade", names="Contato?",
            title="Proporção de contato prévio",
            color="Contato?", color_discrete_map={"sim": "#10b981", "não": "#ef4444"},
        )
        fig.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        cruz = df_clean.groupby(["contato_previo_banco_ia", "resultado_julgamento_regex"]).size().reset_index(name="count")
        cruz["%"] = cruz.groupby("contato_previo_banco_ia")["count"].transform(lambda x: x / x.sum() * 100)
        fig = px.bar(
            cruz, x="contato_previo_banco_ia", y="%", color="resultado_julgamento_regex",
            title="Resultado × contato prévio (%)",
            text=cruz["%"].apply(lambda x: f"{x:.1f}%"),
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig.update_layout(barmode="stack")
        st.plotly_chart(fig, use_container_width=True)

    morais_pos = df_clean[df_clean["valor_danos_morais_ia"] > 0]
    if not morais_pos.empty:
        df_plot = _recortar_para_boxplot(morais_pos, "valor_danos_morais_ia")
        fig = px.box(
            df_plot, x="contato_previo_banco_ia", y="valor_danos_morais_ia",
            color="contato_previo_banco_ia", points="outliers",
            title="Danos morais por contato prévio (>R$0, sem top 1%)",
            color_discrete_map={"sim": "#10b981", "não": "#ef4444"},
        )
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        cruz = df_clean.groupby(["tipo_acao_ia", "contato_previo_banco_ia"]).size().reset_index(name="count")
        cruz["%"] = cruz.groupby("tipo_acao_ia")["count"].transform(lambda x: x / x.sum() * 100)
        fig = px.bar(
            cruz, x="tipo_acao_ia", y="%", color="contato_previo_banco_ia",
            title="Contato por tipo de ação (%)",
            text=cruz["%"].apply(lambda x: f"{x:.0f}%"),
            color_discrete_map={"sim": "#10b981", "não": "#ef4444"},
        )
        fig.update_layout(barmode="stack")
        st.plotly_chart(fig, use_container_width=True)
    with col4:
        cruz = df_clean.groupby(["culpa_atribuida_ia", "contato_previo_banco_ia"]).size().reset_index(name="count")
        cruz["%"] = cruz.groupby("contato_previo_banco_ia")["count"].transform(lambda x: x / x.sum() * 100)
        fig = px.bar(
            cruz, x="contato_previo_banco_ia", y="%", color="culpa_atribuida_ia",
            title="Culpa por contato prévio (%)",
            color_discrete_sequence=px.colors.qualitative.Pastel,
        )
        fig.update_layout(barmode="stack")
        st.plotly_chart(fig, use_container_width=True)

    # ─── Parte 3: canais ────────────────────────────────────────────────────
    st.header("Parte 3: Canais (IA)")
    canais_df = df_clean[df_clean["contato_previo_banco_ia"] == "sim"]
    if not canais_df.empty:
        counts = canais_df["canal_contato_ia"].value_counts().reset_index()
        counts.columns = ["Canal", "Quantidade"]
        fig = px.bar(
            counts, x="Quantidade", y="Canal", orientation="h",
            title="Canais de contato prévio mais utilizados",
            color="Quantidade", color_continuous_scale="Magma",
        )
        st.plotly_chart(fig, use_container_width=True)

        heat = pd.crosstab(canais_df["canal_contato_ia"], canais_df["resultado_julgamento_regex"], normalize="index") * 100
        fig = px.imshow(
            heat, text_auto=".1f", color_continuous_scale="YlGnBu",
            title="Probabilidade de resultado por canal (%)",
            labels=dict(x="Resultado", y="Canal", color="%"),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Nenhum caso com contato_previo_banco_ia='sim' nos filtros atuais.")

    # ─── Falhas ─────────────────────────────────────────────────────────────
    df_falhas = df_ia[df_ia["status_extracao_ia"] == "falha_api"]
    if not df_falhas.empty:
        with st.expander(f"⚠️ {len(df_falhas)} falhas de extração"):
            st.dataframe(df_falhas[["_idx_orig", "decisao"]].head(50), use_container_width=True)
            if st.button("🔄 Reprocessar falhas"):
                decisoes_falhas = {int(r["_idx_orig"]): r["decisao"] for _, r in df_falhas.iterrows()}
                with st.spinner(f"Reprocessando {len(decisoes_falhas)} casos…"):
                    resultados = ia_pipeline.reprocessar_falhas(decisoes_falhas)
                st.success(f"{len(resultados)} casos reprocessados. Recarregue a aba para ver.")

    st.divider()
    st.markdown("### Dados brutos analisados")
    st.dataframe(df_clean, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# MODELOS INFERENCIAIS — helpers, ajuste e render
# ════════════════════════════════════════════════════════════════════════════


def _estrela_inf(p):
    if pd.isna(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    if p < 0.10:
        return "."
    return "ns"


def _agrupar_cats(series, min_count=30):
    counts = series.value_counts()
    raras = counts[counts < min_count].index
    return series.apply(lambda x: "outras" if x in raras else x)


def _preparar_df_inf(ia_df):
    """Transforma ia_results no DataFrame pronto para os modelos. Espelha Cell 65 do notebook."""
    VAR_TRAT = "contato_previo_banco_ia"
    RES_COL = "resultado_julgamento_regex"
    CATS = [
        VAR_TRAT, RES_COL,
        "tipo_acao_ia", "rito_processual_regex",
        "boletim_de_ocorrencia_regex", "justica_gratuita_regex",
        "mencao_reclame_aqui_regex",
    ]
    df = ia_df.copy()
    for col in CATS:
        if col in df.columns:
            df[col] = df[col].apply(corrigir_mojibake).fillna("desconhecido").astype(str)
        else:
            df[col] = "desconhecido"

    df = df[df[VAR_TRAT].isin(["sim", "não"])].copy()
    if df.empty:
        return None

    def _proced(r):
        r = str(r).lower()
        if "improcedente" in r:
            return 0
        if "procedente" in r:
            return 1
        return np.nan

    df["procedente"] = df[RES_COL].apply(_proced)

    for orig, log_col in [
        ("valor_danos_morais_ia", "ln_morais"),
        ("valor_danos_materiais_ia", "ln_materiais"),
    ]:
        if orig in df.columns:
            vals = pd.to_numeric(df[orig], errors="coerce").clip(lower=0).fillna(0)
            df[log_col] = np.where(vals > 0, np.log(vals), np.nan)
            df[orig] = vals
        else:
            df[log_col] = np.nan
            df[orig] = 0.0

    for raw, grp in [("comarca", "comarca_grp"), ("assunto", "assunto_grp")]:
        base = (
            df[raw].fillna("desconhecida").astype(str)
            if raw in df.columns
            else pd.Series("desconhecida", index=df.index)
        )
        df[grp] = _agrupar_cats(base, min_count=30)

    return df


@st.cache_data(show_spinner=False)
def rodar_modelos_inferenciais(ia_hash: bytes):
    """Ajusta os modelos do notebook sobre ia_results. Retorna apenas números (sem objetos statsmodels)."""
    import warnings

    ia_df = st.session_state.get("_df_modelos_inferencial")
    if ia_df is None:
        return {"status": "sem_dados"}

    df = _preparar_df_inf(ia_df)
    if df is None or len(df) < 20:
        return {"status": "insuficiente", "n": 0 if df is None else len(df)}

    VAR_TRAT = "contato_previo_banco_ia"
    CTRL = (
        "C(tipo_acao_ia) + C(rito_processual_regex)"
        " + C(boletim_de_ocorrencia_regex) + C(justica_gratuita_regex)"
        " + C(mencao_reclame_aqui_regex)"
    )

    def _fit_logit(formula, data):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                return smf.logit(formula, data=data).fit(method="bfgs", maxiter=1000, disp=0)
            except Exception:
                try:
                    return smf.logit(formula, data=data).fit(method="nm", maxiter=2000, disp=0)
                except Exception:
                    return None

    def _fit_ols(formula, data):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                return smf.ols(formula, data=data).fit(cov_type="HC3")
            except Exception:
                return None

    def _extrai_trat(m, var_trat, exponenciar=False):
        if m is None:
            return None
        termo = next((t for t in m.params.index if var_trat in t and "sim" in t), None)
        if termo is None:
            return None
        coef = m.params[termo]
        p = m.pvalues[termo]
        if pd.isna(coef) or pd.isna(p):
            return None
        ci = m.conf_int().loc[termo]
        n = int(m.nobs)
        if exponenciar:
            return {
                "point": float(np.exp(coef)),
                "ci_low": float(np.exp(ci.iloc[0])),
                "ci_high": float(np.exp(ci.iloc[1])),
                "p": float(p),
                "n": n,
            }
        pct = (np.exp(coef) - 1) * 100
        return {
            "point": float(pct),
            "ci_low": float((np.exp(ci.iloc[0]) - 1) * 100),
            "ci_high": float((np.exp(ci.iloc[1]) - 1) * 100),
            "p": float(p),
            "n": n,
            "coef": float(coef),
        }

    res = {"df_inf": df, "status": "ok"}

    # ── Modelo A: Logit contato → procedência ───────────────────────────────
    df_a = df.dropna(subset=["procedente"]).copy()
    if len(df_a) >= 30 and df_a["procedente"].nunique() < 2:
        pct_proc = float(df_a["procedente"].mean() * 100)
        res["modelo_a"] = {"status": "sem_variacao", "n": len(df_a), "pct_proc": pct_proc}
    elif len(df_a) >= 30:
        specs_formulas = {
            "Sem controles": f'procedente ~ C({VAR_TRAT}, Treatment("não"))',
            "Com tipo de ação e rito": (
                f'procedente ~ C({VAR_TRAT}, Treatment("não")) + {CTRL}'
            ),
            "Modelo completo\n(comarca incluída)": (
                f'procedente ~ C({VAR_TRAT}, Treatment("não")) + {CTRL}'
                f" + C(comarca_grp) + C(assunto_grp)"
            ),
        }
        rows = []
        for label, formula in specs_formulas.items():
            m = _fit_logit(formula, df_a)
            r = _extrai_trat(m, VAR_TRAT, exponenciar=True)
            if r:
                rows.append({**r, "spec": label, "stars": _estrela_inf(r["p"])})
        res["modelo_a"] = {"specs": pd.DataFrame(rows), "n": len(df_a)}
    else:
        res["modelo_a"] = {"status": "insuficiente", "n": len(df_a)}

    # ── Modelo A-interação: AME por tipo de ação ────────────────────────────
    df_ai = df_a.copy()
    tipos_ok = df_ai.groupby("tipo_acao_ia")["procedente"].count()
    tipos_ok = tipos_ok[tipos_ok >= 20].index.tolist()
    if len(df_ai) >= 50 and len(tipos_ok) >= 2:
        formula_inter = (
            f'procedente ~ C({VAR_TRAT}, Treatment("não")) * C(tipo_acao_ia)'
            f" + C(rito_processual_regex) + C(boletim_de_ocorrencia_regex)"
            f" + C(justica_gratuita_regex) + C(mencao_reclame_aqui_regex)"
            f" + C(comarca_grp) + C(assunto_grp)"
        )
        m_i = _fit_logit(formula_inter, df_ai)
        if m_i is not None:
            ame_rows = []
            for tipo in tipos_ok:
                sub = df_ai[df_ai["tipo_acao_ia"] == tipo].copy()
                try:
                    sub_s = sub.copy()
                    sub_s[VAR_TRAT] = "sim"
                    sub_n = sub.copy()
                    sub_n[VAR_TRAT] = "não"
                    ame = (m_i.predict(sub_s).mean() - m_i.predict(sub_n).mean()) * 100
                    ame_rows.append({"tipo_acao": tipo, "ame_pp": round(float(ame), 1), "n": len(sub)})
                except Exception:
                    pass
            res["modelo_a_inter"] = pd.DataFrame(ame_rows).sort_values("ame_pp") if ame_rows else None
        else:
            res["modelo_a_inter"] = None
    else:
        res["modelo_a_inter"] = None

    # ── Modelo B: OLS contato → ln(danos morais) ────────────────────────────
    df_b = df.dropna(subset=["ln_morais"]).copy()
    if len(df_b) >= 30:
        specs_b = {
            "Sem controles": f'ln_morais ~ C({VAR_TRAT}, Treatment("não"))',
            "Com tipo de ação e rito": (
                f'ln_morais ~ C({VAR_TRAT}, Treatment("não")) + {CTRL}'
            ),
            "Modelo completo\n(comarca incluída)": (
                f'ln_morais ~ C({VAR_TRAT}, Treatment("não")) + {CTRL}'
                f" + C(comarca_grp) + C(assunto_grp)"
            ),
        }
        rows_b = []
        for label, formula in specs_b.items():
            m = _fit_ols(formula, df_b)
            r = _extrai_trat(m, VAR_TRAT, exponenciar=False)
            if r:
                rows_b.append({**r, "spec": label, "stars": _estrela_inf(r["p"])})
        res["modelo_b"] = {"specs": pd.DataFrame(rows_b), "n": len(df_b)}
    else:
        res["modelo_b"] = {"status": "insuficiente", "n": len(df_b)}

    # ── Modelo B': OLS contato → ln(danos materiais) ────────────────────────
    df_bp = df.dropna(subset=["ln_materiais"]).copy()
    if len(df_bp) >= 30:
        m = _fit_ols(
            f'ln_materiais ~ C({VAR_TRAT}, Treatment("não")) + {CTRL}'
            f" + C(comarca_grp) + C(assunto_grp)",
            df_bp,
        )
        r = _extrai_trat(m, VAR_TRAT, exponenciar=False)
        if r:
            res["modelo_bp"] = {**r, "n": len(df_bp), "stars": _estrela_inf(r["p"])}
        else:
            res["modelo_bp"] = {"status": "falha"}
    else:
        res["modelo_bp"] = {"status": "insuficiente", "n": len(df_bp)}

    # ── Modelo C: Logit → perfil dos casos com contato prévio ───────────────
    df_c = df.copy()
    df_c["contato_sim"] = (df_c[VAR_TRAT] == "sim").astype(float)
    if len(df_c) >= 50:
        m_c = _fit_logit(
            "contato_sim ~ C(tipo_acao_ia) + C(rito_processual_regex)"
            " + C(boletim_de_ocorrencia_regex) + C(justica_gratuita_regex)"
            " + C(mencao_reclame_aqui_regex) + C(comarca_grp) + C(assunto_grp)",
            df_c,
        )
        if m_c is not None:
            ci_df = m_c.conf_int()
            top20 = (
                pd.DataFrame({
                    "termo": m_c.params.index,
                    "coef": m_c.params.values,
                    "or": np.exp(m_c.params.values),
                    "ci_low": np.exp(ci_df.iloc[:, 0].values),
                    "ci_high": np.exp(ci_df.iloc[:, 1].values),
                    "p": m_c.pvalues.values,
                    "abs_z": np.abs(m_c.tvalues.values),
                })
                .query("termo != 'Intercept'")
                .sort_values("abs_z", ascending=False)
                .head(20)
                .reset_index(drop=True)
            )
            top20["stars"] = top20["p"].apply(_estrela_inf)
            res["modelo_c"] = top20
        else:
            res["modelo_c"] = None
    else:
        res["modelo_c"] = None

    # ── Magistrados: chi-square de heterogeneidade ──────────────────────────
    df_mag = df.dropna(subset=["procedente"]).copy()
    if "magistrado" in df_mag.columns and len(df_mag) >= 30:
        mag_n = df_mag.groupby("magistrado")["procedente"].count()
        mag_suf = mag_n[mag_n >= 20].index
        df_ms = df_mag[df_mag["magistrado"].isin(mag_suf)].copy()
        if len(mag_suf) >= 3:
            try:
                ct = pd.crosstab(df_ms["magistrado"], df_ms["procedente"])
                chi2_val, p_chi, dof, _ = chi2_contingency(ct)
                mag_agg = (
                    df_ms.groupby("magistrado")["procedente"]
                    .agg(taxa=lambda x: round(x.mean() * 100, 1), n="count")
                    .reset_index()
                    .sort_values("taxa", ascending=False)
                )
                res["magistrados"] = {
                    "chi2": float(chi2_val),
                    "p": float(p_chi),
                    "dof": int(dof),
                    "n_magistrados": int(len(mag_suf)),
                    "mag_df": mag_agg,
                }
            except Exception:
                res["magistrados"] = None
        else:
            res["magistrados"] = {"status": "insuficiente", "n": int(len(mag_suf))}
    else:
        res["magistrados"] = None

    return res


# ── Plots auxiliares ─────────────────────────────────────────────────────────

def _forest_plot_or(df_specs, title):
    """Forest plot de Odds Ratio com IC 95% por especificação do modelo."""
    fig = go.Figure()
    for _, row in df_specs.iterrows():
        sig = row["p"] < 0.05
        if sig and row["point"] > 1:
            color = "#10b981"
        elif sig and row["point"] < 1:
            color = "#ef4444"
        else:
            color = "#94a3b8"
        fig.add_trace(go.Scatter(
            x=[row["point"]],
            y=[row["spec"]],
            mode="markers",
            marker=dict(size=14, color=color),
            error_x=dict(
                type="data", symmetric=False,
                array=[row["ci_high"] - row["point"]],
                arrayminus=[row["point"] - row["ci_low"]],
                color=color, thickness=2, width=8,
            ),
            text=(
                f"OR={row['point']:.2f} [IC95%: {row['ci_low']:.2f}–{row['ci_high']:.2f}]"
                f"  p={row['p']:.4f} {row['stars']}  N={int(row['n']):,}".replace(",", ".")
            ),
            hoverinfo="text",
            showlegend=False,
        ))
    fig.add_vline(x=1, line_dash="dash", line_color="#64748b", line_width=1.5)
    fig.update_layout(
        title=title,
        xaxis_title="Odds Ratio  (1 = sem efeito, linha tracejada)",
        yaxis=dict(autorange="reversed"),
        height=max(280, 160 + 80 * len(df_specs)),
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def _forest_plot_pct(df_specs, title):
    """Forest plot de % de mudança com IC 95% por especificação (modelos OLS em log)."""
    fig = go.Figure()
    for _, row in df_specs.iterrows():
        sig = row["p"] < 0.05
        if sig and row["point"] > 0:
            color = "#9333ea"
        elif sig and row["point"] < 0:
            color = "#f59e0b"
        else:
            color = "#94a3b8"
        fig.add_trace(go.Scatter(
            x=[row["point"]],
            y=[row["spec"]],
            mode="markers",
            marker=dict(size=14, color=color),
            error_x=dict(
                type="data", symmetric=False,
                array=[row["ci_high"] - row["point"]],
                arrayminus=[row["point"] - row["ci_low"]],
                color=color, thickness=2, width=8,
            ),
            text=(
                f"{row['point']:+.1f}% [IC95%: {row['ci_low']:+.1f}% a {row['ci_high']:+.1f}%]"
                f"  p={row['p']:.4f} {row['stars']}  N={int(row['n']):,}".replace(",", ".")
            ),
            hoverinfo="text",
            showlegend=False,
        ))
    fig.add_vline(x=0, line_dash="dash", line_color="#64748b", line_width=1.5)
    fig.update_layout(
        title=title,
        xaxis_title="Variação percentual no valor  (0 = sem efeito, linha tracejada)",
        yaxis=dict(autorange="reversed"),
        height=max(280, 160 + 80 * len(df_specs)),
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def _callout_or(r, desc_pos, desc_neg):
    p, or_v = r["p"], r["point"]
    ci_low, ci_high, n = r["ci_low"], r["ci_high"], r["n"]
    stars = _estrela_inf(p)
    pct = abs(or_v - 1) * 100
    detalhe = (
        f"**OR = {or_v:.2f}** (IC95%: {ci_low:.2f} – {ci_high:.2f})"
        f"  |  p = {p:.4f} {stars}  |  N = {n:,}".replace(",", ".")
    )
    if p < 0.05 and or_v > 1:
        st.success(f"✅ {desc_pos} (~{pct:.0f}% mais chances de vitória)\n\n{detalhe}")
    elif p < 0.05 and or_v < 1:
        st.error(f"⚠️ {desc_neg} (~{pct:.0f}% menos chances de vitória)\n\n{detalhe}")
    else:
        st.info(f"↔️ Sem efeito estatisticamente significativo detectado.\n\n{detalhe}")


def _callout_pct(r, desc_pos, desc_neg):
    p, pct = r["p"], r["point"]
    ci_low, ci_high, n = r["ci_low"], r["ci_high"], r["n"]
    stars = _estrela_inf(p)
    sinal = "+" if pct > 0 else ""
    detalhe = (
        f"**{sinal}{pct:.1f}%** de variação"
        f"  |  IC95%: {ci_low:+.1f}% a {ci_high:+.1f}%"
        f"  |  p = {p:.4f} {stars}  |  N = {n:,}".replace(",", ".")
    )
    if p < 0.05 and pct > 0:
        st.success(f"✅ {desc_pos}\n\n{detalhe}")
    elif p < 0.05 and pct < 0:
        st.warning(f"⬇️ {desc_neg}\n\n{detalhe}")
    else:
        st.info(f"↔️ Sem efeito estatisticamente significativo detectado.\n\n{detalhe}")


# ── Seções de render ─────────────────────────────────────────────────────────

def _render_modelo_a(res):
    st.markdown("### Contato prévio com o banco aumenta as chances de vitória?")
    st.caption(
        "Compara a taxa de procedência entre casos onde o cliente entrou em contato com o banco "
        "antes de processar (SAC, Procon, Ouvidoria etc.) e casos sem contato registrado. "
        "O modelo controla pelo tipo de ação, rito processual, BO e comarca — isolando o efeito puro do contato."
    )

    modelo_a = res.get("modelo_a", {})
    df_inf = res.get("df_inf")

    if "status" in modelo_a:
        if modelo_a["status"] == "insuficiente":
            st.warning(f"Amostra insuficiente (N={modelo_a.get('n', 0)} casos com resultado identificado, mínimo: 30).")
        elif modelo_a["status"] == "sem_variacao":
            pct = modelo_a.get("pct_proc", 100.0)
            st.info(
                f"ℹ️ **Resultado uniforme na amostra**: {pct:.0f}% dos {modelo_a.get('n', 0)} casos processados "
                f"resultaram em procedência — não há variação suficiente para ajustar o modelo logístico. "
                "Processe casos com resultados variados (procedente e improcedente) para habilitar esta análise."
            )
            # Ainda mostra o gráfico descritivo mesmo sem o modelo
            if df_inf is not None:
                df_plot = df_inf.dropna(subset=["procedente"]).copy()
                taxa = (
                    df_plot.groupby("contato_previo_banco_ia")["procedente"]
                    .agg(taxa=lambda x: round(x.mean() * 100, 1), n="count")
                    .reset_index()
                )
                taxa["label"] = taxa.apply(lambda r: f"{r['taxa']:.1f}%  (N={int(r['n'])})", axis=1)
                fig = px.bar(
                    taxa, x="contato_previo_banco_ia", y="taxa",
                    text="label",
                    color="contato_previo_banco_ia",
                    color_discrete_map={"sim": "#10b981", "não": "#94a3b8"},
                    title="Taxa de vitória observada por contato prévio (sem modelo — dados sem variação)",
                    labels={"contato_previo_banco_ia": "Contato prévio", "taxa": "% de casos procedentes"},
                )
                fig.update_traces(textposition="outside")
                fig.update_layout(showlegend=False, yaxis_range=[0, 110])
                st.plotly_chart(fig, use_container_width=True)
        return

    df_specs = modelo_a.get("specs", pd.DataFrame())
    if df_specs.empty:
        st.warning("Não foi possível ajustar o modelo A.")
        return

    completo = df_specs.iloc[-1].to_dict()
    _callout_or(
        completo,
        "Contato prévio está associado a MAIS chances de vitória",
        "Contato prévio está associado a MENOS chances de vitória",
    )

    col1, col2 = st.columns(2)
    with col1:
        if df_inf is not None:
            df_plot = df_inf.dropna(subset=["procedente"]).copy()
            taxa = (
                df_plot.groupby("contato_previo_banco_ia")["procedente"]
                .agg(taxa=lambda x: round(x.mean() * 100, 1), n="count")
                .reset_index()
            )
            taxa["label"] = taxa.apply(lambda r: f"{r['taxa']:.1f}%  (N={int(r['n'])})", axis=1)
            fig = px.bar(
                taxa, x="contato_previo_banco_ia", y="taxa",
                text="label",
                color="contato_previo_banco_ia",
                color_discrete_map={"sim": "#10b981", "não": "#94a3b8"},
                title="Taxa de vitória observada por contato prévio",
                labels={"contato_previo_banco_ia": "Contato prévio", "taxa": "% de casos procedentes"},
            )
            fig.update_traces(textposition="outside")
            fig.update_layout(showlegend=False, yaxis_range=[0, 110])
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = _forest_plot_or(df_specs, "Odds Ratio do contato prévio — robustez entre modelos")
        st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "**Como ler o forest plot:** OR > 1 (à direita da linha tracejada) = contato prévio associado a mais chances "
        "de vitória. OR < 1 = menos chances. Se os três pontos ficam do mesmo lado, o resultado é robusto. "
        "Barras horizontais = IC de 95%. Cinza = não significativo (p ≥ 0.05)."
    )


def _render_modelo_a_inter(res):
    df_ame = res.get("modelo_a_inter")
    if df_ame is None or df_ame.empty:
        with st.expander("Efeito por tipo de ação — dados insuficientes"):
            st.info("São necessários ao menos 2 tipos de ação com ≥ 20 casos cada para esta análise.")
        return

    st.markdown("### O efeito do contato varia conforme o tipo de ação?")
    st.caption(
        "Efeito marginal médio (AME): diferença em pontos percentuais (pp) na probabilidade de vitória "
        "ao comparar casos com e sem contato prévio, separado por tipo de ação. "
        "Calculado a partir do modelo com interação."
    )

    df_ame = df_ame.copy()
    df_ame["label"] = df_ame.apply(lambda r: f"{r['ame_pp']:+.1f} pp  (N={int(r['n'])})", axis=1)

    fig = px.bar(
        df_ame, x="ame_pp", y="tipo_acao", orientation="h",
        text="label",
        color="ame_pp",
        color_continuous_scale=[[0, "#ef4444"], [0.5, "#e2e8f0"], [1, "#10b981"]],
        color_continuous_midpoint=0,
        title="Efeito do contato prévio na taxa de vitória por tipo de ação (pontos percentuais)",
        labels={"ame_pp": "Efeito marginal (pp)", "tipo_acao": "Tipo de ação"},
    )
    fig.add_vline(x=0, line_dash="dash", line_color="#64748b")
    fig.update_traces(textposition="outside")
    fig.update_layout(
        showlegend=False, coloraxis_showscale=False,
        height=max(300, 200 + 50 * len(df_ame)),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Verde = contato prévio ajuda nesse tipo de ação  |  Vermelho = atrapalha  |  "
        "Os valores são aproximados (efeito marginal médio no modelo com interação)."
    )


def _render_modelo_b(res):
    st.markdown("### Contato prévio muda o valor dos danos morais recebidos?")
    st.caption(
        "Analisa somente os casos com condenação em danos morais (valor > R$ 0). "
        "O modelo OLS estima a variação percentual no valor em função do contato prévio, "
        "controlando pelo mesmo conjunto de variáveis do Modelo A."
    )

    modelo_b = res.get("modelo_b", {})
    df_inf = res.get("df_inf")

    if "status" in modelo_b and modelo_b["status"] == "insuficiente":
        st.info(f"Poucos casos com danos morais > R$0 (N={modelo_b.get('n', 0)}). Mínimo: 30.")
    else:
        df_specs = modelo_b.get("specs", pd.DataFrame())
        if not df_specs.empty:
            completo = df_specs.iloc[-1].to_dict()
            _callout_pct(
                completo,
                "Contato prévio está associado a danos morais MAIORES",
                "Contato prévio está associado a danos morais MENORES",
            )
            col1, col2 = st.columns(2)
            with col1:
                if df_inf is not None and "valor_danos_morais_ia" in df_inf.columns:
                    df_plot = df_inf[df_inf["valor_danos_morais_ia"] > 0].copy()
                    if not df_plot.empty:
                        df_plot = _recortar_para_boxplot(df_plot, "valor_danos_morais_ia")
                        fig = px.box(
                            df_plot, x="contato_previo_banco_ia", y="valor_danos_morais_ia",
                            color="contato_previo_banco_ia",
                            color_discrete_map={"sim": "#9333ea", "não": "#94a3b8"},
                            title="Danos morais por contato prévio (casos com valor > R$0)",
                            labels={
                                "contato_previo_banco_ia": "Contato prévio",
                                "valor_danos_morais_ia": "Valor (R$)",
                            },
                            points="outliers",
                        )
                        fig.update_layout(showlegend=False)
                        st.plotly_chart(fig, use_container_width=True)
            with col2:
                fig = _forest_plot_pct(df_specs, "% de variação nos danos morais — robustez entre modelos")
                st.plotly_chart(fig, use_container_width=True)

            st.caption(
                "**Como ler:** Valores à direita de 0% = contato prévio associado a danos maiores. "
                "À esquerda = danos menores. Modelo OLS em log(valor), convertido para % de variação: exp(β) − 1."
            )

    # Modelo B' em expander
    modelo_bp = res.get("modelo_bp", {})
    with st.expander("Danos materiais — Modelo B'"):
        if isinstance(modelo_bp, dict) and "status" in modelo_bp:
            st.info(
                f"Modelo indisponível: {modelo_bp.get('status', '')} "
                f"(N={modelo_bp.get('n', 0)} casos com danos materiais > R$0, mínimo: 30)."
            )
        elif modelo_bp:
            _callout_pct(
                modelo_bp,
                "Contato prévio está associado a danos materiais MAIORES",
                "Contato prévio está associado a danos materiais MENORES",
            )


def _render_modelo_c(res):
    df_c = res.get("modelo_c")
    if df_c is None or df_c.empty:
        st.info("Modelo C indisponível (mínimo 50 casos necessários).")
        return

    st.markdown("### Qual é o perfil dos casos com contato prévio?")
    st.caption(
        "Quais características do processo estão mais associadas ao cliente ter contatado o banco "
        "antes de processar? Os fatores abaixo são ranqueados pela força da associação (|z-score|)."
    )

    _VAR_LABELS = {
        "tipo_acao_ia": "tipo de ação",
        "rito_processual_regex": "rito processual",
        "boletim_de_ocorrencia_regex": "boletim de ocorrência",
        "justica_gratuita_regex": "justiça gratuita",
        "mencao_reclame_aqui_regex": "Reclame Aqui",
        "comarca_grp": "comarca",
        "assunto_grp": "assunto",
        "contato_previo_banco_ia": "contato prévio",
    }

    def _clean_termo(t):
        m = re.match(r"C\(([^,)]+)[^)]*\)\[T\.(.+)\]$", t)
        if m:
            var, val = m.group(1).strip(), m.group(2).strip()
            return f"{_VAR_LABELS.get(var, var)}: {val}"
        return t

    df_plot = df_c.copy()
    # Remove ORs extremos (separação perfeita / quasi-separação) que distorcem a escala
    df_plot = df_plot[(df_plot["or"] < 500) & (df_plot["or"] > 0.002)].copy()
    if df_plot.empty:
        st.info("Todos os termos apresentaram separação perfeita — OR não estimável de forma estável.")
        return
    df_plot["termo_limpo"] = df_plot["termo"].apply(_clean_termo)
    df_plot["label"] = df_plot.apply(
        lambda r: f"OR={r['or']:.2f} {r['stars']}  [IC: {r['ci_low']:.2f}–{r['ci_high']:.2f}]",
        axis=1,
    )

    fig = px.bar(
        df_plot, x="or", y="termo_limpo", orientation="h",
        text="label",
        color="or",
        color_continuous_scale=[[0, "#ef4444"], [0.5, "#e2e8f0"], [1.0, "#10b981"]],
        color_continuous_midpoint=1.0,
        title="Fatores associados ao contato prévio com o banco (top 20 por intensidade de associação)",
        labels={"or": "Odds Ratio", "termo_limpo": "Fator"},
    )
    fig.add_vline(x=1, line_dash="dash", line_color="#64748b")
    fig.update_traces(textposition="outside")
    fig.update_layout(
        showlegend=False,
        coloraxis_showscale=False,
        height=max(400, 260 + 22 * len(df_plot)),
        yaxis=dict(autorange="reversed"),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "OR > 1 (verde) = esse grupo tem mais probabilidade de ter feito contato prévio. "
        "OR < 1 (vermelho) = menos. "
        "Fatores sem asterisco (ns) = associação não significativa. "
        "Termos com OR extremo (separação perfeita) foram excluídos para preservar a escala."
    )


def _render_magistrados(res):
    mag = res.get("magistrados")
    if mag is None:
        st.info("Dados de magistrado não disponíveis ou N insuficiente.")
        return
    if "status" in mag:
        st.info(f"Magistrados insuficientes para o teste chi-quadrado (N={mag.get('n', 0)} com ≥ 20 casos, mínimo: 3).")
        return

    st.markdown("### Há diferenças significativas entre magistrados?")
    st.caption(
        "Testa se a variação na taxa de procedência entre juízes é maior do que o esperado pelo acaso "
        "(teste chi-quadrado de independência). Somente magistrados com ≥ 20 casos incluídos."
    )

    chi2_v = mag["chi2"]
    p = mag["p"]
    n_mag = mag["n_magistrados"]
    stars = _estrela_inf(p)
    mag_df = mag["mag_df"]

    if p < 0.05:
        st.error(
            f"⚠️ **Variação estatisticamente significativa** entre magistrados detectada  "
            f"(χ² = {chi2_v:.1f}, p = {p:.4f}{stars}, {n_mag} magistrados). "
            "O resultado do processo pode variar conforme o juiz — vale considerar isso na estratégia."
        )
    else:
        st.success(
            f"✅ **Sem variação significativa** entre magistrados  "
            f"(χ² = {chi2_v:.1f}, p = {p:.4f}{stars}, {n_mag} magistrados). "
            "Os resultados parecem relativamente uniformes entre os juízes desta amostra."
        )

    col1, col2 = st.columns([1, 1])
    with col1:
        fig = px.histogram(
            mag_df, x="taxa", nbins=15,
            title=f"Distribuição da taxa de procedência por magistrado (N={n_mag})",
            labels={"taxa": "Taxa de procedência (%)"},
            color_discrete_sequence=["#6366f1"],
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("**Top 10 — maior taxa de procedência**")
        top10 = mag_df.head(10)[["magistrado", "taxa", "n"]].copy()
        top10.columns = ["Magistrado", "% Procedente", "N casos"]
        st.dataframe(top10, use_container_width=True, hide_index=True)

    with st.expander("Bottom 10 — menor taxa de procedência"):
        bot10 = mag_df.tail(10).sort_values("taxa")[["magistrado", "taxa", "n"]].copy()
        bot10.columns = ["Magistrado", "% Procedente", "N casos"]
        st.dataframe(bot10, use_container_width=True, hide_index=True)


def render_aba_inferencial(ia_df):
    """Orquestra a Tab 3 — Modelos Inferenciais."""
    if ia_df is None:
        st.info(
            "📐 Nenhuma análise IA rodada nesta sessão.\n\n"
            "Vá para a aba **📊 Escopo & Regex**, role até **🚀 Rodar análise com IA** "
            "e processe ao menos 50–100 casos. Os modelos inferenciais precisam dos campos extraídos pela IA."
        )
        return

    ia_hash = hashlib.md5(
        str(sorted(ia_df["id_processo"].tolist())).encode("utf-8", errors="replace")
    ).digest()
    st.session_state["_df_modelos_inferencial"] = ia_df

    with st.spinner("Ajustando modelos inferenciais…"):
        res = rodar_modelos_inferenciais(ia_hash)

    if res.get("status") in ("sem_dados", "insuficiente"):
        st.warning(
            f"Amostra insuficiente para os modelos (N={res.get('n', 0)} casos com contato prévio identificado). "
            "Processe mais casos pela IA (recomendado: ≥ 50)."
        )
        return

    st.header("Modelo A — Efeito do Contato Prévio na Vitória Judicial")
    _render_modelo_a(res)
    st.divider()

    st.header("Heterogeneidade por Tipo de Ação")
    _render_modelo_a_inter(res)
    st.divider()

    st.header("Modelo B — Efeito do Contato Prévio nos Danos Morais")
    _render_modelo_b(res)
    st.divider()

    st.header("Modelo C — Perfil dos Casos com Contato Prévio")
    _render_modelo_c(res)
    st.divider()

    st.header("Heterogeneidade entre Magistrados")
    _render_magistrados(res)


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    st.title("⚖️ Dashboard Jurídico — Petições Iniciais Itaú")

    caminho_local = _localizar_csv_local()
    csv_source = None
    origem_label = None

    if caminho_local is not None:
        csv_source = caminho_local
        origem_label = f"local::{caminho_local.name}"
    else:
        st.info(
            f"📂 Arquivo **`{CSV_FILENAME}`** não encontrado em `dashboard/` ou na raiz do projeto. "
            "Faça o upload abaixo para iniciar a análise. O arquivo fica apenas em memória nesta sessão."
        )
        uploaded = st.file_uploader(
            "Selecione o CSV da base de processos",
            type=["csv"],
            accept_multiple_files=False,
            help="Mesmo formato esperado pelo notebook (UTF-8, com a coluna `decisao`).",
        )
        if uploaded is None:
            st.stop()
        csv_source = uploaded
        origem_label = f"upload::{uploaded.name}::{uploaded.size}"

    with st.spinner("Carregando dataset e rodando regex…"):
        try:
            df = load_data(_ler_bytes_csv(csv_source), origem_label)
        except Exception as e:
            st.error(f"Falha ao processar o CSV: {e}")
            st.stop()

    geojson_data = load_geojson()
    comarca_map = build_comarca_to_geocodigo(geojson_data)

    # ── Sidebar global ──────────────────────────────────────────────────────
    st.sidebar.header("🔍 Filtros globais")
    st.sidebar.caption(
        "Filtros aplicam às duas abas. Aba IA mostra apenas casos que passaram pela IA "
        "nesta sessão **e** que ainda casam com os filtros."
    )

    nivel_opts = {1: "Dentro (L1)", 2: "Incerto (L2)", 3: "Fora (L3)"}
    niveis_sel = st.sidebar.multiselect(
        "Nível de confiança regex",
        options=list(nivel_opts.keys()),
        format_func=lambda x: nivel_opts[x],
    )

    rito_opts = sorted(df["rito_processual_regex"].dropna().unique().tolist())
    rito_sel = st.sidebar.multiselect("Rito processual", rito_opts)

    jg_opts = sorted(df["justica_gratuita_regex"].dropna().unique().tolist())
    jg_sel = st.sidebar.multiselect("Justiça gratuita", jg_opts)

    resultado_opts = sorted(df["resultado_julgamento_regex"].dropna().unique().tolist())
    resultado_sel = st.sidebar.multiselect("Resultado julgamento", resultado_opts)

    st.sidebar.divider()
    if "assunto" in df.columns:
        assunto_opts = sorted(df["assunto"].dropna().unique().tolist())
        assunto_sel = st.sidebar.multiselect("Assunto", assunto_opts)
    else:
        assunto_sel = []

    if "comarca" in df.columns:
        comarca_opts = sorted(df["comarca"].dropna().unique().tolist())
        comarca_sel = st.sidebar.multiselect("Comarca", comarca_opts)
    else:
        comarca_sel = []

    df_filtered = df.copy()
    if niveis_sel:
        df_filtered = df_filtered[df_filtered["nivel_confianca_regex"].isin(niveis_sel)]
    if rito_sel:
        df_filtered = df_filtered[df_filtered["rito_processual_regex"].isin(rito_sel)]
    if jg_sel:
        df_filtered = df_filtered[df_filtered["justica_gratuita_regex"].isin(jg_sel)]
    if resultado_sel:
        df_filtered = df_filtered[df_filtered["resultado_julgamento_regex"].isin(resultado_sel)]
    if assunto_sel:
        df_filtered = df_filtered[df_filtered["assunto"].isin(assunto_sel)]
    if comarca_sel:
        df_filtered = df_filtered[df_filtered["comarca"].isin(comarca_sel)]

    st.sidebar.divider()
    st.sidebar.metric("Casos totais", f"{len(df):,}".replace(",", "."))
    st.sidebar.metric("Após filtros", f"{len(df_filtered):,}".replace(",", "."))

    # ── Abas ────────────────────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs(["📊 Escopo & Regex", "🤖 Análise IA", "📐 Modelos Inferenciais"])

    with tab1:
        render_distribuicao_escopo(df_filtered)
        st.divider()
        df_dentro = df_filtered[df_filtered["nivel_confianca_regex"] == 1]
        render_graficos_regex_dentro(df_dentro, geojson_data, comarca_map)
        render_aviso_danos_regex(df_dentro)
        render_painel_pipeline_ia(df_filtered)

    with tab2:
        ia_df = st.session_state.get("ia_results")
        if ia_df is not None:
            # Filtra os resultados IA pelos mesmos filtros globais
            idx_filtered = set(df_filtered.index)
            ia_filtrado = ia_df[ia_df["_idx_orig"].isin(idx_filtered)] if "_idx_orig" in ia_df.columns else ia_df
            if ia_filtrado.empty:
                st.warning(
                    "Filtros globais não casam com nenhum caso analisado pela IA. "
                    "Ajuste os filtros no sidebar."
                )
            else:
                st.session_state["_ia_results_filtered_view"] = ia_filtrado
                # Override temporário para a função render_aba_ia ler a versão filtrada
                ia_original = st.session_state["ia_results"]
                st.session_state["ia_results"] = ia_filtrado
                try:
                    render_aba_ia(geojson_data, comarca_map)
                finally:
                    st.session_state["ia_results"] = ia_original
        else:
            render_aba_ia(geojson_data, comarca_map)

    with tab3:
        ia_df_full = st.session_state.get("ia_results")
        render_aba_inferencial(ia_df_full)


if __name__ == "__main__":
    main()
