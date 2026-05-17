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

import json
import os
import unicodedata
from datetime import datetime
from pathlib import Path

import nest_asyncio
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

import ia_pipeline
from regex_otimizado import classificar_escopo_regex, processar_batch

nest_asyncio.apply()
load_dotenv()

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
def load_data():
    df = pd.read_csv("dataset_clinica_20261.csv", encoding="utf-8")
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


@st.cache_data(show_spinner=False)
def load_geojson():
    with open(Path("geodata") / "SP.json", "r", encoding="utf-8") as f:
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
    """Agrega por comarca: total de casos e % dentro do escopo (L1)."""
    agg = df.groupby("comarca").agg(
        total_processos=("id_processo", "count"),
    ).reset_index()

    def _pct_dentro(group):
        total = len(group)
        if total == 0:
            return 0.0
        dentro = (group["nivel_confianca_regex"] == 1).sum()
        return dentro / total * 100

    pct = df.groupby("comarca").apply(_pct_dentro, include_groups=False).reset_index()
    pct.columns = ["comarca", "pct_dentro"]

    agg = agg.merge(pct, on="comarca", how="left")
    agg["comarca_norm"] = agg["comarca"].apply(_normalizar_nome)
    agg["geocodigo"] = agg["comarca_norm"].map(comarca_map)
    agg["pct_dentro"] = agg["pct_dentro"].round(1)
    return agg


# ════════════════════════════════════════════════════════════════════════════
# CONFIGS DE MAPA
# ════════════════════════════════════════════════════════════════════════════

MAPA_REGEX_CONFIGS = {
    "pct_dentro": {
        "col": "pct_dentro",
        "title": "% de casos DENTRO do escopo por comarca",
        "label": "% Dentro",
        "scale": [[0.0, "#022c22"], [0.3, "#059669"], [0.7, "#34d399"], [1.0, "#f0fdf4"]],
        "fmt": ".1f",
        "zmax_quantile": 0.90,
    },
    "total_processos": {
        "col": "total_processos",
        "title": "Total de processos por comarca",
        "label": "Total",
        "scale": [[0.0, "#1e1b4b"], [0.3, "#4338ca"], [0.7, "#818cf8"], [1.0, "#eef2ff"]],
        "fmt": ",.0f",
        "zmax_quantile": 0.85,
    },
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

    st.markdown("### Mapas Coropléticos — Estado de SP")
    metrica = st.selectbox(
        "Métrica:",
        list(MAPA_REGEX_CONFIGS.keys()),
        format_func=lambda x: MAPA_REGEX_CONFIGS[x]["title"],
        key="metrica_mapa_regex",
    )
    df_mapa = agregar_mapa_regex(df_dentro, comarca_map)
    render_mapa_sp(df_mapa, geojson_data, MAPA_REGEX_CONFIGS[metrica], key_suffix="regex")


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
            fig = px.histogram(
                morais_pos, x="valor_danos_morais_regex", nbins=20,
                title="Danos morais (regex, >R$0)",
                labels={"valor_danos_morais_regex": "Valor (R$)"},
                color_discrete_sequence=["#9333ea"],
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    with col2:
        if not materiais_pos.empty:
            fig = px.histogram(
                materiais_pos, x="valor_danos_materiais_regex", nbins=20,
                title="Danos materiais (regex, >R$0)",
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
            ["L1 + L2 (incluir incertos)", "Apenas L1 (certos)"],
            index=0,
            help=(
                "**L1 + L2**: IA primeiro reclassifica os incertos (Nível 2) com LLM para "
                "decidir dentro/fora; depois extrai campos dos confirmados dentro. **Apenas L1**: "
                "pula L2 e extrai campos só dos casos já confirmados dentro pelo regex."
            ),
        )
        modo = "l1_plus_l2" if "L1 + L2" in modo_label else "l1_only"

    with col_b:
        if modo == "l1_only":
            n_disponivel = int((df_filtered["nivel_confianca_regex"] == 1).sum())
        else:
            n_disponivel = int(df_filtered["nivel_confianca_regex"].isin([1, 2]).sum())
        if n_disponivel == 0:
            st.info(f"Nenhum caso elegível ({modo_label}) nos filtros atuais.")
            return
        n_max = min(n_disponivel, 500)
        n_amostra = st.slider(
            f"Casos a analisar (máx {n_max}, há {n_disponivel:,} disponíveis):".replace(",", "."),
            min_value=1, max_value=n_max,
            value=min(50, n_max),
            help="Amostragem aleatória com seed fixa (random_state=42), reproduzível entre execuções.",
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
        if not morais_pos.empty:
            fig = px.histogram(
                morais_pos, x="valor_danos_morais_ia", nbins=20,
                title="Danos morais condenados (IA, >R$0)",
                color_discrete_sequence=["#9333ea"],
            )
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
        fig = px.box(
            morais_pos, x="contato_previo_banco_ia", y="valor_danos_morais_ia",
            color="contato_previo_banco_ia",
            title="Danos morais por contato prévio (>R$0)",
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
# MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    st.title("⚖️ Dashboard Jurídico — Petições Iniciais Itaú")

    with st.spinner("Carregando dataset e rodando regex…"):
        try:
            df = load_data()
        except FileNotFoundError:
            st.error("Arquivo `dataset_clinica_20261.csv` não encontrado na raiz do projeto.")
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
    tab1, tab2 = st.tabs(["📊 Escopo & Regex", "🤖 Análise IA"])

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


if __name__ == "__main__":
    main()
