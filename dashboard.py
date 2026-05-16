import streamlit as st
import pandas as pd
import re
import unicodedata
import os
import json
import asyncio
import hashlib
from pathlib import Path
from dotenv import load_dotenv
import plotly.express as px
import plotly.graph_objects as go
from openai import AsyncOpenAI
import nest_asyncio

nest_asyncio.apply()
load_dotenv()

st.set_page_config(page_title="Dashboard Jurídico", layout="wide")

# --- REGEX FUNCTIONS ---
def _norm(texto):
    if pd.isna(texto): return ""
    t = unicodedata.normalize("NFKD", str(texto).lower()).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"\s+", " ", t)

_re_jg = re.compile(r"\b(justica\s+gratuita|gratuidade\s+da\s+justica|beneficio\s+da\s+justica\s+gratuita|assistencia\s+judiciaria\s+gratuita|\bajg\b)", re.IGNORECASE)
def regex_justica_gratuita(t): return "sim" if _re_jg.search(_norm(t)) else "não"

_re_juizado = re.compile(r"\b(juizado\s+especial|lei\s*(n[o°º]?\s*)?9\.099)\b", re.IGNORECASE)
_re_comum   = re.compile(r"\b(procedimento\s+comum|rito\s+ordinario|procedimento\s+ordinario)\b", re.IGNORECASE)
def regex_rito_processual(t):
    n = _norm(t)
    if _re_juizado.search(n): return "Juizado Especial"
    if _re_comum.search(n):   return "Procedimento Comum"
    return "não identificado"

_re_revisao  = re.compile(r"\b(revisao\s+contratual|revisao\s+de\s+juros|juros\s+(remuneratorios|abusivos)|capitalizacao|anatocismo|cet\s+abusivo)\b", re.IGNORECASE)
_re_emprest  = re.compile(r"\b(emprestimo\s+(nao\s+)?(solicitado|reconhecido|autorizado)|credito\s+(nao\s+)?(solicitado|reconhecido|autorizado)|contrato\s+nao\s+reconhecido|descontos?\s+nao\s+(autorizados?|reconhecidos?))\b", re.IGNORECASE)
_re_cobranca = re.compile(r"\b(cobranca\s+indevida|desconto\s+indevido|tarifa\s+indevida|lancamento\s+indevido|cobrado\s+indevidamente|valor\s+cobrado\s+a\s+maior)\b", re.IGNORECASE)
_re_fraude   = re.compile(r"\b(vitima\s+de\s+(fraude|golpe|estelionato)|transac[ao]\w*\s+fraudulent\w*|operac[ao]\w*\s+fraudulent\w*|fraude\s+bancar\w*|sofreu\s+(fraude|golpe)|clonac[ao]\w*\s+de\s+cart[ao]|phishing)\b", re.IGNORECASE)
def regex_tipo_acao(t):
    n = _norm(t)
    if _re_revisao.search(n):  return "revisão contratual"
    if _re_emprest.search(n):  return "empréstimo não reconhecido"
    if _re_cobranca.search(n): return "cobrança indevida"
    if _re_fraude.search(n):   return "fraude ou golpe"
    return "outro"

_re_contato = re.compile(r"\b(sac|ouvidoria|procon|reclame\s+aqui|reclamac[ao]\s+(administrativa|junto\s+ao?|no|na)\s*(banco|sac|procon|ouvidoria)?\b|protocolo\s+de\s+(atendimento|reclamac[ao])|procurou\s+o\s+banco|acionou\s+o\s+banco|comunicou\s+(ao|o)\s+banco|tentou\s+(resolver|contato\s+com\s+o\s+banco|solucionar)\s+(junto\s+ao?\s+banco|o\s+problema))\b", re.IGNORECASE)
def regex_contato_previo(t): return "sim" if _re_contato.search(_norm(t)) else "não"

_re_reclame   = re.compile(r"\breclame\s+aqui\b", re.IGNORECASE)
_re_procon    = re.compile(r"\b(procon|decon|programa\s+de\s+protecao\s+e\s+defesa\s+do\s+consumidor)\b", re.IGNORECASE)
_re_ouvidoria = re.compile(r"\bouvidoria\b", re.IGNORECASE)
_re_sac       = re.compile(r"\b(sac|servico\s+de\s+atendimento\s+ao\s+consumidor)\b", re.IGNORECASE)
_re_agencia   = re.compile(r"(compareceu|dirigiu.se|foi\s+(ate|a\s+uma?)|entrou\s+em\s+contato\s+na|presencialmente\s+na|atendimento\s+presencial)\s+(a\s+)?agencia", re.IGNORECASE)
def regex_canal_contato(t):
    n = _norm(t)
    if _re_reclame.search(n):   return "Reclame Aqui"
    if _re_procon.search(n):    return "Procon"
    if _re_ouvidoria.search(n): return "Ouvidoria"
    if _re_sac.search(n):       return "SAC"
    if _re_agencia.search(n):   return "Agência"
    return "não identificado"

def regex_mencao_reclame(t): return "sim" if _re_reclame.search(_norm(t)) else "não"

_re_bo = re.compile(r"\b(boletim\s+de\s+ocorrencia|registro\s+(de\s+)?(ocorrencia|policial)|B\.O\.\s*n[o°º]?)\b", re.IGNORECASE)
def regex_boletim(t): return "sim" if _re_bo.search(_norm(t)) else "não"

def regex_resultado(t):
    n = _norm(t)
    m = re.search(r"\b(?:julgo|homologo)\b(.{0,220})", n)
    trecho = m.group(1) if m else n[:220]
    if re.search(r"parcialmente\s+procedente", trecho): return "parcialmente procedente"
    if re.search(r"improcedent(?:e|es)", trecho): return "improcedente"
    if re.search(r"procedent(?:e|es)", trecho): return "procedente"
    if re.search(r"extint[ao].*sem\s+resolucao\s+do\s+merito|sem\s+analise\s+do\s+merito", trecho): return "extinto"
    if re.search(r"extint[ao]", trecho): return "extinto"
    if re.search(r"homologo.*desistencia|desistencia\s+da\s+acao", trecho): return "extinto"
    if re.search(r"indefiro\s+a\s+peticao\s+inicial", trecho): return "extinto"
    return "não identificado"

_re_culpa_compart  = re.compile(r"\b(culpa\s+(concorrente|compartilhada|reciproca)|concorrencia\s+de\s+culpas?)\b", re.IGNORECASE)
_re_culpa_terceiro = re.compile(r"\b(culpa\s+(exclusiva\s+)?de\s+terceiro|fraude\s+(praticada\s+)?por\s+terceiro|estelionato\s+(praticado\s+)?por\s+terceiro)\b", re.IGNORECASE)
_re_culpa_banco    = re.compile(r"\b(falha\s+(na\s+)?prestacao\s+de\s+servicos?|responsabilidade\s+(civil\s+)?do\s+(banco|requerido|reu)\s+(e\s+)?reconhecida|condeno\s+o\s+(banco|requerido|reu)|dano\s+causado\s+pelo\s+(banco|requerido|reu))\b", re.IGNORECASE)
_re_culpa_cons     = re.compile(r"\b(culpa\s+(exclusiva\s+)?d[oa]\s+(autor[a]?|requerente|consumidor[a]?)|ausencia\s+de\s+(conduta\s+ilicita|falha\s+na\s+prestacao|nexo\s+causal)|nao\s+h[ao]\s+(falha|vicio|defeito)\s+na\s+prestacao)\b", re.IGNORECASE)
def regex_culpa(t):
    n = _norm(t)
    if _re_culpa_compart.search(n):  return "compartilhada"
    if _re_culpa_terceiro.search(n): return "terceiro"
    if _re_culpa_banco.search(n):    return "banco"
    if _re_culpa_cons.search(n):     return "consumidor"
    return "não identificado"

def _extrair_valor(texto, campo):
    t = str(texto)
    for pos in [m.start() for m in re.finditer(campo, t, re.IGNORECASE)]:
        trecho = t[max(0, pos - 150): pos + 150]
        for v in re.findall(r"R\$\s*([\d.,]+)", trecho):
            try:
                return float(v.replace(".", "").replace(",", "."))
            except ValueError:
                pass
    return 0.0

def regex_valor_morais(t):    return _extrair_valor(t, r"danos\s+morais")
def regex_valor_materiais(t): return _extrair_valor(t, r"danos\s+materiais")

# --- DATA PROCESSING ---
def corrigir_mojibake(valor):
    if isinstance(valor, str) and ("Ã" in valor or "Â" in valor):
        try:
            return valor.encode("latin1").decode("utf-8")
        except UnicodeError:
            return valor
    return valor

@st.cache_data
def load_data():
    df = pd.read_csv("dataset_clinica_20261.csv", encoding="utf-8")
    colunas_texto = df.select_dtypes(include="object").columns
    for col in colunas_texto:
        df[col] = df[col].apply(corrigir_mojibake)
    if "magistrado" in df.columns:
        df["magistrado"] = df["magistrado"].astype(str).str.strip().str.upper()
    
    decisao = df["decisao"].fillna("")
    df["justica_gratuita_regex"]      = decisao.apply(regex_justica_gratuita)
    df["rito_processual_regex"]       = decisao.apply(regex_rito_processual)
    df["tipo_acao_regex"]             = decisao.apply(regex_tipo_acao)
    df["contato_previo_banco_regex"]  = decisao.apply(regex_contato_previo)
    df["canal_contato_regex"]         = decisao.apply(regex_canal_contato)
    df["mencao_reclame_aqui_regex"]   = decisao.apply(regex_mencao_reclame)
    df["boletim_de_ocorrencia_regex"] = decisao.apply(regex_boletim)
    df["resultado_julgamento_regex"]  = decisao.apply(regex_resultado)
    df["culpa_atribuida_regex"]       = decisao.apply(regex_culpa)
    df["valor_danos_morais_regex"]    = decisao.apply(regex_valor_morais)
    df["valor_danos_materiais_regex"] = decisao.apply(regex_valor_materiais)
    return df

# --- GEOJSON & MAP FUNCTIONS ---
GEOJSON_PATH = Path("geodata") / "SP.json"

@st.cache_data
def load_geojson():
    with open(GEOJSON_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def _normalizar_nome(nome):
    if pd.isna(nome): return ""
    t = unicodedata.normalize("NFKD", str(nome)).encode("ascii", "ignore").decode("ascii")
    return t.strip().lower()

@st.cache_data
def build_comarca_to_geocodigo(geojson_data):
    mapping = {}
    for feature in geojson_data["features"]:
        nome = feature["properties"]["NOME"]
        geocodigo = feature["properties"]["GEOCODIGO"]
        mapping[_normalizar_nome(nome)] = geocodigo
    return mapping

def agregar_mapa_data(df, col_resultado, col_morais, comarca_map):
    agg = df.groupby("comarca").agg(
        total_processos=("id_processo", "count"),
        valor_medio_morais=(col_morais, "mean"),
    ).reset_index()
    
    def _calc_taxa(group):
        total = len(group)
        if total == 0: return 0.0
        favoraveis = group[col_resultado].isin(["procedente", "parcialmente procedente"]).sum()
        return (favoraveis / total) * 100
    
    taxa = df.groupby("comarca").apply(_calc_taxa, include_groups=False).reset_index()
    taxa.columns = ["comarca", "taxa_procedencia"]
    
    agg = agg.merge(taxa, on="comarca", how="left")
    agg["comarca_norm"] = agg["comarca"].apply(_normalizar_nome)
    agg["geocodigo"] = agg["comarca_norm"].map(comarca_map)
    agg["valor_medio_morais"] = agg["valor_medio_morais"].round(2)
    agg["taxa_procedencia"] = agg["taxa_procedencia"].round(1)
    return agg

def render_mapa_sp(df_mapa, geojson_data, metrica, key_suffix=""):
    df_plot = df_mapa[df_mapa["geocodigo"].notna()].copy()
    if df_plot.empty:
        st.warning("Nenhuma comarca pôde ser mapeada para os municípios de SP.")
        return
    
    metric_config = {
        "valor_morais": {
            "col": "valor_medio_morais",
            "title": "Valor Médio de Danos Morais por Comarca",
            "label": "Valor Médio (R$)",
            "scale": [[0.0, '#2e1065'], [0.3, '#7e22ce'], [0.7, '#d946ef'], [1.0, '#fdf4ff']],
            "fmt": ",.2f",
            "zmax": df_plot[df_plot["valor_medio_morais"] > 0]["valor_medio_morais"].quantile(0.80) if not df_plot[df_plot["valor_medio_morais"] > 0].empty else 100
        },
        "taxa_procedencia": {
            "col": "taxa_procedencia",
            "title": "Taxa de Procedência por Comarca",
            "label": "Procedência (%)",
            "scale": [[0.0, '#022c22'], [0.3, '#059669'], [0.7, '#34d399'], [1.0, '#f0fdf4']],
            "fmt": ".1f",
            "zmax": df_plot[df_plot["taxa_procedencia"] > 0]["taxa_procedencia"].quantile(0.85) if not df_plot[df_plot["taxa_procedencia"] > 0].empty else 100
        },
    }
    cfg = metric_config[metrica]

    all_geocodigos = [feat["properties"]["GEOCODIGO"] for feat in geojson_data["features"]]
    all_nomes = [feat["properties"]["NOME"] for feat in geojson_data["features"]]

    base = go.Choropleth(
        geojson=geojson_data,
        locations=all_geocodigos,
        featureidkey="properties.GEOCODIGO",
        z=[0] * len(all_geocodigos),
        colorscale=[[0, "#1f2937"], [1, "#1f2937"]],
        showscale=False,
        marker_line_color="rgba(255,255,255,0.35)",
        marker_line_width=0.4,
        hovertext=all_nomes,
        hovertemplate="<b>%{hovertext}</b><br><i>sem processos</i><extra></extra>",
    )

    data_layer = go.Choropleth(
        geojson=geojson_data,
        locations=df_plot["geocodigo"],
        featureidkey="properties.GEOCODIGO",
        z=df_plot[cfg["col"]],
        zmax=cfg["zmax"],
        colorscale=cfg["scale"],
        marker_line_color="white",
        marker_line_width=0.9,
        customdata=df_plot[["comarca", "total_processos", "valor_medio_morais", "taxa_procedencia"]].values,
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Processos: %{customdata[1]:,}<br>"
            "Danos morais médios: R$ %{customdata[2]:,.2f}<br>"
            "Procedência: %{customdata[3]:.1f}%"
            "<extra></extra>"
        ),
        colorbar=dict(title=cfg["label"], thickness=14, len=0.75, outlinewidth=0),
    )

    fig = go.Figure(data=[base, data_layer])
    fig.update_geos(fitbounds="geojson", visible=False, projection_type="mercator", bgcolor="rgba(0,0,0,0)")
    fig.update_layout(
        title=dict(text=cfg["title"], x=0.02, xanchor="left"),
        margin={"r": 0, "t": 50, "l": 0, "b": 0},
        height=620,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    
    st.plotly_chart(fig, use_container_width=True, key=f"mapa_{metrica}_{key_suffix}")

# --- OPENAI ASYNC ---
CACHE_PATH = "cache/cache_openai.json"

def _carregar_cache():
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def _salvar_cache(cache):
    os.makedirs("cache", exist_ok=True)
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)

def _hash_decisao(texto): return hashlib.md5(texto.encode("utf-8")).hexdigest()

_campos_padrao = {
    "justica_gratuita": "não", "rito_processual": "não identificado", "tipo_acao": "não identificado",
    "contato_previo_banco": "não", "canal_contato": "não identificado", "mencao_reclame_aqui": "não",
    "boletim_de_ocorrencia": "não", "resultado_julgamento": "não identificado", "culpa_atribuida": "não identificado",
    "valor_danos_morais": 0.0, "valor_danos_materiais": 0.0,
}

template_prompt = """Analise a decisão judicial abaixo e extraia as variáveis indicadas.

VARIÁVEIS:
1. justica_gratuita: ("sim" ou "não")
2. rito_processual: ("Juizado Especial" ou "Procedimento Comum")
3. tipo_acao: ("fraude ou golpe" / "cobrança indevida" / "empréstimo não reconhecido" / "revisão contratual" / "outro")
4. contato_previo_banco: ("sim" ou "não") — houve contato com o banco antes do ajuizamento?
5. canal_contato: ("SAC" / "Ouvidoria" / "Procon" / "Reclame Aqui" / "Agência" / "não identificado")
6. mencao_reclame_aqui: ("sim" ou "não")
7. boletim_de_ocorrencia: ("sim" ou "não")
8. resultado_julgamento: ("procedente" / "improcedente" / "parcialmente procedente" / "extinto")
9. culpa_atribuida: ("banco" / "consumidor" / "terceiro" / "compartilhada" / "não identificado")
10. valor_danos_morais: número em reais (0.0 se não condenado)
11. valor_danos_materiais: número em reais (0.0 se não condenado)

REGRAS DE PREENCHIMENTO:
- justica_gratuita, contato_previo_banco, mencao_reclame_aqui, boletim_de_ocorrencia:
  retorne "não" se não houver menção explícita no texto. Nunca use "não identificado" nesses campos.
- canal_contato: retorne o canal apenas se o texto mencionar explicitamente SAC, Ouvidoria, Procon, Reclame Aqui ou agência.
- culpa_atribuida:
    • resultado=improcedente → "consumidor"
    • resultado=procedente ou parcialmente procedente → "banco"
    • resultado=extinto → "não identificado"

Retorne exatamente este JSON:
{{"justica_gratuita":"...","rito_processual":"...","tipo_acao":"...","contato_previo_banco":"...","canal_contato":"...","mencao_reclame_aqui":"...","boletim_de_ocorrencia":"...","resultado_julgamento":"...","culpa_atribuida":"...","valor_danos_morais":0.0,"valor_danos_materiais":0.0}}

Decisão:
{decisao}"""

prompt_sistema = "Você é um analista jurídico especializado. Extraia as informações. Responda APENAS com JSON válido."

async def _chamar_decisao_async(idx, texto, semaforo, async_client, max_tentativas=4):
    prompt = template_prompt.format(decisao=texto)
    async with semaforo:
        for tentativa in range(max_tentativas):
            try:
                resposta = await async_client.chat.completions.create(
                    model="gpt-4.1", messages=[{"role": "system", "content": prompt_sistema}, {"role": "user", "content": prompt}],
                    temperature=0, response_format={"type": "json_object"}
                )
                dados = json.loads(resposta.choices[0].message.content.strip())
                for campo in ("valor_danos_morais", "valor_danos_materiais"):
                    dados[campo] = float(dados.get(campo) or 0.0)
                return idx, {**_campos_padrao, **dados}
            except Exception as e:
                if "429" in str(e) and tentativa < max_tentativas - 1:
                    await asyncio.sleep(2 ** tentativa)
                    continue
                return idx, {**_campos_padrao, "resultado_julgamento": f"ERRO: {str(e)}"}

async def processar_async_batch(decisoes, progress_bar):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key: return {"error": "OPENAI_API_KEY não definida no .env"}
    async_client = AsyncOpenAI(api_key=api_key)
    cache_local = _carregar_cache()
    
    resultados_final = [None] * len(decisoes)
    indices_api = []
    
    for i, texto in enumerate(decisoes):
        h = _hash_decisao(texto)
        if h in cache_local: resultados_final[i] = cache_local[h]
        else: indices_api.append(i)
            
    semaforo = asyncio.Semaphore(5)
    tasks = [_chamar_decisao_async(i, decisoes[i], semaforo, async_client) for i in indices_api]
    
    pendentes_flush = concluidas = 0
    total = len(tasks)
    
    if total > 0:
        for coro in asyncio.as_completed(tasks):
            idx, resultado = await coro
            resultados_final[idx] = resultado
            if not str(resultado.get("resultado_julgamento", "")).startswith("ERRO"):
                cache_local[_hash_decisao(decisoes[idx])] = resultado
                pendentes_flush += 1
            if pendentes_flush >= 10:
                _salvar_cache(cache_local)
                pendentes_flush = 0
            concluidas += 1
            progress_bar.progress(concluidas / total)
    if pendentes_flush > 0: _salvar_cache(cache_local)
    return resultados_final

def run_ai_extraction(decisoes):
    progress_bar = st.progress(0)
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(processar_async_batch(decisoes, progress_bar))


# --- EDA REUSABLE FUNCTIONS ---
def padronizar_canal(c):
    c = str(c).lower().strip()
    if 'whatsapp' in c or 'whats' in c: return 'whatsapp'
    if 'e-mail' in c or 'email' in c: return 'e-mail'
    if 'telefone' in c or 'sac' in c or '0800' in c: return 'telefone/sac'
    if 'procon' in c: return 'procon'
    if 'consumidor.gov' in c: return 'consumidor.gov'
    if 'reclame aqui' in c or 'reclameaqui' in c: return 'reclame aqui'
    if 'agência' in c or 'agencia' in c or 'presencial' in c: return 'agência presencial'
    if c in ['não identificado', 'nan', 'não aplicável']: return 'não identificado'
    return 'outro'

def limpar_e_padronizar_dados(df, suffix):
    cols_map = {
        f"valor_danos_morais_{suffix}": "valor_danos_morais",
        f"valor_danos_materiais_{suffix}": "valor_danos_materiais",
        f"resultado_julgamento_{suffix}": "resultado_julgamento",
        f"contato_previo_banco_{suffix}": "contato_previo_banco",
        f"canal_contato_{suffix}": "canal_contato",
        f"tipo_acao_{suffix}": "tipo_acao",
        f"culpa_atribuida_{suffix}": "culpa_atribuida",
        f"boletim_de_ocorrencia_{suffix}": "boletim_de_ocorrencia"
    }
    
    df_eda = df.copy()
    for old_col, new_col in cols_map.items():
        if old_col in df_eda.columns:
            df_eda[new_col] = df_eda[old_col]
            
    valores_positivos = df_eda[df_eda['valor_danos_morais'] > 0]['valor_danos_morais']
    if not valores_positivos.empty:
        q1_pos = valores_positivos.quantile(0.25)
        q3_pos = valores_positivos.quantile(0.75)
        iqr_pos = q3_pos - q1_pos
        limite_superior = q3_pos + 1.5 * iqr_pos
        if pd.isna(limite_superior):
            limite_superior = 50000
    else:
        limite_superior = 0
        
    outliers = df_eda[df_eda['valor_danos_morais'] > limite_superior]
    df_clean = df_eda[df_eda['valor_danos_morais'] <= limite_superior].copy()
    
    df_clean['canal_contato'] = df_clean['canal_contato'].apply(padronizar_canal)
    freq_canais = df_clean[df_clean['canal_contato'] != 'não identificado']['canal_contato'].value_counts()
    
    if len(df_clean[df_clean['canal_contato'] != 'não identificado']) > 0:
        limite_minimo = len(df_clean[df_clean['canal_contato'] != 'não identificado']) * 0.005
        canais_relevantes = freq_canais[freq_canais >= max(2, limite_minimo)].index.tolist()
        df_clean['canal_contato'] = df_clean['canal_contato'].apply(
            lambda x: x if x in canais_relevantes or x == 'não identificado' else 'outro'
        )
        
    return df_clean, limite_superior, len(outliers)

def renderizar_eda_completa(df, suffix, geojson_data, comarca_map):
    df_clean, lim_sup, q_outliers = limpar_e_padronizar_dados(df, suffix)
    
    if df_clean.empty:
        st.warning("A base filtrada está vazia. Altere os filtros para visualizar os gráficos.")
        return
        
    # ==============================================================
    # PARTE 1: VISÃO GERAL
    # ==============================================================
    st.header("Parte 1: Visão Geral da Base de Dados")
    
    colA, colB = st.columns(2)
    with colA:
        df_plot_res = df_clean[df_clean['resultado_julgamento'] != 'outro']
        if not df_plot_res.empty:
            res_counts = df_plot_res['resultado_julgamento'].value_counts().reset_index()
            res_counts.columns = ['Resultado', 'Quantidade']
            fig_res = px.bar(res_counts, x='Quantidade', y='Resultado', orientation='h',
                             title="Distribuição dos Resultados dos Julgamentos",
                             color='Resultado', color_discrete_sequence=px.colors.qualitative.Pastel)
            fig_res.update_layout(showlegend=False)
            st.plotly_chart(fig_res, use_container_width=True)
            
    with colB:
        danos_morais_positivos = df_clean[df_clean['valor_danos_morais'] > 0]
        if not danos_morais_positivos.empty:
            fig_hist = px.histogram(danos_morais_positivos, x='valor_danos_morais', nbins=15,
                                    title="Distribuição dos Valores de Danos Morais (Condenações > R$ 0)",
                                    labels={'valor_danos_morais': 'Valor (R$)'},
                                    color_discrete_sequence=['#9333ea'])
            st.plotly_chart(fig_hist, use_container_width=True)

    st.markdown("### Mapas Coropléticos — Processos por Comarca (Estado de SP)")
    metrica = st.selectbox(
        "Selecione a métrica para visualizar no mapa:",
        ["valor_morais", "taxa_procedencia"],
        format_func=lambda x: {"valor_morais": "💰 Valor Médio de Morais", "taxa_procedencia": "⚖️ Taxa de Procedência (%)"}[x],
        key=f"metrica_mapa_{suffix}"
    )
    df_mapa = agregar_mapa_data(df_clean, "resultado_julgamento", "valor_danos_morais", comarca_map)
    render_mapa_sp(df_mapa, geojson_data, metrica, key_suffix=suffix)
    
    # ==============================================================
    # PARTE 2: IMPACTO DO CONTATO PRÉVIO
    # ==============================================================
    st.header("Parte 2: Impacto do Contato Prévio")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        contagens = df_clean['contato_previo_banco'].value_counts().reset_index()
        contagens.columns = ['Houve Contato?', 'Quantidade']
        fig_pie = px.pie(contagens, values='Quantidade', names='Houve Contato?',
                         title='Proporção de Contato Prévio',
                         color='Houve Contato?',
                         color_discrete_map={'sim':'#10b981', 'não':'#ef4444'})
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        contato_res = df_clean.groupby(['contato_previo_banco', 'resultado_julgamento']).size().reset_index(name='count')
        contato_res['Proporção (%)'] = contato_res.groupby('contato_previo_banco')['count'].transform(lambda x: x/x.sum() * 100)
        fig_bar_res = px.bar(contato_res, x='contato_previo_banco', y='Proporção (%)', color='resultado_julgamento',
                             title='Resultado do Julgamento por Contato Prévio (%)',
                             labels={'contato_previo_banco': 'Houve Contato Prévio?'},
                             text=contato_res['Proporção (%)'].apply(lambda x: f'{x:.1f}%'),
                             color_discrete_sequence=px.colors.qualitative.Set2)
        fig_bar_res.update_layout(barmode='stack')
        st.plotly_chart(fig_bar_res, use_container_width=True)

    if not danos_morais_positivos.empty:
        fig_box = px.box(danos_morais_positivos, x='contato_previo_banco', y='valor_danos_morais', color='contato_previo_banco',
                         title='Distribuição do Valor de Danos Morais por Contato Prévio (Apenas > R$ 0)',
                         labels={'contato_previo_banco': 'Houve Contato?', 'valor_danos_morais': 'Valor Danos Morais (R$)'},
                         color_discrete_map={'sim':'#10b981', 'não':'#ef4444'})
        st.plotly_chart(fig_box, use_container_width=True)
    
    col3, col4 = st.columns(2)
    with col3:
        tipo_contato = df_clean.groupby(['tipo_acao', 'contato_previo_banco']).size().reset_index(name='count')
        tipo_contato['Proporção (%)'] = tipo_contato.groupby('tipo_acao')['count'].transform(lambda x: x/x.sum() * 100)
        fig_tipo = px.bar(tipo_contato, x='tipo_acao', y='Proporção (%)', color='contato_previo_banco',
                          title='Contato Prévio por Tipo de Ação (%)',
                          labels={'tipo_acao': 'Tipo de Ação'},
                          text=tipo_contato['Proporção (%)'].apply(lambda x: f'{x:.0f}%'),
                          color_discrete_map={'sim':'#10b981', 'não':'#ef4444'})
        fig_tipo.update_layout(barmode='stack')
        st.plotly_chart(fig_tipo, use_container_width=True)
        
    with col4:
        culpa_contato = df_clean.groupby(['culpa_atribuida', 'contato_previo_banco']).size().reset_index(name='count')
        culpa_contato['Proporção (%)'] = culpa_contato.groupby('contato_previo_banco')['count'].transform(lambda x: x/x.sum() * 100)
        fig_culpa = px.bar(culpa_contato, x='contato_previo_banco', y='Proporção (%)', color='culpa_atribuida',
                           title='Culpa Atribuída por Contato Prévio (%)',
                           labels={'contato_previo_banco': 'Houve Contato Prévio?'},
                           text=culpa_contato['Proporção (%)'].apply(lambda x: f'{x:.0f}%'),
                           color_discrete_sequence=px.colors.qualitative.Pastel)
        fig_culpa.update_layout(barmode='stack')
        st.plotly_chart(fig_culpa, use_container_width=True)

    # ==============================================================
    # PARTE 3: ANÁLISE ESPECÍFICA POR CANAIS
    # ==============================================================
    st.header("Parte 3: Análise Específica por Canais")
    canais = df_clean[df_clean['contato_previo_banco'] != 'não']
    
    if not canais.empty:
        canais_counts = canais['canal_contato'].value_counts().reset_index()
        canais_counts.columns = ['Canal de Contato', 'Quantidade']
        fig_canais = px.bar(canais_counts, x='Quantidade', y='Canal de Contato', orientation='h',
                            title='Canais de Contato Prévio Mais Utilizados',
                            color='Quantidade', color_continuous_scale='Magma')
        st.plotly_chart(fig_canais, use_container_width=True)

        heatmap_data = pd.crosstab(canais['canal_contato'], canais['resultado_julgamento'], normalize='index') * 100
        fig_heat = px.imshow(heatmap_data, text_auto=".1f", color_continuous_scale='YlGnBu',
                             title='Probabilidade de Resultado por Canal de Contato (%)',
                             labels=dict(x="Resultado", y="Canal de Contato", color="Proporção (%)"))
        st.plotly_chart(fig_heat, use_container_width=True)
    else:
        st.info("Não há processos com contato prévio nos dados filtrados para exibir a análise de canais.")
        
    st.divider()
    st.markdown("### Processos Encontrados (Base Limpa e Padronizada)")
    st.dataframe(df_clean, use_container_width=True)

# --- UI APP MAIN ---
def main():
    st.title("⚖️ Dashboard Analítico - Processos Cíveis")
    
    with st.spinner("Carregando e extraindo dados base (Regex)..."):
        try:
            df = load_data()
        except FileNotFoundError:
            st.error("O arquivo 'dataset_clinica_20261.csv' não foi encontrado na pasta raiz.")
            st.stop()
    
    st.sidebar.header("🔍 Filtros de Análise")
    st.sidebar.markdown("Use os filtros abaixo para segmentar os casos e, em seguida, você pode analisá-los com IA.")
    
    # 1. Contato Prévio (DESTAQUE)
    contato_opts = ["Todos"] + sorted([str(x) for x in df["contato_previo_banco_regex"].dropna().unique()])
    contato_sel = st.sidebar.selectbox("📞 Houve Contato Prévio?", contato_opts, index=0)
    
    st.sidebar.divider()
    
    # 2. Outros filtros importantes
    tipo_opts = sorted(df["tipo_acao_regex"].dropna().unique().tolist())
    tipo_sel = st.sidebar.multiselect("Tipo de Ação", tipo_opts)
    
    resultado_opts = sorted(df["resultado_julgamento_regex"].dropna().unique().tolist())
    resultado_sel = st.sidebar.multiselect("Resultado do Julgamento", resultado_opts)
    
    rito_opts = sorted(df["rito_processual_regex"].dropna().unique().tolist())
    rito_sel = st.sidebar.multiselect("Rito Processual", rito_opts)
    
    jg_opts = sorted(df["justica_gratuita_regex"].dropna().unique().tolist())
    jg_sel = st.sidebar.multiselect("Justiça Gratuita", jg_opts)
    
    st.sidebar.divider()
    
    # 3. Metadados do Caso
    assuntos = sorted(df["assunto"].dropna().unique().tolist())
    assunto_sel = st.sidebar.multiselect("Assunto", assuntos)
    
    comarcas = sorted(df["comarca"].dropna().unique().tolist())
    comarca_sel = st.sidebar.multiselect("Comarca", comarcas)
    
    # Apply global filters based on dataset (using raw columns)
    df_filtered = df.copy()
    
    if contato_sel != "Todos":
        df_filtered = df_filtered[df_filtered["contato_previo_banco_regex"] == contato_sel]
    if tipo_sel:
        df_filtered = df_filtered[df_filtered["tipo_acao_regex"].isin(tipo_sel)]
    if resultado_sel:
        df_filtered = df_filtered[df_filtered["resultado_julgamento_regex"].isin(resultado_sel)]
    if rito_sel:
        df_filtered = df_filtered[df_filtered["rito_processual_regex"].isin(rito_sel)]
    if jg_sel:
        df_filtered = df_filtered[df_filtered["justica_gratuita_regex"].isin(jg_sel)]
    if assunto_sel:
        df_filtered = df_filtered[df_filtered["assunto"].isin(assunto_sel)]
    if comarca_sel:
        df_filtered = df_filtered[df_filtered["comarca"].isin(comarca_sel)]
        
    st.sidebar.success(f"**{len(df_filtered)} processos** correspondem aos filtros.")
    
    tab1, tab2 = st.tabs(["📊 Visão Geral (Regex)", "🤖 Análise com IA (OpenAI)"])
    
    geojson_data = load_geojson()
    comarca_map = build_comarca_to_geocodigo(geojson_data)
    
    with tab1:
        st.subheader("Análise baseada em Expressões Regulares")
        st.markdown("Esta aba exibe a Estatística Descritiva com base na extração via expressões regulares.")
        renderizar_eda_completa(df_filtered, suffix="regex", geojson_data=geojson_data, comarca_map=comarca_map)
        
    with tab2:
        st.subheader("Extração Avançada (OpenAI)")
        st.markdown("Nesta aba, você pode enviar os casos **atualmente filtrados na barra lateral** para a IA analisar de forma minuciosa. Os dados são armazenados em cache local, economizando custos.")
        
        if st.button(f"🚀 Analisar {len(df_filtered)} processos com IA", type="primary"):
            if len(df_filtered) == 0:
                st.warning("Nenhum processo selecionado para analisar.")
            else:
                with st.spinner("Extraindo dados com OpenAI... Isto pode levar um minuto."):
                    decisoes = df_filtered["decisao"].fillna("").tolist()
                    resultados_ia = run_ai_extraction(decisoes)
                    
                    if isinstance(resultados_ia, dict) and "error" in resultados_ia:
                        st.error(resultados_ia["error"])
                    else:
                        st.success("Análise concluída!")
                        df_ia = pd.DataFrame(resultados_ia)
                        df_ia = df_ia.rename(columns={c: f"{c}_ia" for c in _campos_padrao.keys()})
                        df_display = df_filtered.reset_index(drop=True).copy()
                        df_display = pd.concat([df_display, df_ia], axis=1)
                        st.session_state["df_ia_display"] = df_display
                        
        if "df_ia_display" in st.session_state:
            df_display = st.session_state["df_ia_display"]
            st.divider()
            renderizar_eda_completa(df_display, suffix="ia", geojson_data=geojson_data, comarca_map=comarca_map)
            
if __name__ == "__main__":
    main()
