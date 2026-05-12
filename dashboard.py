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
    
    # Apply regex
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
    """Load the São Paulo municipalities GeoJSON."""
    with open(GEOJSON_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def _normalizar_nome(nome):
    """Normalize a name for comparison: remove accents, lowercase, strip."""
    if pd.isna(nome):
        return ""
    t = unicodedata.normalize("NFKD", str(nome)).encode("ascii", "ignore").decode("ascii")
    return t.strip().lower()

@st.cache_data
def build_comarca_to_geocodigo(geojson_data):
    """Build a mapping from normalized municipality name to GEOCODIGO."""
    mapping = {}
    for feature in geojson_data["features"]:
        nome = feature["properties"]["NOME"]
        geocodigo = feature["properties"]["GEOCODIGO"]
        mapping[_normalizar_nome(nome)] = geocodigo
    return mapping

def agregar_mapa_data(df, col_resultado, col_morais, comarca_map):
    """Aggregate data by comarca and match to GeoJSON geocodigos.
    
    Returns a DataFrame with columns:
      - comarca, geocodigo, total_processos, valor_medio_morais, taxa_procedencia
    """
    # Group by comarca
    agg = df.groupby("comarca").agg(
        total_processos=("id_processo", "count"),
        valor_medio_morais=(col_morais, "mean"),
    ).reset_index()
    
    # Taxa de procedência: % of processes that are "procedente" or "parcialmente procedente"
    def _calc_taxa(group):
        total = len(group)
        if total == 0:
            return 0.0
        favoraveis = group[col_resultado].isin(["procedente", "parcialmente procedente"]).sum()
        return (favoraveis / total) * 100
    
    taxa = df.groupby("comarca").apply(_calc_taxa, include_groups=False).reset_index()
    taxa.columns = ["comarca", "taxa_procedencia"]
    
    agg = agg.merge(taxa, on="comarca", how="left")
    
    # Match comarca to geocodigo
    agg["comarca_norm"] = agg["comarca"].apply(_normalizar_nome)
    agg["geocodigo"] = agg["comarca_norm"].map(comarca_map)
    
    # Round values
    agg["valor_medio_morais"] = agg["valor_medio_morais"].round(2)
    agg["taxa_procedencia"] = agg["taxa_procedencia"].round(1)
    
    return agg

def render_mapa_sp(df_mapa, geojson_data, metrica, key_suffix=""):
    """Render a choropleth map of São Paulo state.
    
    Args:
        df_mapa: DataFrame with geocodigo and metric columns
        geojson_data: GeoJSON dict
        metrica: One of 'volume', 'valor_morais', 'taxa_procedencia'
        key_suffix: Unique key suffix for Streamlit widgets
    """
    # Filter rows that have a valid geocodigo match
    df_plot = df_mapa[df_mapa["geocodigo"].notna()].copy()
    
    if df_plot.empty:
        st.warning("Nenhuma comarca pôde ser mapeada para os municípios de SP.")
        return
    
    # Metric config
    metric_config = {
        "volume": {
            "col": "total_processos",
            "title": "Volume de Processos por Comarca",
            "label": "Nº Processos",
            "scale": "YlOrRd",
            "fmt": ",.0f",
        },
        "valor_morais": {
            "col": "valor_medio_morais",
            "title": "Valor Médio de Danos Morais por Comarca",
            "label": "Valor Médio (R$)",
            "scale": "Purples",
            "fmt": ",.2f",
        },
        "taxa_procedencia": {
            "col": "taxa_procedencia",
            "title": "Taxa de Procedência por Comarca",
            "label": "Procedência (%)",
            "scale": "Greens",
            "fmt": ".1f",
        },
    }
    
    cfg = metric_config[metrica]
    
    fig = px.choropleth(
        df_plot,
        geojson=geojson_data,
        locations="geocodigo",
        featureidkey="properties.GEOCODIGO",
        color=cfg["col"],
        color_continuous_scale=cfg["scale"],
        hover_name="comarca",
        hover_data={
            "geocodigo": False,
            "total_processos": ":,",
            "valor_medio_morais": ":,.2f",
            "taxa_procedencia": ":.1f",
        },
        labels={
            cfg["col"]: cfg["label"],
            "total_processos": "Processos",
            "valor_medio_morais": "Danos Morais Médio (R$)",
            "taxa_procedencia": "Procedência (%)",
        },
        title=cfg["title"],
    )
    
    fig.update_geos(
        fitbounds="locations",
        visible=False,
        bgcolor="rgba(0,0,0,0)",
    )
    
    fig.update_layout(
        margin={"r": 0, "t": 40, "l": 0, "b": 0},
        height=600,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        coloraxis_colorbar=dict(
            title=cfg["label"],
            thickness=15,
            len=0.7,
        ),
    )
    
    st.plotly_chart(fig, use_container_width=True, key=f"mapa_{metrica}_{key_suffix}")
    
    # Stats cards
    matched = len(df_plot)
    total_comarcas = len(df_mapa)
    unmatched = total_comarcas - matched
    
    if unmatched > 0:
        st.caption(f"📍 {matched} comarcas mapeadas de {total_comarcas} ({unmatched} sem correspondência no mapa)")
    
    # Top 10 ranking table
    with st.expander("📊 Ranking — Top 20 Comarcas", expanded=False):
        top = df_mapa.nlargest(20, cfg["col"])[["comarca", "total_processos", "valor_medio_morais", "taxa_procedencia"]]
        top.columns = ["Comarca", "Processos", "Danos Morais Médio (R$)", "Procedência (%)"]
        st.dataframe(top.reset_index(drop=True), use_container_width=True)

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

def _hash_decisao(texto):
    return hashlib.md5(texto.encode("utf-8")).hexdigest()

_campos_padrao = {
    "justica_gratuita": "não",
    "rito_processual": "não identificado",
    "tipo_acao": "não identificado",
    "contato_previo_banco": "não",
    "canal_contato": "não identificado",
    "mencao_reclame_aqui": "não",
    "boletim_de_ocorrencia": "não",
    "resultado_julgamento": "não identificado",
    "culpa_atribuida": "não identificado",
    "valor_danos_morais": 0.0,
    "valor_danos_materiais": 0.0,
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
- canal_contato: retorne o canal apenas se o texto mencionar explicitamente SAC, Ouvidoria, Procon,
  Reclame Aqui ou visita à agência antes do ajuizamento. Caso contrário, retorne "não identificado".
- culpa_atribuida:
    • resultado=improcedente → "consumidor" (salvo se o texto indicar outro responsável)
    • resultado=procedente ou parcialmente procedente → "banco" (salvo exceção explícita no texto)
    • resultado=extinto → "não identificado"
    • Use "não identificado" apenas se genuinamente ambíguo mesmo conhecendo o resultado.

Retorne exatamente este JSON:
{{"justica_gratuita":"...","rito_processual":"...","tipo_acao":"...","contato_previo_banco":"...","canal_contato":"...","mencao_reclame_aqui":"...","boletim_de_ocorrencia":"...","resultado_julgamento":"...","culpa_atribuida":"...","valor_danos_morais":0.0,"valor_danos_materiais":0.0}}

Decisão:
{decisao}"""

prompt_sistema = (
    "Você é um analista jurídico especializado em processos cíveis contra bancos. "
    "Extraia informações estruturadas de decisões judiciais em português. "
    "Responda APENAS com JSON válido, sem texto adicional."
)

async def _chamar_decisao_async(idx, texto, semaforo, async_client, max_tentativas=4):
    prompt = template_prompt.format(decisao=texto)
    async with semaforo:
        for tentativa in range(max_tentativas):
            try:
                resposta = await async_client.chat.completions.create(
                    model="gpt-4.1",
                    messages=[
                        {"role": "system", "content": prompt_sistema},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0,
                    response_format={"type": "json_object"},
                )
                conteudo = resposta.choices[0].message.content.strip()
                try:
                    dados = json.loads(conteudo)
                except:
                    m = re.search(r"\{.*\}", conteudo, flags=re.DOTALL)
                    if m:
                        dados = json.loads(m.group(0))
                    else:
                        raise ValueError("No JSON found")
                
                for campo in ("valor_danos_morais", "valor_danos_materiais"):
                    try:
                        dados[campo] = float(dados.get(campo) or 0.0)
                    except:
                        dados[campo] = 0.0
                return idx, {**_campos_padrao, **dados}

            except Exception as e:
                is_429 = "429" in str(e)
                is_last = tentativa == max_tentativas - 1
                if is_429 and not is_last:
                    await asyncio.sleep(2 ** tentativa)
                    continue
                return idx, {**_campos_padrao, "resultado_julgamento": f"ERRO: {str(e)}"}

async def processar_async_batch(decisoes, progress_bar):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return {"error": "OPENAI_API_KEY não definida no .env"}
        
    async_client = AsyncOpenAI(api_key=api_key)
    cache_local = _carregar_cache()
    
    resultados_final = [None] * len(decisoes)
    indices_api = []
    
    for i, texto in enumerate(decisoes):
        h = _hash_decisao(texto)
        if h in cache_local:
            resultados_final[i] = cache_local[h]
        else:
            indices_api.append(i)
            
    semaforo = asyncio.Semaphore(5)
    tasks = [_chamar_decisao_async(i, decisoes[i], semaforo, async_client) for i in indices_api]
    
    pendentes_flush = 0
    concluidas = 0
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
            
    if pendentes_flush > 0:
        _salvar_cache(cache_local)
        
    return resultados_final

def run_ai_extraction(decisoes):
    progress_bar = st.progress(0)
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(processar_async_batch(decisoes, progress_bar))

# --- UI APP ---
def main():
    st.title("⚖️ Dashboard Analítico - Processos Cíveis")
    
    with st.spinner("Carregando e extraindo dados base (Regex)..."):
        try:
            df = load_data()
        except FileNotFoundError:
            st.error("O arquivo 'dataset_clinica_20261.csv' não foi encontrado na pasta raiz.")
            st.stop()
    
    st.sidebar.header("Filtros")
    assuntos = sorted(df["assunto"].dropna().unique().tolist())
    assunto_sel = st.sidebar.multiselect("Assunto", assuntos)
    
    comarcas = sorted(df["comarca"].dropna().unique().tolist())
    comarca_sel = st.sidebar.multiselect("Comarca", comarcas)
    
    resultados_regex = sorted(df["resultado_julgamento_regex"].dropna().unique().tolist())
    resultado_sel = st.sidebar.multiselect("Resultado (Regex)", resultados_regex)
    
    df_filtered = df.copy()
    if assunto_sel:
        df_filtered = df_filtered[df_filtered["assunto"].isin(assunto_sel)]
    if comarca_sel:
        df_filtered = df_filtered[df_filtered["comarca"].isin(comarca_sel)]
    if resultado_sel:
        df_filtered = df_filtered[df_filtered["resultado_julgamento_regex"].isin(resultado_sel)]
        
    st.sidebar.markdown(f"**Total de processos filtrados: {len(df_filtered)}**")
    
    tab1, tab2 = st.tabs(["📊 Visão Geral (Regex)", "🤖 Análise com IA (OpenAI)"])
    
    # Load GeoJSON for maps
    geojson_data = load_geojson()
    comarca_map = build_comarca_to_geocodigo(geojson_data)
    
    with tab1:
        st.subheader("Análise baseada em Expressões Regulares")
        
        col1, col2, col3 = st.columns(3)
        total_morais = df_filtered["valor_danos_morais_regex"].sum()
        total_materiais = df_filtered["valor_danos_materiais_regex"].sum()
        
        col1.metric("Total de Processos", len(df_filtered))
        col2.metric("Danos Morais (Total)", f"R$ {total_morais:,.2f}")
        col3.metric("Danos Materiais (Total)", f"R$ {total_materiais:,.2f}")
        
        col_fig1, col_fig2 = st.columns(2)
        
        tipo_acao_counts = df_filtered["tipo_acao_regex"].value_counts().reset_index()
        tipo_acao_counts.columns = ["Tipo de Ação", "Quantidade"]
        fig1 = px.bar(tipo_acao_counts, x="Tipo de Ação", y="Quantidade", title="Volume por Tipo de Ação")
        col_fig1.plotly_chart(fig1, use_container_width=True)
        
        resultado_counts = df_filtered["resultado_julgamento_regex"].value_counts().reset_index()
        resultado_counts.columns = ["Resultado", "Quantidade"]
        fig2 = px.pie(resultado_counts, names="Resultado", values="Quantidade", title="Resultados dos Julgamentos")
        col_fig2.plotly_chart(fig2, use_container_width=True)
        
        # --- Mapa Coroplético (Regex) ---
        st.divider()
        st.markdown("### 🗺️ Mapa de Processos — Estado de São Paulo")
        
        metrica_regex = st.selectbox(
            "Selecione a métrica para o mapa:",
            ["volume", "valor_morais", "taxa_procedencia"],
            format_func=lambda x: {
                "volume": "📊 Volume de Processos",
                "valor_morais": "💰 Valor Médio de Danos Morais",
                "taxa_procedencia": "⚖️ Taxa de Procedência (%)",
            }[x],
            key="metrica_mapa_regex",
        )
        
        df_mapa_regex = agregar_mapa_data(
            df_filtered,
            col_resultado="resultado_julgamento_regex",
            col_morais="valor_danos_morais_regex",
            comarca_map=comarca_map,
        )
        render_mapa_sp(df_mapa_regex, geojson_data, metrica_regex, key_suffix="regex")
        
        st.divider()
        st.markdown("### Processos Encontrados")
        cols_to_show = ["id_processo", "assunto", "tipo_acao_regex", "resultado_julgamento_regex", "valor_danos_morais_regex", "culpa_atribuida_regex"]
        st.dataframe(df_filtered[cols_to_show], use_container_width=True)
        
        # Expander para ver as decisões
        if not df_filtered.empty:
            processo_sel = st.selectbox("Selecione um processo para ler a decisão:", df_filtered["id_processo"].tolist())
            if processo_sel:
                texto_decisao = df_filtered[df_filtered["id_processo"] == processo_sel]["decisao"].values[0]
                with st.expander("Ler Decisão Original"):
                    st.text(texto_decisao)
        
    with tab2:
        st.subheader("Extração Avançada (OpenAI)")
        st.markdown("Nesta aba, você pode enviar os casos **atualmente filtrados** para a IA analisar de forma minuciosa. Os dados são armazenados em cache local, economizando custos.")
        
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
                        
                        # Store in session state corresponding to the filtered index
                        df_display = df_filtered.reset_index(drop=True).copy()
                        df_display = pd.concat([df_display, df_ia], axis=1)
                        
                        # Store exactly this snapshot in session_state
                        st.session_state["df_ia_display"] = df_display
                        
        if "df_ia_display" in st.session_state:
            df_display = st.session_state["df_ia_display"]
            
            st.divider()
            st.markdown("### Resultados da IA")
            
            col1, col2, col3 = st.columns(3)
            total_morais_ia = df_display["valor_danos_morais_ia"].sum()
            total_materiais_ia = df_display["valor_danos_materiais_ia"].sum()
            
            col1.metric("Total de Processos (IA)", len(df_display))
            col2.metric("Danos Morais (IA)", f"R$ {total_morais_ia:,.2f}")
            col3.metric("Danos Materiais (IA)", f"R$ {total_materiais_ia:,.2f}")
            
            col_fig1, col_fig2 = st.columns(2)
            
            tipo_acao_counts = df_display["tipo_acao_ia"].value_counts().reset_index()
            tipo_acao_counts.columns = ["Tipo de Ação", "Quantidade"]
            fig1 = px.bar(tipo_acao_counts, x="Tipo de Ação", y="Quantidade", title="Volume por Tipo de Ação (IA)", color_discrete_sequence=["#FF7F0E"])
            col_fig1.plotly_chart(fig1, use_container_width=True)
            
            resultado_counts = df_display["resultado_julgamento_ia"].value_counts().reset_index()
            resultado_counts.columns = ["Resultado", "Quantidade"]
            fig2 = px.pie(resultado_counts, names="Resultado", values="Quantidade", title="Resultados dos Julgamentos (IA)")
            col_fig2.plotly_chart(fig2, use_container_width=True)
            
            # --- Mapa Coroplético (IA) ---
            st.divider()
            st.markdown("### 🗺️ Mapa de Processos — Análise IA")
            
            metrica_ia = st.selectbox(
                "Selecione a métrica para o mapa:",
                ["volume", "valor_morais", "taxa_procedencia"],
                format_func=lambda x: {
                    "volume": "📊 Volume de Processos",
                    "valor_morais": "💰 Valor Médio de Danos Morais",
                    "taxa_procedencia": "⚖️ Taxa de Procedência (%)",
                }[x],
                key="metrica_mapa_ia",
            )
            
            df_mapa_ia = agregar_mapa_data(
                df_display,
                col_resultado="resultado_julgamento_ia",
                col_morais="valor_danos_morais_ia",
                comarca_map=comarca_map,
            )
            render_mapa_sp(df_mapa_ia, geojson_data, metrica_ia, key_suffix="ia")
            
            st.divider()
            cols_to_show_ia = ["id_processo", "assunto", "tipo_acao_ia", "resultado_julgamento_ia", "culpa_atribuida_ia", "valor_danos_morais_ia"]
            st.dataframe(df_display[cols_to_show_ia], use_container_width=True)
            
if __name__ == "__main__":
    main()
