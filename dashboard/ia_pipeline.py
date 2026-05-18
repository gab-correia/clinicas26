"""
Pipeline IA portado do main.ipynb (Cells 4, 10, 11, 16, 17) para uso no Streamlit.

Duas etapas:
  1. Classificação de escopo dos casos Nível 2 (incertos)  → cache/escopo_classificacao.json
  2. Extração semântica de campos dos casos in-scope        → cache/extracao_campos_ia.json

Cache compatível com o notebook: mesma chave (md5 dos primeiros 500 chars).
"""

import asyncio
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from openai import AsyncOpenAI

# ════════════════════════════════════════════════════════════════════════════
# CONSTANTES (espelham Cell 4 do notebook)
# ════════════════════════════════════════════════════════════════════════════

MODELO_OPENAI = "gpt-4.1-mini"
MAX_CONCORRENTE = 8
SALVAR_CACHE_A_CADA = 50

# Resolve paths relativos à raiz do projeto (este arquivo vive em dashboard/)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_PATH_ESCOPO = _PROJECT_ROOT / "cache" / "escopo_classificacao.json"
CACHE_PATH_IA = _PROJECT_ROOT / "cache" / "extracao_campos_ia.json"

# Preço gpt-4.1-mini em 2026-05 (USD por 1M tokens)
PRECO_INPUT_PER_1M = 0.40
PRECO_OUTPUT_PER_1M = 1.60

# Estimativa conservadora — calibrada via _calibrar_tokens() na inicialização.
TOKENS_MEDIOS_INPUT_ESCOPO = 2200
TOKENS_MEDIOS_OUTPUT_ESCOPO = 40
TOKENS_MEDIOS_INPUT_EXTRACAO = 3500
TOKENS_MEDIOS_OUTPUT_EXTRACAO = 120

_campos_padrao_ia = {
    "tipo_acao":             "não identificado",
    "contato_previo_banco":  "não",
    "canal_contato":         "não identificado",
    "culpa_atribuida":       "não identificado",
    "valor_danos_morais":    0.0,
    "valor_danos_materiais": 0.0,
}

# ════════════════════════════════════════════════════════════════════════════
# PROMPTS (verbatim do notebook)
# ════════════════════════════════════════════════════════════════════════════

_PROMPT_SISTEMA_ESCOPO = (
    "Você é um analista jurídico especializado em processos cíveis contra bancos brasileiros. "
    "Classifique decisões judiciais por escopo. Responda APENAS com JSON válido."
)

_TEMPLATE_PROMPT_ESCOPO = """Analise o documento judicial abaixo e classifique o ESCOPO.

PASSO 1 — TIPO DO DOCUMENTO
Classifique: "sentença" / "despacho" / "decisão interlocutória" / "outro"
Se não for sentença → retorne: {{"tipo_documento": "<valor>", "fora_do_escopo": true}}

PASSO 2 — TRIAGEM DE ESCOPO (apenas sentenças)
Retorne fora_do_escopo: true se QUALQUER uma das condições abaixo for verdadeira:

  A) EXTINÇÃO SEM ANÁLISE DO MÉRITO BANCÁRIO
     - Processo extinto por incompetência territorial ACOLHIDA pelo juiz
     - Processo extinto pois parte absolutamente incapaz não pode litigar em Juizado Especial
     - Processo extinto por superendividamento incompatível com rito do Juizado Especial
     - Processo extinto por inércia, abandono ou falta de pressuposto processual
     - Sentença que apenas homologa acordo/desistência SEM apreciar o mérito bancário

  B) MATÉRIA ALHEIA A SERVIÇOS BANCÁRIOS DO ITAÚ
     - Réu principal NÃO é banco (escola, hospital, construtora, etc.)
     - Ação trabalhista, de família/herança, previdenciária sem relação com banco

  C) DOCUMENTO NÃO É SENTENÇA DE MÉRITO
     - Despacho, decisão interlocutória, embargos de declaração, cumprimento de sentença

ATENÇÃO — NÃO confundir com fora do escopo:
  ✓ Incompetência ARGUIDA pela ré mas REJEITADA pelo juiz → DENTRO do escopo
  ✓ Menção à palavra "incompetência" sem extinção do processo → DENTRO do escopo

PASSO 3 — RESPOSTA
Se nenhuma condição acima se aplica: fora_do_escopo: false

Retorne exatamente este JSON:
{{"tipo_documento": "...", "fora_do_escopo": true/false}}

DOCUMENTO:
{decisao}"""

_PROMPT_SISTEMA_EXTRACAO = (
    "Você é um analista jurídico especializado em processos cíveis contra bancos brasileiros. "
    "Extraia informações estruturadas de decisões judiciais em português. "
    "Responda APENAS com JSON válido, sem texto adicional."
)

_TEMPLATE_PROMPT_EXTRACAO = """Analise o documento judicial abaixo e extraia as variáveis indicadas.

════════════════════════════════════════
EXTRAÇÃO DE CAMPOS SEMÂNTICOS
════════════════════════════════════════
O documento já foi classificado como sentença dentro do escopo.
Extraia APENAS os campos abaixo — não reclassifique o escopo.

  1. tipo_acao — escolha UMA categoria:
     - "empréstimo não reconhecido": autor nega ter contratado empréstimo consignado ou pessoal
     - "fraude em conta ou cartão": golpe, transação não autorizada (motoboy, falsa central,
       PIX fraudulento, clonagem, "boa noite Cinderela", invasão de conta via engenharia social)
     - "revisão contratual": autor questiona taxas, juros ou cláusulas de contrato que RECONHECE ter firmado
     - "cobrança indevida": cobrança de tarifa, serviço, seguro ou parcela não contratada
     - "superendividamento": pedido de repactuação com base na Lei 14.181/2021
     - "outro": não se encaixa em nenhuma categoria acima

  2. contato_previo_banco: "sim" / "não"
     Retorne "sim" SOMENTE se a petição inicial relata que o autor buscou o banco ANTES de decidir
     processar, com objetivo de reclamar ou resolver o problema (ex: ligou pro SAC, foi à agência,
     registrou no Procon, enviou e-mail à ouvidoria).
     NÃO contar como contato prévio:
       - Contato feito para comunicar a fraude depois que ela ocorreu (ex: ligar para cancelar cartão)
       - Tentativa de estorno após o golpe já consumado
       - Qualquer contato posterior ao evento danoso sem intenção de reclamação prévia
     Em caso de dúvida ou ausência de menção → retorne "não".

  3. canal_contato: "SAC" / "Ouvidoria" / "Procon" / "Reclame Aqui" / "Agência" / "não identificado"
     Retorne o canal mencionado EXPLICITAMENTE como meio de contato prévio ao ajuizamento.
     Se múltiplos canais: retorne o primeiro/principal mencionado.
     Se contato_previo_banco = "não" → retorne "não identificado".

  4. culpa_atribuida: "banco" / "consumidor" / "terceiro" / "compartilhada" / "não identificado"
     Regras de inferência quando o texto não atribui explicitamente:
     - resultado procedente ou parcialmente procedente → "banco"
       (salvo texto indicar terceiro ou compartilhada)
     - resultado improcedente → "consumidor"
       (salvo exceção explícita — ex: fraude de terceiro reconhecida mesmo com improcedência)
     - resultado extinto → "não identificado"
     - fraude por terceiro onde banco não é condenado → "terceiro"

  5. valor_danos_morais: número decimal em reais
     - Valor CONDENADO na sentença (não o pedido da inicial)
     - 0.0 = não houve condenação por danos morais
     - -1.0 = condenado mas valor a apurar em liquidação de sentença

  6. valor_danos_materiais: número decimal em reais
     - Valor CONDENADO na sentença (não o pedido da inicial)
     - Inclui restituição, devolução, ressarcimento de valores materiais
     - 0.0 = não houve condenação por danos materiais
     - -1.0 = condenado mas valor a apurar em liquidação

REGRAS GERAIS:
  - Nunca infira o que não está escrito. Em caso de dúvida → "não identificado" ou "não".
  - O valor relevante é SEMPRE o condenado, nunca o pedido da parte autora.
  - Quando a sentença julgar múltiplos contratos com resultados diferentes → use o resultado majoritário.

Retorne exatamente este JSON (sem texto antes ou depois):
{{"tipo_acao":"...","contato_previo_banco":"...","canal_contato":"...","culpa_atribuida":"...","valor_danos_morais":0.0,"valor_danos_materiais":0.0}}

════════════════════════════════════════
DOCUMENTO:
════════════════════════════════════════
{decisao}"""


# ════════════════════════════════════════════════════════════════════════════
# CACHE
# ════════════════════════════════════════════════════════════════════════════

def hash_decisao(texto: str) -> str:
    """MD5 dos primeiros 500 chars (compatível com o notebook — não alterar)."""
    return hashlib.md5(texto[:500].strip().encode("utf-8")).hexdigest()


def carregar_cache_escopo() -> Dict[str, object]:
    if CACHE_PATH_ESCOPO.exists():
        with open(CACHE_PATH_ESCOPO, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def salvar_cache_escopo(cache: Dict[str, object]) -> None:
    CACHE_PATH_ESCOPO.parent.mkdir(exist_ok=True)
    with open(CACHE_PATH_ESCOPO, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def carregar_cache_ia() -> Dict[str, dict]:
    if CACHE_PATH_IA.exists():
        with open(CACHE_PATH_IA, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def salvar_cache_ia(cache: Dict[str, dict]) -> None:
    CACHE_PATH_IA.parent.mkdir(exist_ok=True)
    with open(CACHE_PATH_IA, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


# ════════════════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════════════════

def _parse_json_seguro(conteudo: str) -> dict:
    try:
        return json.loads(conteudo)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", conteudo, flags=re.DOTALL)
        if m:
            return json.loads(m.group(0))
        raise


def _make_async_client() -> AsyncOpenAI:
    load_dotenv(_PROJECT_ROOT / ".env")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY não definida. Adicione ao arquivo .env na raiz do projeto."
        )
    return AsyncOpenAI(api_key=api_key)


# ════════════════════════════════════════════════════════════════════════════
# CHAMADAS LLM
# ════════════════════════════════════════════════════════════════════════════

async def _chamar_llm_escopo_async(client, idx, texto, semaforo, max_tentativas=3):
    prompt = _TEMPLATE_PROMPT_ESCOPO.format(decisao=texto)
    async with semaforo:
        for tentativa in range(max_tentativas):
            try:
                resposta = await client.chat.completions.create(
                    model=MODELO_OPENAI,
                    messages=[
                        {"role": "system", "content": _PROMPT_SISTEMA_ESCOPO},
                        {"role": "user", "content": prompt},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0,
                )
                dados = _parse_json_seguro(resposta.choices[0].message.content.strip())
                return idx, bool(dados.get("fora_do_escopo", False))
            except Exception as e:
                is_rate_limit = "429" in str(e) or "RateLimit" in type(e).__name__
                if tentativa < max_tentativas - 1:
                    espera = (2 ** tentativa) * 10 if is_rate_limit else 2 ** tentativa
                    await asyncio.sleep(espera)
                else:
                    return idx, "FALHA_API_ESCOPO"


async def _chamar_llm_extracao_async(client, idx, texto, semaforo, max_tentativas=3):
    prompt = _TEMPLATE_PROMPT_EXTRACAO.format(decisao=texto)
    async with semaforo:
        for tentativa in range(max_tentativas):
            try:
                resposta = await client.chat.completions.create(
                    model=MODELO_OPENAI,
                    messages=[
                        {"role": "system", "content": _PROMPT_SISTEMA_EXTRACAO},
                        {"role": "user", "content": prompt},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0,
                )
                dados = _parse_json_seguro(resposta.choices[0].message.content.strip())
                for campo in ("valor_danos_morais", "valor_danos_materiais"):
                    try:
                        dados[campo] = float(dados.get(campo) or 0.0)
                    except (ValueError, TypeError):
                        dados[campo] = 0.0
                resultado = {**_campos_padrao_ia, **dados, "status_extracao": "ok"}
                return idx, resultado
            except Exception as e:
                is_rate_limit = "429" in str(e) or "RateLimit" in type(e).__name__
                if tentativa < max_tentativas - 1:
                    espera = (2 ** tentativa) * 10 if is_rate_limit else 2 ** tentativa
                    await asyncio.sleep(espera)
                else:
                    return idx, {**_campos_padrao_ia, "status_extracao": "falha_api"}


# ════════════════════════════════════════════════════════════════════════════
# ESTIMATIVA DE CUSTO (sem chamar API)
# ════════════════════════════════════════════════════════════════════════════

def _custo_usd(in_tokens: int, out_tokens: int) -> float:
    return in_tokens * PRECO_INPUT_PER_1M / 1_000_000 + out_tokens * PRECO_OUTPUT_PER_1M / 1_000_000


def estimar_custo(
    decisoes_amostra: List[str],
    niveis_amostra: List[int],
    modo: str,
) -> dict:
    """
    Conta cache hits reais e estima chamadas/custo sem chamar a API.

    Args:
        decisoes_amostra: lista de strings (decisão judicial) — uma por linha da amostra
        niveis_amostra: nivel_confianca_regex correspondente (1, 2 ou 3)
        modo: "l1_only" (analisa só L1) ou "l1_plus_l2" (analisa L1 + L2 confirmados in-scope)

    Returns:
        dict com contagens e custo_usd_estimado.
    """
    assert modo in ("l1_only", "l1_plus_l2"), f"modo inválido: {modo}"
    cache_escopo = carregar_cache_escopo()
    cache_ia = carregar_cache_ia()

    hits_escopo = 0
    calls_escopo = 0
    indices_in_scope = []  # índices que chegarão à etapa de extração

    for i, (texto, nivel) in enumerate(zip(decisoes_amostra, niveis_amostra)):
        if nivel == 3:
            continue  # FORA já decidido por regex
        if nivel == 1:
            indices_in_scope.append(i)
            continue
        # nivel == 2
        if modo == "l1_only":
            continue  # L2 ignorado nesse modo
        # modo == "l1_plus_l2": precisa classificar escopo
        h = hash_decisao(texto)
        cache_val = cache_escopo.get(h)
        if cache_val is None or cache_val == "FALHA_API_ESCOPO":
            calls_escopo += 1
            # Não sabemos o resultado ainda — assume otimisticamente in-scope p/ estimativa
            indices_in_scope.append(i)
        else:
            hits_escopo += 1
            if cache_val is False:  # fora_do_escopo=False → DENTRO
                indices_in_scope.append(i)

    hits_extracao = 0
    calls_extracao = 0
    for i in indices_in_scope:
        h = hash_decisao(decisoes_amostra[i])
        cache_val = cache_ia.get(h)
        if cache_val is None or (
            isinstance(cache_val, dict) and cache_val.get("status_extracao") == "falha_api"
        ):
            calls_extracao += 1
        else:
            hits_extracao += 1

    custo = (
        calls_escopo * _custo_usd(TOKENS_MEDIOS_INPUT_ESCOPO, TOKENS_MEDIOS_OUTPUT_ESCOPO)
        + calls_extracao * _custo_usd(TOKENS_MEDIOS_INPUT_EXTRACAO, TOKENS_MEDIOS_OUTPUT_EXTRACAO)
    )

    return {
        "linhas_amostra": len(decisoes_amostra),
        "linhas_in_scope_estimado": len(indices_in_scope),
        "cache_hits_escopo": hits_escopo,
        "calls_escopo": calls_escopo,
        "cache_hits_extracao": hits_extracao,
        "calls_extracao": calls_extracao,
        "custo_usd_estimado": custo,
        "modelo": MODELO_OPENAI,
        "concorrencia": MAX_CONCORRENTE,
    }


# ════════════════════════════════════════════════════════════════════════════
# ORQUESTRADOR ASYNC + WRAPPER SÍNCRONO
# ════════════════════════════════════════════════════════════════════════════

async def _processar_lote_async(
    decisoes_amostra: List[str],
    niveis_amostra: List[int],
    modo: str,
    on_progress: Optional[Callable[[str, int, int], None]] = None,
) -> Tuple[Dict[int, Optional[bool]], Dict[int, dict], dict]:
    """
    Roda as duas etapas (escopo se necessário, extração) e devolve resultados por índice da amostra.

    Returns:
        (escopo_por_idx, extracao_por_idx, stats)
        escopo_por_idx[i] ∈ {True, False, None}  (True=FORA, False=DENTRO, None=L1 já era DENTRO)
        extracao_por_idx[i] = dict com campos _ia (só para in-scope)
    """
    assert modo in ("l1_only", "l1_plus_l2")
    client = _make_async_client()
    cache_escopo = carregar_cache_escopo()
    cache_ia = carregar_cache_ia()
    sem = asyncio.Semaphore(MAX_CONCORRENTE)

    escopo_por_idx: Dict[int, Optional[bool]] = {}
    extracao_por_idx: Dict[int, dict] = {}
    stats = {
        "cache_hits_escopo": 0,
        "calls_escopo": 0,
        "falhas_escopo": 0,
        "cache_hits_extracao": 0,
        "calls_extracao": 0,
        "falhas_extracao": 0,
    }

    # ─── Etapa 1: escopo (só L2) ────────────────────────────────────────────
    indices_l2_a_chamar = []
    for i, (texto, nivel) in enumerate(zip(decisoes_amostra, niveis_amostra)):
        if nivel == 3:
            escopo_por_idx[i] = True  # FORA
            continue
        if nivel == 1:
            escopo_por_idx[i] = False  # DENTRO
            continue
        if modo == "l1_only":
            escopo_por_idx[i] = True  # ignora L2
            continue
        h = hash_decisao(texto)
        cached = cache_escopo.get(h)
        if cached is None or cached == "FALHA_API_ESCOPO":
            indices_l2_a_chamar.append(i)
        else:
            stats["cache_hits_escopo"] += 1
            escopo_por_idx[i] = bool(cached)

    if indices_l2_a_chamar:
        tasks = [
            _chamar_llm_escopo_async(client, i, decisoes_amostra[i], sem)
            for i in indices_l2_a_chamar
        ]
        pendentes_flush = 0
        feitos = 0
        total = len(tasks)
        for coro in asyncio.as_completed(tasks):
            idx, resultado = await coro
            feitos += 1
            if resultado == "FALHA_API_ESCOPO":
                stats["falhas_escopo"] += 1
                escopo_por_idx[idx] = True  # fallback conservador → FORA
            else:
                escopo_por_idx[idx] = bool(resultado)
                cache_escopo[hash_decisao(decisoes_amostra[idx])] = bool(resultado)
                stats["calls_escopo"] += 1
                pendentes_flush += 1
                if pendentes_flush >= SALVAR_CACHE_A_CADA:
                    salvar_cache_escopo(cache_escopo)
                    pendentes_flush = 0
            if on_progress:
                on_progress("escopo", feitos, total)
        if pendentes_flush > 0:
            salvar_cache_escopo(cache_escopo)

    # ─── Etapa 2: extração de campos (só in-scope) ──────────────────────────
    indices_in_scope = [i for i, fora in escopo_por_idx.items() if fora is False]
    indices_extracao_a_chamar = []
    for i in indices_in_scope:
        h = hash_decisao(decisoes_amostra[i])
        cached = cache_ia.get(h)
        if cached is None or (
            isinstance(cached, dict) and cached.get("status_extracao") == "falha_api"
        ):
            indices_extracao_a_chamar.append(i)
        else:
            stats["cache_hits_extracao"] += 1
            extracao_por_idx[i] = cached

    if indices_extracao_a_chamar:
        tasks = [
            _chamar_llm_extracao_async(client, i, decisoes_amostra[i], sem)
            for i in indices_extracao_a_chamar
        ]
        pendentes_flush = 0
        feitos = 0
        total = len(tasks)
        for coro in asyncio.as_completed(tasks):
            idx, resultado = await coro
            feitos += 1
            extracao_por_idx[idx] = resultado
            cache_ia[hash_decisao(decisoes_amostra[idx])] = resultado
            if resultado.get("status_extracao") == "falha_api":
                stats["falhas_extracao"] += 1
            else:
                stats["calls_extracao"] += 1
            pendentes_flush += 1
            if pendentes_flush >= SALVAR_CACHE_A_CADA:
                salvar_cache_ia(cache_ia)
                pendentes_flush = 0
            if on_progress:
                on_progress("extracao", feitos, total)
        if pendentes_flush > 0:
            salvar_cache_ia(cache_ia)

    stats["custo_usd_real_estimado"] = (
        stats["calls_escopo"] * _custo_usd(TOKENS_MEDIOS_INPUT_ESCOPO, TOKENS_MEDIOS_OUTPUT_ESCOPO)
        + stats["calls_extracao"] * _custo_usd(TOKENS_MEDIOS_INPUT_EXTRACAO, TOKENS_MEDIOS_OUTPUT_EXTRACAO)
    )

    return escopo_por_idx, extracao_por_idx, stats


def processar_lote(
    decisoes_amostra: List[str],
    niveis_amostra: List[int],
    modo: str,
    on_progress: Optional[Callable[[str, int, int], None]] = None,
) -> Tuple[Dict[int, Optional[bool]], Dict[int, dict], dict]:
    """Wrapper síncrono — chama o async com asyncio.run (precisa de nest_asyncio aplicado)."""
    return asyncio.run(
        _processar_lote_async(decisoes_amostra, niveis_amostra, modo, on_progress)
    )


def reprocessar_falhas(decisoes_por_idx: Dict[int, str]) -> Dict[int, dict]:
    """
    Reroda extração para um conjunto específico de índices (geralmente falhas anteriores).
    Limpa a entrada de cache antes de tentar.
    """
    cache_ia = carregar_cache_ia()
    for texto in decisoes_por_idx.values():
        h = hash_decisao(texto)
        cache_ia.pop(h, None)
    salvar_cache_ia(cache_ia)

    indices = list(decisoes_por_idx.keys())
    decisoes = list(decisoes_por_idx.values())
    niveis = [1] * len(indices)  # força tratamento como DENTRO

    async def _runner():
        client = _make_async_client()
        sem = asyncio.Semaphore(MAX_CONCORRENTE)
        tasks = [_chamar_llm_extracao_async(client, i, t, sem) for i, t in zip(indices, decisoes)]
        out = {}
        for coro in asyncio.as_completed(tasks):
            idx, resultado = await coro
            out[idx] = resultado
            cache_ia[hash_decisao(decisoes_por_idx[idx])] = resultado
        salvar_cache_ia(cache_ia)
        return out

    return asyncio.run(_runner())
