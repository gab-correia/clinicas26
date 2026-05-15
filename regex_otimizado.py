"""
Versão otimizada em Python puro com regex pré-compiladas.

Performance esperada:
  - ~8-10x mais rápido que Python normal (por pré-compilação de regex)
  - Sem dependência de compilador C++
  - 22k documentos × 14 funções em ~2-3 minutos

Uso:
  from regex_otimizado import processar_batch
  resultados = processar_batch(lista_de_decisoes)
"""

import re
from typing import List, Dict, Any


# ════════════════════════════════════════════════════════════════════════════
# PADRÕES PRÉ-COMPILADOS (compilados uma única vez na inicialização)
# ════════════════════════════════════════════════════════════════════════════

_PATTERNS = {
    "tipo_documento": re.compile(r"SENTENÇA|DESPACHO|DECISÃO", re.IGNORECASE),
    "fora_do_escopo": re.compile(r"família|divórcio|herança|sucessão", re.IGNORECASE),
    "justica_gratuita": re.compile(r"justiça\s+gratuita|assistência\s+judiciária|pobre", re.IGNORECASE),
    "rito_juizado": re.compile(r"Juizado\s+Especial|Juizado", re.IGNORECASE),
    "rito_comum": re.compile(r"Procedimento\s+Comum|Comum", re.IGNORECASE),
    "tipo_acao": re.compile(r"fraude|golpe|cobrança\s+indevida|empréstimo\s+não\s+reconhecido|revisão\s+contratual", re.IGNORECASE),
    "contato_previo": re.compile(r"SAC|Serviço\s+de\s+Atendimento|contato\s+prévio|antes\s+de\s+ajuizar", re.IGNORECASE),
    "canal_sac": re.compile(r"SAC\s+da\s+instituição|SAC|Serviço\s+de\s+Atendimento", re.IGNORECASE),
    "canal_procon": re.compile(r"PROCON|Procon", re.IGNORECASE),
    "canal_ouvidoria": re.compile(r"Ouvidoria", re.IGNORECASE),
    "canal_reclame": re.compile(r"Reclame\s+Aqui|ReclameAqui", re.IGNORECASE),
    "canal_agencia": re.compile(r"agência|banco", re.IGNORECASE),
    "boletim": re.compile(r"boletim\s+de\s+ocorrência|b\.?\s*o\.?|boletim", re.IGNORECASE),
    "resultado_parcial": re.compile(r"procedente\s+em\s+parte|parcialmente\s+procedente", re.IGNORECASE),
    "resultado_procedente": re.compile(r"procedente", re.IGNORECASE),
    "resultado_improcedente": re.compile(r"improcedente", re.IGNORECASE),
    "resultado_extinto": re.compile(r"extinto", re.IGNORECASE),
    "culpa_banco": re.compile(r"banco|instituição financeira|responsabilidade\s+da\s+ré", re.IGNORECASE),
    "culpa_consumidor": re.compile(r"consumidor|cliente|responsabilidade\s+da\s+parte\s+autora", re.IGNORECASE),
    "repeticao": re.compile(r"repetição\s+indébito|indébito|valor\s+indevido", re.IGNORECASE),
}


# ════════════════════════════════════════════════════════════════════════════
# FUNÇÕES OTIMIZADAS
# ════════════════════════════════════════════════════════════════════════════

def tipo_documento(texto: str) -> str:
    """Classifica tipo de documento."""
    if not texto:
        return "não identificado"

    match = _PATTERNS["tipo_documento"].search(texto)
    if match:
        tipo = match.group().upper()
        if "SENTENÇA" in tipo:
            return "Sentença"
        elif "DESPACHO" in tipo:
            return "Despacho"
        elif "DECISÃO" in tipo:
            return "Decisão"
    return "não identificado"


def fora_do_escopo(texto: str) -> bool:
    """Verifica se está fora do escopo."""
    return bool(_PATTERNS["fora_do_escopo"].search(texto)) if texto else False


def justica_gratuita(texto: str) -> str:
    """Identifica justiça gratuita."""
    return "sim" if (texto and _PATTERNS["justica_gratuita"].search(texto)) else "não"


def rito_processual(texto: str) -> str:
    """Identifica rito processual."""
    if not texto:
        return "não identificado"

    if _PATTERNS["rito_juizado"].search(texto):
        return "Juizado Especial"
    elif _PATTERNS["rito_comum"].search(texto):
        return "Procedimento Comum"
    return "não identificado"


def tipo_acao(texto: str) -> str:
    """Classifica tipo de ação."""
    if not texto:
        return "não identificado"

    texto_lower = texto.lower()
    # Prioridade: mais específico primeiro
    if ("fraude" in texto_lower or "golpe" in texto_lower):
        return "fraude/golpe"
    elif ("cobrança" in texto_lower and "indevida" in texto_lower):
        return "cobrança indevida"
    elif ("empréstimo" in texto_lower and "não reconhecido" in texto_lower):
        return "empréstimo não reconhecido"
    elif ("revisão" in texto_lower and "contratual" in texto_lower):
        return "revisão contratual"

    return "não identificado"


def contato_previo_banco(texto: str) -> str:
    """Verifica contato prévio com banco."""
    return "sim" if (texto and _PATTERNS["contato_previo"].search(texto)) else "não"


def canal_contato(texto: str) -> str:
    """Identifica canal de contato."""
    if not texto:
        return "não identificado"

    if _PATTERNS["canal_sac"].search(texto):
        return "SAC"
    elif _PATTERNS["canal_procon"].search(texto):
        return "Procon"
    elif _PATTERNS["canal_ouvidoria"].search(texto):
        return "Ouvidoria"
    elif _PATTERNS["canal_reclame"].search(texto):
        return "Reclame Aqui"
    elif _PATTERNS["canal_agencia"].search(texto):
        return "Agência"

    return "não identificado"


def mencao_reclame_aqui(texto: str) -> str:
    """Verifica menção a Reclame Aqui."""
    return "sim" if (texto and _PATTERNS["canal_reclame"].search(texto)) else "não"


def boletim_de_ocorrencia(texto: str) -> str:
    """Verifica menção a boletim de ocorrência."""
    return "sim" if (texto and _PATTERNS["boletim"].search(texto)) else "não"


def resultado_julgamento(texto: str) -> str:
    """Identifica resultado do julgamento."""
    if not texto:
        return "não identificado"

    if _PATTERNS["resultado_parcial"].search(texto):
        return "parcialmente procedente"
    elif _PATTERNS["resultado_procedente"].search(texto):
        return "procedente"
    elif _PATTERNS["resultado_improcedente"].search(texto):
        return "improcedente"
    elif _PATTERNS["resultado_extinto"].search(texto):
        return "extinto"

    return "não identificado"


def culpa_atribuida(texto: str) -> str:
    """Identifica a quem é atribuída a culpa."""
    if not texto:
        return "não identificado"

    if _PATTERNS["culpa_banco"].search(texto):
        return "banco"
    elif _PATTERNS["culpa_consumidor"].search(texto):
        return "consumidor"

    return "não identificado"


def repeticao_indebito(texto: str) -> str:
    """Verifica se há repetição de indébito."""
    return "sim" if (texto and _PATTERNS["repeticao"].search(texto)) else "não"


def _extrair_valor(texto: str, tipo: str) -> float:
    """Extrai valores monetários (danos morais ou materiais)."""
    if not texto:
        return 0.0

    # Define padrão baseado no tipo
    if tipo == "morais":
        pattern = r"danos\s+morais[^R]*?R\$\s*[\s\.,0-9]+"
    else:  # materiais
        pattern = r"danos\s+materiais[^R]*?R\$\s*[\s\.,0-9]+"

    match = re.search(pattern, texto, re.IGNORECASE)
    if not match:
        return 0.0

    # Extrai todos os números do match
    valores = re.findall(r"\d+[.,]?\d*", match.group())
    if not valores:
        return 0.0

    try:
        # Pega o último número encontrado (geralmente é o valor)
        valor_str = valores[-1]
        # Normaliza separadores: "." vira vazio, "," vira "."
        valor_str = valor_str.replace(".", "").replace(",", ".")
        return float(valor_str)
    except (ValueError, IndexError):
        return 0.0


def valor_danos_morais(texto: str) -> float:
    """Extrai valor de danos morais."""
    return _extrair_valor(texto, "morais")


def valor_danos_materiais(texto: str) -> float:
    """Extrai valor de danos materiais."""
    return _extrair_valor(texto, "materiais")


# ════════════════════════════════════════════════════════════════════════════
# FUNÇÃO DE BATCH (processa lista inteira de forma eficiente)
# ════════════════════════════════════════════════════════════════════════════

def processar_batch(textos: List[str]) -> Dict[str, List[Any]]:
    """
    Processa um lote de textos e retorna dicionário com todos os resultados.

    Args:
        textos: Lista de strings (decisões)

    Returns:
        dict com listas de resultados para cada campo
    """
    resultados = {
        "tipo_documento_regex": [],
        "fora_do_escopo_regex": [],
        "justica_gratuita_regex": [],
        "rito_processual_regex": [],
        "tipo_acao_regex": [],
        "contato_previo_banco_regex": [],
        "canal_contato_regex": [],
        "mencao_reclame_aqui_regex": [],
        "boletim_de_ocorrencia_regex": [],
        "resultado_julgamento_regex": [],
        "culpa_atribuida_regex": [],
        "repeticao_indebito_regex": [],
        "valor_danos_morais_regex": [],
        "valor_danos_materiais_regex": [],
    }

    for texto in textos:
        resultados["tipo_documento_regex"].append(tipo_documento(texto))
        resultados["fora_do_escopo_regex"].append(fora_do_escopo(texto))
        resultados["justica_gratuita_regex"].append(justica_gratuita(texto))
        resultados["rito_processual_regex"].append(rito_processual(texto))
        resultados["tipo_acao_regex"].append(tipo_acao(texto))
        resultados["contato_previo_banco_regex"].append(contato_previo_banco(texto))
        resultados["canal_contato_regex"].append(canal_contato(texto))
        resultados["mencao_reclame_aqui_regex"].append(mencao_reclame_aqui(texto))
        resultados["boletim_de_ocorrencia_regex"].append(boletim_de_ocorrencia(texto))
        resultados["resultado_julgamento_regex"].append(resultado_julgamento(texto))
        resultados["culpa_atribuida_regex"].append(culpa_atribuida(texto))
        resultados["repeticao_indebito_regex"].append(repeticao_indebito(texto))
        resultados["valor_danos_morais_regex"].append(valor_danos_morais(texto))
        resultados["valor_danos_materiais_regex"].append(valor_danos_materiais(texto))

    return resultados
