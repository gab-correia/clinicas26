# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""
Versão Cython otimizada de todas as funções regex do projeto.
Compile com: python setup.py build_ext --inplace

Performance esperada:
  - ~10-15x mais rápido que Python puro
  - ~3-4x mais rápido que multiprocessing
  - 22k documentos × 14 funções em ~1-2 minutos
"""

import re
from typing import List


# ════════════════════════════════════════════════════════════════════════════
# PADRÕES COMPILADOS (compilados uma única vez na inicialização)
# ════════════════════════════════════════════════════════════════════════════

cdef dict _patterns = {
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
    "valor_pattern": re.compile(r"\d+[.,]\d{2,3}"),
}


# ════════════════════════════════════════════════════════════════════════════
# FUNÇÕES CYTHON OTIMIZADAS
# ════════════════════════════════════════════════════════════════════════════

cdef inline str _tipo_documento(str texto):
    """Classifica tipo de documento."""
    if not texto:
        return "não identificado"

    if _patterns["tipo_documento"].search(texto):
        match = _patterns["tipo_documento"].search(texto)
        tipo = match.group().upper()
        if "SENTENÇA" in tipo:
            return "Sentença"
        elif "DESPACHO" in tipo:
            return "Despacho"
        elif "DECISÃO" in tipo:
            return "Decisão"
    return "não identificado"


cdef inline bint _fora_do_escopo(str texto):
    """Verifica se está fora do escopo (processos de família, herança, etc)."""
    return bool(_patterns["fora_do_escopo"].search(texto)) if texto else False


cdef inline str _justica_gratuita(str texto):
    """Identifica justiça gratuita."""
    return "sim" if (texto and _patterns["justica_gratuita"].search(texto)) else "não"


cdef inline str _rito_processual(str texto):
    """Identifica rito processual."""
    if not texto:
        return "não identificado"

    if _patterns["rito_juizado"].search(texto):
        return "Juizado Especial"
    elif _patterns["rito_comum"].search(texto):
        return "Procedimento Comum"
    return "não identificado"


cdef inline str _tipo_acao(str texto):
    """Classifica tipo de ação."""
    if not texto:
        return "não identificado"

    # Prioridade: mais específico primeiro
    if "fraude" in texto.lower() or "golpe" in texto.lower():
        return "fraude/golpe"
    elif "cobrança" in texto.lower() and "indevida" in texto.lower():
        return "cobrança indevida"
    elif "empréstimo" in texto.lower() and "não reconhecido" in texto.lower():
        return "empréstimo não reconhecido"
    elif "revisão" in texto.lower() and "contratual" in texto.lower():
        return "revisão contratual"

    return "não identificado"


cdef inline str _contato_previo(str texto):
    """Verifica contato prévio com banco."""
    return "sim" if (texto and _patterns["contato_previo"].search(texto)) else "não"


cdef inline str _canal_contato(str texto):
    """Identifica canal de contato."""
    if not texto:
        return "não identificado"

    if _patterns["canal_sac"].search(texto):
        return "SAC"
    elif _patterns["canal_procon"].search(texto):
        return "Procon"
    elif _patterns["canal_ouvidoria"].search(texto):
        return "Ouvidoria"
    elif _patterns["canal_reclame"].search(texto):
        return "Reclame Aqui"
    elif _patterns["canal_agencia"].search(texto):
        return "Agência"

    return "não identificado"


cdef inline str _mencao_reclame(str texto):
    """Verifica menção a Reclame Aqui."""
    return "sim" if (texto and _patterns["canal_reclame"].search(texto)) else "não"


cdef inline str _boletim_ocorrencia(str texto):
    """Verifica menção a boletim de ocorrência."""
    return "sim" if (texto and _patterns["boletim"].search(texto)) else "não"


cdef inline str _resultado_julgamento(str texto):
    """Identifica resultado do julgamento."""
    if not texto:
        return "não identificado"

    if _patterns["resultado_parcial"].search(texto):
        return "parcialmente procedente"
    elif _patterns["resultado_procedente"].search(texto):
        return "procedente"
    elif _patterns["resultado_improcedente"].search(texto):
        return "improcedente"
    elif _patterns["resultado_extinto"].search(texto):
        return "extinto"

    return "não identificado"


cdef inline str _culpa_atribuida(str texto):
    """Identifica a quem é atribuída a culpa."""
    if not texto:
        return "não identificado"

    if _patterns["culpa_banco"].search(texto):
        return "banco"
    elif _patterns["culpa_consumidor"].search(texto):
        return "consumidor"

    return "não identificado"


cdef inline str _repeticao_indebito(str texto):
    """Verifica se há repetição de indébito."""
    return "sim" if (texto and _patterns["repeticao"].search(texto)) else "não"


cdef inline float _extrair_valor(str texto, str tipo):
    """Extrai valores monetários genéricos (danos morais ou materiais)."""
    cdef list valores

    if not texto:
        return 0.0

    # Busca o padrão "danos morais" ou "danos materiais" seguido de R$
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


cdef inline float _valor_morais(str texto):
    """Extrai valor de danos morais."""
    return _extrair_valor(texto, "morais")


cdef inline float _valor_materiais(str texto):
    """Extrai valor de danos materiais."""
    return _extrair_valor(texto, "materiais")


# ════════════════════════════════════════════════════════════════════════════
# WRAPPERS PÚBLICOS (interface Python)
# ════════════════════════════════════════════════════════════════════════════

def tipo_documento(str texto):
    """Classifica tipo de documento."""
    return _tipo_documento(texto)

def fora_do_escopo(str texto):
    """Verifica se está fora do escopo."""
    return _fora_do_escopo(texto)

def justica_gratuita(str texto):
    """Identifica justiça gratuita."""
    return _justica_gratuita(texto)

def rito_processual(str texto):
    """Identifica rito processual."""
    return _rito_processual(texto)

def tipo_acao(str texto):
    """Classifica tipo de ação."""
    return _tipo_acao(texto)

def contato_previo_banco(str texto):
    """Verifica contato prévio com banco."""
    return _contato_previo(texto)

def canal_contato(str texto):
    """Identifica canal de contato."""
    return _canal_contato(texto)

def mencao_reclame_aqui(str texto):
    """Verifica menção a Reclame Aqui."""
    return _mencao_reclame(texto)

def boletim_de_ocorrencia(str texto):
    """Verifica menção a boletim de ocorrência."""
    return _boletim_ocorrencia(texto)

def resultado_julgamento(str texto):
    """Identifica resultado do julgamento."""
    return _resultado_julgamento(texto)

def culpa_atribuida(str texto):
    """Identifica a quem é atribuída a culpa."""
    return _culpa_atribuida(texto)

def repeticao_indebito(str texto):
    """Verifica se há repetição de indébito."""
    return _repeticao_indebito(texto)

def valor_danos_morais(str texto):
    """Extrai valor de danos morais."""
    return _valor_morais(texto)

def valor_danos_materiais(str texto):
    """Extrai valor de danos materiais."""
    return _valor_materiais(texto)


# ════════════════════════════════════════════════════════════════════════════
# FUNÇÃO DE BATCH (processa lista inteira)
# ════════════════════════════════════════════════════════════════════════════

def processar_batch(list textos):
    """
    Processa um lote de textos e retorna dicionário com todos os resultados.

    Args:
        textos: Lista de strings (decisões)

    Returns:
        dict com listas de resultados para cada campo
    """
    cdef dict resultados = {
        "tipo_documento": [],
        "fora_do_escopo": [],
        "justica_gratuita": [],
        "rito_processual": [],
        "tipo_acao": [],
        "contato_previo_banco": [],
        "canal_contato": [],
        "mencao_reclame_aqui": [],
        "boletim_de_ocorrencia": [],
        "resultado_julgamento": [],
        "culpa_atribuida": [],
        "repeticao_indebito": [],
        "valor_danos_morais": [],
        "valor_danos_materiais": [],
    }

    cdef str texto
    for texto in textos:
        resultados["tipo_documento"].append(_tipo_documento(texto))
        resultados["fora_do_escopo"].append(_fora_do_escopo(texto))
        resultados["justica_gratuita"].append(_justica_gratuita(texto))
        resultados["rito_processual"].append(_rito_processual(texto))
        resultados["tipo_acao"].append(_tipo_acao(texto))
        resultados["contato_previo_banco"].append(_contato_previo(texto))
        resultados["canal_contato"].append(_canal_contato(texto))
        resultados["mencao_reclame_aqui"].append(_mencao_reclame(texto))
        resultados["boletim_de_ocorrencia"].append(_boletim_ocorrencia(texto))
        resultados["resultado_julgamento"].append(_resultado_julgamento(texto))
        resultados["culpa_atribuida"].append(_culpa_atribuida(texto))
        resultados["repeticao_indebito"].append(_repeticao_indebito(texto))
        resultados["valor_danos_morais"].append(_valor_morais(texto))
        resultados["valor_danos_materiais"].append(_valor_materiais(texto))

    return resultados
