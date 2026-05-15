"""Teste de performance: Python puro vs Regex otimizado"""
import time
from regex_otimizado import processar_batch

# Gera dados de teste realistas
templates = [
    """SENTENÇA

    Processo Digital nº: {id}
    Parte autora: {autor}
    Parte ré: {reu}

    RELATÓRIO
    Trata-se de ação cível ordinária com benefício da justiça gratuita.
    O autor alega ter feito contato prévio com o SAC do banco, conforme documentação.

    VOTO
    Verifica-se que a ação é procedente em parte.
    Condena-se a ré ao pagamento de danos morais no valor de R$ 5.000,00
    e danos materiais no valor de R$ 2.500,00.
    """,

    """SENTENÇA

    Processo Digital nº: {id}
    Procedimento Comum Cível

    O demandante ajuizou ação contra o banco sem contato prévio.
    A sentença é improcedente.
    """,

    """DESPACHO

    Processo nº: {id}
    Juizado Especial Cível

    Ouve-se que o consumidor contatou a Ouvidoria do banco antes de ajuizar.
    Verifica-se condenação por danos materiais: R$ 1.500,00
    """,
]

# Testa com diferentes tamanhos
for n in [100, 500, 1000, 5000]:
    textos = []
    for i in range(n):
        template = templates[i % len(templates)]
        texto = template.format(id=f"{i:08d}", autor=f"Autor {i}", reu="Banco XYZ")
        textos.append(texto)

    print(f"\n{'─' * 60}")
    print(f"Processando {n} documentos × 14 funções regex...")
    print(f"{'─' * 60}")

    t0 = time.time()
    resultados = processar_batch(textos)
    tempo = time.time() - t0

    total_ops = n * 14
    ops_por_segundo = total_ops / tempo

    print(f"✓ Tempo total: {tempo:.2f}s")
    print(f"✓ Operações: {total_ops:,} ({ops_por_segundo:,.0f} ops/s)")
    print(f"✓ Resultado: {len(resultados)} colunas × {len(resultados['tipo_documento_regex'])} linhas")

print(f"\n✅ Teste concluído!")
