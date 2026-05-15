# ──────────────────────────────────────────────────────────────────────────────
# SUBSTITUA A CÉLULA f7091b9b (Classificação por Regex) POR ESTE CÓDIGO
# ──────────────────────────────────────────────────────────────────────────────

from regex_otimizado import processar_batch
import time as time_module

print("Processando 14 funções regex em sequencial otimizado...")
t_inicio = time_module.time()

decisao_list = df_curto["decisao"].fillna("").tolist()

# Processa tudo em uma única passada (sem multiprocessing)
resultados_dict = processar_batch(decisao_list)

# Adiciona as colunas ao dataframe
for col_nome, valores in resultados_dict.items():
    df_curto[col_nome] = valores
    print(f"  ✓ {col_nome}")

tempo_total = time_module.time() - t_inicio
print(f"\n✓ Concluído em {tempo_total:.2f}s")
print(f"Colunas _regex adicionadas: {len([c for c in df_curto.columns if c.endswith('_regex')])}")
