# Clínica 2 — Análise de Petições Iniciais (Itaú)

O programa recebe como entrada um arquivo CSV com metadados e trechos de petições iniciais ajuizadas contra o banco na justiça estadual e gera dois arquivos:

- `output.xlsx` — lista dos processos em que a parte autora fez contato com o banco antes do ajuizamento
- `relatorio.xlsx` — estatísticas descritivas com percentual de processos com contato prévio e médias dos pedidos de danos materiais e morais

## Como usar

1. Instale as dependências com `pip install -r requirements.txt`
2. Abra o notebook no Jupyter
3. Localize o comentário `LINHA DE SELEÇÃO DO INPUT` e aponte para o seu arquivo CSV
4. Execute todas as células

## Observação

O CSV de entrada deve ter o mesmo formato do dataset fornecido pelo Itaú, com os mesmos nomes de colunas.

