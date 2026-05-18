Clínica 2 — Análise de Processos do Itaú

O que o programa faz

Este programa analisa processos judiciais (ações) ajuizadas contra o Itaú. Ele lê um arquivo de dados (CSV) contendo informações sobre esses processos e as sentenças (decisões judiciais) e produz dois arquivos com os resultados.

CSV: arquivo de dados em formato de tabela, com linhas e colunas, que pode ser aberto em programas como Excel.

Entradas

O programa recebe um arquivo CSV com:
- Metadados dos processos (informações básicas como número do processo, data etc)
- Trechos das sentenças (partes do texto da decisão judicial)

Saídas

O programa gera dois arquivos:

1) output.xlsx — lista dos processos válidos
   Contém os processos em que o cliente contatou o banco antes de ajuizar a ação (fazer a reclamação antes de ir à justiça). Este arquivo inclui o número identificador único de cada processo (id) e outras informações que o grupo considerou importante analisar.

2) relatorio.xlsx — estatísticas dos resultados
   Apresenta três números principais:
   - A porcentagem de processos (em relação ao total) em que o cliente contactou o banco antes de ajuizar
   - O valor médio em reais das indenizações por danos materiais (danos físicos, perda de dinheiro, etc)
   - O valor médio em reais das indenizações por danos morais (humilhação, constrangimento, etc)

Como usar o programa

1) Instale as dependências
   Na linha de comando, execute: pip install -r requirements.txt
   
   Dependências: bibliotecas e ferramentas que o programa precisa para funcionar.

2) Abra o arquivo principal
   Abra o arquivo main.ipynb em um programa de notebook (como Jupyter Notebook ou JupyterLab).
   
   Notebook: um documento interativo que mistura código, explicações e resultados em um só lugar.

3) Localize a linha de importação
   Procure no código pelo comentário: LINHA DE SELEÇÃO DO INPUT
   Esta linha importa seu arquivo CSV. Você precisa apontar para o caminho correto do seu arquivo.

4) Execute o programa
   Execute todas as células do notebook (opção "Run All" ou "Executar Tudo").

Requisitos do arquivo CSV

O arquivo CSV de entrada deve ter o mesmo formato do dataset fornecido pelo Itaú no início da Clínica 2. Isso significa:
- Os nomes das colunas devem ser exatamente os mesmos
- Os tipos de dados (texto, números etc) devem ser iguais

