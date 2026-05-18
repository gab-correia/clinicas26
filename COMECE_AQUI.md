Comece Aqui

Instruções rápidas para começar

1. Prepare o Ambiente

Abra o terminal ou prompt de comando e execute:

```
pip install -r requirements.txt
```

2. Abra o Notebook

Abra o arquivo main.ipynb com Jupyter Notebook ou JupyterLab.

Jupyter: programa que permite editar e executar código Python em documentos interativos.

3. Configure Seu Arquivo de Entrada

Localize no notebook o comentário:

LINHA DE SELEÇÃO DO INPUT

Nessa linha, aponte para seu arquivo CSV. Exemplo:

```
df = pd.read_csv('caminho/do/seu/arquivo.csv')
```

4. Execute o Programa

Clique em "Run All" ou "Executar Tudo" para rodar o programa inteiro.

5. Veja os Resultados

O programa cria automaticamente:
- output.xlsx (lista de processos com contato prévio)
- relatorio.xlsx (estatísticas principais)
- Gráficos e tabelas no notebook

Se Precisar de Ajuda

Consultando Documentação

- README.md — Como usar o programa
- GUIA.md — Estrutura e fluxo do projeto
- TERMOS.md — Explicação de palavras técnicas

Entendendo o Código

O notebook está dividido em seções claras:

1. Configuração (setup da IA)
2. Leitura e Filtros (dados de entrada)
3. Extração (coleta de informações)
4. Análise (gráficos e estatísticas)
5. Modelos (análises avançadas)

Cada seção tem texto explicativo acima do código.

Requisitos do Arquivo CSV

Seu arquivo CSV deve ter:
- As mesmas colunas do dataset fornecido
- Os mesmos tipos de dados
- Informações sobre processos judiciais contra o Itaú

Solução de Problemas

Erro ao Instalar Dependências

Se o pip install falhar, tente:

```
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Ou instale manualmente:

```
pip install pandas openpyxl requests anthropic folium plotly
```

Erro ao Executar o Notebook

Verifique:
1. O caminho do seu arquivo CSV está correto?
2. O arquivo CSV tem o formato esperado?
3. Todas as dependências foram instaladas?

Se ainda tiver problemas, veja a seção de documentação acima.

Próximos Passos

Após rodar o programa:

1. Abra os arquivos output.xlsx e relatorio.xlsx em Excel
2. Leia os resultados e gráficos gerados
3. Consulte GUIA.md para entender o fluxo completo
4. Explore o código modificando variáveis e testando
