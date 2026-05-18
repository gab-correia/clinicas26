Guia de Estrutura do Projeto

Entender o que cada parte do projeto faz

Estrutura Geral

Este projeto tem 3 objetivos principais:

1. Filtrar: Selecionar apenas os processos que fazem parte do escopo

2. Extrair: Coletar informações importantes de cada processo

3. Analisar: Entender padrões e relações nos dados

Como Funciona o Projeto

Passo 1: Entrada de Dados

Você fornece um arquivo CSV com:
- Metadados: informações básicas dos processos
- Sentença: texto completo da decisão judicial

Passo 2: Filtros (Seleção)

O programa passa o arquivo por 3 filtros:

Filtro 1: Menciona o Itaú?
O programa verifica se o texto da sentença menciona o Itaú. Se não menciona, remove.

Filtro 2: Dentro do Escopo? (3 Níveis)
O programa classifica em:
- Nível 1 "Dentro": Tem padrão claro de decisão (julgo procedente/improcedente)
- Nível 2 "Incerto": Sem padrão claro - vai para IA analisar
- Nível 3 "Fora": Tem padrão claro de não-decisão de mérito (acordo, desistência, etc)

Filtro 3: IA Decide
Para os casos "Incertos", a inteligência artificial decide se estão dentro ou fora.

Resultado: Apenas processos "Dentro" continuam para a próxima fase.

Passo 3: Extração (Coleta de Informações)

O programa coleta informações importantes de cada processo dentro do escopo.

Alguns campos são extraídos por padrão de texto (regex):
- Resultado do julgamento
- Menção a gratuidade
- Menção a boletim de ocorrência
- Menção a Reclame Aqui

Outros campos precisam de interpretação contextual (IA):
- Tipo de ação
- Contato prévio com o banco
- Canal de contato
- Culpa atribuída
- Valores de indenização

Passo 4: Análise (Entendimento dos Dados)

O programa cria:
- Gráficos mostrando distribuição de resultados
- Mapas geográficos (por comarca)
- Comparações (contato prévio vs vitória, etc)
- Modelos estatísticos para entender relações

Passo 5: Exportação (Saída de Dados)

O programa gera dois arquivos:

output.xlsx
- Lista de todos os processos dentro do escopo
- Inclui informações extraídas e enriquecidas
- Sempre tem coluna "id" com número do processo

relatorio.xlsx
- Estatísticas principais
- Percentual de processos com contato prévio
- Média de indenizações por danos materiais
- Média de indenizações por danos morais

Arquivos do Projeto

README.md
Instruções de como usar o programa. Comece por aqui.

TERMOS.md
Explicação de palavras técnicas usadas no projeto.

GUIA.md (este arquivo)
Estrutura e fluxo geral do projeto.

main.ipynb
Arquivo principal contendo todo o código. Cada seção explica o que está sendo feito.

requirements.txt
Lista de bibliotecas que o programa precisa para funcionar.

Como Ler o Notebook Principal

O arquivo main.ipynb está organizado em seções:

1. Configuração
Setup da inteligência artificial

2. Leitura e Filtros
Carregamento dos dados e aplicação dos 3 filtros

3. Extração
Coleta de informações (regex e IA)

4. Exportação
Criação dos arquivos de saída

5. Análise Exploratória
Gráficos e estatísticas da base de dados

6. Modelo Inferencial
Análises estatísticas avançadas

Cada seção tem células de código (código Python) e células de texto (markdown) explicando o que está acontecendo.

Célula: bloco de código ou texto no notebook que pode ser executado independentemente.

Conceitos Importantes

Escopo

Define quais processos o projeto está analisando. Neste projeto: processos judiciais que:
- Mencionam o Itaú
- Tiveram decisão de mérito (juiz analisou o caso)
- Foram julgados em primeira instância

Contato Prévio

Quando o cliente tentou resolver o problema com o banco antes de entrar com ação na justiça. Pode ser:
- SAC: Serviço de Atendimento ao Consumidor
- Procon: Órgão de proteção ao consumidor
- Reclame Aqui: Plataforma online de reclamações
- Outro tipo de contato

Indenização

Valor em dinheiro que o juiz determina que o banco deve pagar ao cliente para compensar os danos.

Pode ser:
- Por danos materiais: perdas financeiras reais
- Por danos morais: sofrimento, constrangimento, humilhação

Procedência

Resultado positivo para o cliente. O juiz determinou que o cliente estava certo e merecia ganhar a causa.

Taxa de Procedência

Percentual de processos em que o cliente ganhou (procedência) em relação ao total.

Perguntas que o Projeto Responde

1. Que percentual de processos tem registro de contato prévio com o banco?

2. Clientes que contataram o banco antes de processar têm taxa de vitória maior?

3. Os juízes determinam indenizações maiores quando há comprovação de contato prévio?

4. Qual canal de contato (SAC, Procon, Reclame Aqui) tem melhor resultado?

5. Quais características dos casos estão associadas ao contato prévio?

Próximos Passos

1. Leia o README.md para instruções de uso
2. Consulte TERMOS.md se encontrar palavras desconhecidas
3. Abra main.ipynb e execute célula por célula
4. Observe os gráficos e resultados gerados
