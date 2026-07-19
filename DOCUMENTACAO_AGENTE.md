# Como o agente GAIA funciona

## Visão geral

Este projeto é um agente criado para responder às questões da avaliação GAIA,
permitir que as respostas sejam revisadas e somente depois enviá-las ao sistema
oficial de pontuação.

O agente não é apenas uma caixa de conversa que envia todas as perguntas para
um modelo de inteligência artificial. Ele combina o Gemini com ferramentas
especializadas. Cada ferramenta foi criada para lidar com um tipo de informação,
como páginas da internet, documentos, áudios, imagens, planilhas, tabelas,
estatísticas esportivas ou versões antigas da Wikipédia.

O Gemini atua como o coordenador da pesquisa. Ele interpreta a pergunta,
observa as informações encontradas pelas ferramentas, decide se ainda precisa
pesquisar e produz a resposta final. Algumas situações previsíveis são
identificadas pelo próprio programa antes da primeira chamada ao Gemini. Nesses
casos, a ferramenta mais apropriada é executada automaticamente e sua evidência
é entregue ao modelo já no início.

O objetivo desse desenho é obter respostas mais confiáveis, reduzir chamadas
desnecessárias e impedir que textos explicativos sejam enviados como resposta
em uma avaliação que utiliza correspondência exata.

## O caminho percorrido por uma pergunta

Quando uma pergunta é executada, o sistema realiza o seguinte ciclo:

1. A pergunta e seu identificador são recebidos.
2. O sistema verifica se existe um anexo oficial associado à questão.
3. O tipo do anexo é identificado pelo nome e pela extensão do arquivo.
4. O controlador verifica se a pergunta corresponde a uma rota especializada,
   como áudio, planilha, estatística de beisebol ou promoção de artigo destacado
   da Wikipédia.
5. Se houver uma rota especializada, a ferramenta correspondente é executada
   imediatamente.
6. A pergunta, as informações sobre o anexo e as evidências já coletadas são
   entregues ao Gemini.
7. O Gemini pode responder diretamente ou utilizar outras ferramentas para
   completar a pesquisa.
8. Toda informação devolvida pelas ferramentas é registrada pelo controlador.
9. Quando há evidência suficiente, o Gemini entrega uma resposta candidata.
10. O programa remove explicações, marcações e outros elementos que poderiam
    prejudicar a comparação exata.
11. Uma revisão final do Gemini verifica se a resposta está direta e no formato
    solicitado.
12. A resposta é salva para revisão humana. Ela não é enviada automaticamente.

Assim, uma pergunta que pede uma quantidade deve terminar apenas com o número.
Uma pergunta que pede um código deve terminar apenas com o código. Uma pergunta
que pede uma lista separada por vírgulas deve manter somente os itens, na ordem
e com o separador solicitados.

## Papel do Gemini

O modelo principal configurado atualmente é o Gemini. Ele é acessado por meio do
LiteLLM e controlado pelo `smolagents`, que fornece ao modelo uma lista fechada
de ferramentas disponíveis.

O Gemini não pode inventar novas ferramentas nem executar comandos no servidor.
Ele escolhe apenas entre as funções que o projeto disponibiliza. As chamadas de
ferramenta são processadas de forma sequencial, o que reduz duplicações e deixa
o histórico da pesquisa mais previsível.

O modelo recebe instruções para:

- dar preferência a fontes oficiais ou primárias;
- não tratar um trecho de resultado de busca como prova suficiente quando a
  página original pode ser consultada;
- não repetir pesquisas quase idênticas;
- encerrar a pesquisa assim que o valor solicitado estiver comprovado;
- retornar somente o valor final, sem raciocínio, explicação ou citação.

O número de passos está configurado como praticamente ilimitado. Isso não
significa que o agente possa pesquisar indefinidamente. Um detector separado
observa o comportamento das ferramentas e interrompe ciclos sem progresso.

## Rotas automáticas

Antes de permitir que o Gemini escolha uma ferramenta, o programa procura
sinais claros na pergunta e no anexo. Isso evita gastar uma chamada do modelo
apenas para tomar uma decisão que o sistema já consegue tomar com segurança.

As principais rotas automáticas são:

- Arquivo de áudio: o sistema chama a transcrição de áudio.
- Imagem: o sistema chama a análise visual.
- Planilha: o sistema descreve primeiro as abas, colunas e primeiras linhas.
- PDF, DOCX, texto, código ou ZIP: o sistema extrai o conteúdo relevante.
- Pergunta sobre fala em um vídeo do YouTube: o sistema busca a transcrição.
- Pergunta histórica de MLB com time, ano e estatística: o sistema consulta a
  API oficial da MLB.
- Pergunta sobre um artigo destacado da Wikipédia promovido em determinado mês
  e ano: o sistema lê diretamente a tabela oficial de promoções da Wikipédia.

Depois dessa coleta inicial, o Gemini recebe a orientação de não repetir a mesma
chamada. Se a evidência já contiver a resposta, ele deve finalizar imediatamente.

## Ferramentas disponíveis

### Busca na internet

A ferramenta de busca utiliza o DDGS e não exige uma chave de API. Ela devolve
somente alguns resultados compactos, com título, endereço e um pequeno trecho.
Buscas idênticas são mantidas em cache durante a execução para evitar chamadas
duplicadas.

A busca serve para descobrir fontes. Ela não deve ser usada como única prova
quando existe uma página que pode ser aberta.

### Leitura de páginas

A ferramenta de páginas abre um endereço exato, extrai o conteúdo principal e
seleciona os trechos mais relacionados à pergunta. Partes de navegação,
publicidade e conteúdo repetitivo são reduzidas antes de o texto chegar ao
Gemini. Quando uma página bloqueia a leitura comum, o sistema pode tentar uma
representação textual alternativa.

### Leitura de documentos publicados

Quando a fonte é um PDF, DOCX, CSV ou arquivo de texto disponível por endereço
direto, a ferramenta baixa o documento, extrai seu conteúdo e conserva somente
as passagens relacionadas aos termos procurados. Há um limite de tamanho para
evitar downloads excessivos.

### Consulta de tabelas da internet

Para perguntas de maior, menor, ranking, total ou empate, o sistema pode ler uma
tabela HTML e fazer a seleção localmente. O Gemini recebe apenas o resultado e
uma pequena amostra, em vez de receber a tabela inteira. Em empates, os nomes
são ordenados alfabeticamente.

### Estatísticas da MLB

A ferramenta de beisebol consulta a API oficial da MLB, identifica o time e a
temporada, ordena os jogadores pela estatística solicitada e devolve também
campos relacionados, como aparições no bastão e caminhadas. Ela não depende de
busca genérica nem de páginas que possam mudar de formato.

### Wikipédia atual

A busca comum da Wikipédia é usada para informações enciclopédicas atuais,
quando a pergunta não exige uma versão histórica específica.

### Versões históricas da Wikipédia

Quando a pergunta menciona uma versão, revisão ou ano da Wikipédia, a ferramenta
histórica recupera a última revisão existente até 31 de dezembro daquele ano.
Ela trabalha com o conteúdo original da revisão e preserva tabelas que poderiam
desaparecer em um resumo comum.

### Artigos destacados da Wikipédia

Perguntas sobre artigos destacados promovidos em um mês e ano seguem uma rota
própria. A ferramenta lê a tabela oficial daquele período, extrai artigo e
nomeador e consulta o resumo e as categorias de cada artigo para identificar o
tema solicitado.

Por exemplo, para a pergunta sobre o único artigo de dinossauro promovido em
novembro de 2016, a ferramenta encontra `Giganotosaurus` e identifica
`FunkMonk` como nomeador. Isso substitui uma sequência longa de buscas por uma
consulta direta e verificável.

### Anexos de texto e documentos

O sistema tenta obter o anexo primeiro pela API do exercício. Se esse endereço
estiver indisponível, utiliza o arquivo correspondente no dataset oficial GAIA.

PDFs, documentos do Word, arquivos de texto, páginas HTML, código e ZIP podem
ser inspecionados. Para arquivos longos, somente os trechos mais relevantes são
mantidos.

### Planilhas

Planilhas XLSX, XLSM, CSV e TSV são processadas localmente. O fluxo recomendado
é primeiro descrever o arquivo e depois executar uma operação específica, como
contagem, soma, média, mínimo, máximo, valores únicos ou seleção de linhas.

Filtros podem ser aplicados antes do cálculo. Dessa forma, a planilha completa
não é enviada ao Gemini.

### Áudio

Áudios são baixados e enviados ao serviço de reconhecimento de fala da Hugging
Face. O modelo padrão é o Whisper Large V3. O texto transcrito é então entregue
ao Gemini para que ele extraia somente a informação solicitada.

Essa etapa depende do `HF_TOKEN` e da disponibilidade do serviço de inferência
da Hugging Face.

### Vídeos do YouTube

Para perguntas sobre o que foi dito em um vídeo, o sistema tenta obter as
legendas ou a transcrição disponível no YouTube. Ele não interpreta eventos
puramente visuais do vídeo.

### Imagens

Imagens oficiais do GAIA são enviadas a um modelo visual disponível pelo
roteador da Hugging Face. Essa ferramenta é apropriada para diagramas, detalhes
visuais, posições de xadrez e outras perguntas que dependam dos pixels.

Essa etapa também depende do `HF_TOKEN` e da disponibilidade do modelo visual.

### Calculadora

Operações matemáticas são executadas localmente. A calculadora aceita
aritmética, potências, parênteses e algumas funções matemáticas. O cálculo não
consome uma chamada adicional de modelo.

## Controle de repetição e consumo de tokens

Todas as ferramentas são observadas por um registrador de evidências. Ele conta
quantas chamadas foram feitas, identifica resultados repetidos e percebe
sequências de respostas sem informação útil.

Por padrão, a pesquisa é interrompida quando ocorre uma das seguintes situações:

- duas observações repetidas;
- três resultados consecutivos sem evidência útil;
- oito buscas de descoberta sem resposta final;
- quatorze chamadas de ferramenta sem resposta final.

Esses limites podem ser ajustados por variáveis de ambiente.

Quando o detector interrompe o agente, as evidências úteis já coletadas não são
descartadas. O Gemini recebe uma chamada de consolidação sem acesso a novas
ferramentas. Ele deve responder apenas se o material existente for suficiente.
Caso contrário, a questão é marcada como falha e nenhuma resposta vazia ou
inacabada é salva.

## Tratamento da resposta final

O GAIA usa correspondência exata. Uma resposta verdadeira, mas acompanhada de
explicações, pode ser considerada errada. Por isso, o projeto aplica duas
camadas de tratamento.

A primeira camada é determinística. Ela remove blocos de código, Markdown,
rótulos como “Final answer”, frases introdutórias e chamadas de ferramenta que
tenham sido devolvidas por engano. Também reconhece alguns formatos importantes,
como números de páginas, quantidades, valores em dólares, números de prêmio,
nomes e listas.

A segunda camada é uma revisão curta feita pelo Gemini. O revisor recebe a
pergunta, a resposta candidata e as evidências coletadas. Ele devolve um objeto
estruturado contendo apenas a resposta final e uma nota interna opcional.

Se essa revisão falhar por cota, resposta inválida ou indisponibilidade, a
resposta primária não é perdida. O sistema preserva a versão já limpa pela
camada determinística.

## Obtenção das 20 questões

O sistema tenta carregar as questões pelo endpoint oficial do exercício. Quando
esse serviço retorna erro, o programa acessa diretamente o dataset GAIA na
Hugging Face e reproduz o filtro usado pela avaliação.

As questões carregadas ficam em cache durante a sessão. Os anexos baixados
também são mantidos em memória para evitar novos downloads da mesma tarefa.

O fallback exige que o usuário tenha aceitado as condições do dataset GAIA e
que o `HF_TOKEN` possua permissão de leitura.

## Interface web

A interface possui três abas.

### Testar agente

Essa aba carrega uma única questão oficial aleatória. O usuário pode executá-la
sem preencher uma pergunta manualmente e sem enviar nada ao sistema de
pontuação.

Depois da execução, a aba mostra a resposta e um índice de prontidão. Esse
índice verifica apenas se existe erro e se o formato parece adequado. Ele não
tem acesso ao gabarito e não representa a pontuação oficial.

### Executar e revisar

Essa aba carrega as 20 questões. É possível executar todas de uma vez ou
selecionar uma questão específica.

Cada resposta fica associada ao respectivo identificador. O usuário pode:

- responder novamente uma questão;
- substituir uma resposta anterior;
- digitar uma correção manual;
- salvar a resposta revisada;
- acompanhar o número de questões respondidas;
- revisar a tabela completa antes do envio.

Executar uma questão novamente substitui somente a resposta daquela questão.
As demais respostas permanecem salvas na sessão.

### Enviar resultado

Essa aba envia os valores que estão na tabela de revisão. Antes do envio, o
sistema bloqueia respostas vazias e linhas que ainda contenham erros.

O payload enviado contém o usuário, a URL pública do código do agente e a lista
de identificadores com suas respostas. A pontuação oficial só é conhecida
depois que o endpoint de submissão aceita esse payload.

Se a API estiver indisponível, as respostas continuam na tabela da sessão para
que o envio seja tentado novamente mais tarde.

## Telegram

Quando `TELEGRAM_BOT_TOKEN` está configurado, o bot é iniciado junto com a
interface web. Ele chama as mesmas funções de carregamento, execução, revisão e
envio utilizadas pela interface.

O bot não controla o navegador e não compartilha esta conversa do Codex. Ele é
uma segunda interface para o mesmo programa que está rodando no Codespaces.

Somente o chat configurado em `TELEGRAM_ALLOWED_CHAT_ID` pode executar comandos.
Enquanto esse identificador não estiver definido, apenas o comando `/id`
funciona.

Pelo Telegram é possível carregar as 20 questões, listar identificadores,
sortear ou selecionar uma questão, executar uma questão, executar o lote,
interromper o lote, consultar progresso, editar respostas e confirmar o envio.

A execução das 20 questões ocorre em segundo plano para que o bot continue
respondendo a comandos de status. O comando de parada solicita a interrupção e
o lote termina de forma controlada.

O bot também verifica periodicamente a disponibilidade do serviço oficial e
avisa quando ele volta a funcionar. Esse monitor depende de o Codespaces
continuar ativo.

## Índice de prontidão e pontuação oficial

O índice de prontidão não mede acurácia. Ele considera uma resposta pronta
quando ela:

- não está vazia;
- não começa com uma mensagem de erro;
- não é uma resposta inconclusiva;
- não contém o rótulo proibido “final answer”;
- não parece excessivamente longa para correspondência exata.

Uma resposta pode receber 100% de prontidão e ainda estar factualmente errada.
Da mesma forma, esse índice não revela antecipadamente qual seria a pontuação
GAIA. Somente a submissão oficial compara as respostas com o gabarito.

## Configuração e serviços externos

O funcionamento completo depende principalmente destas configurações:

- `GEMINI_API_KEY`: autentica o modelo principal e o revisor.
- `HF_TOKEN`: permite acessar o dataset, anexos, transcrição e análise visual.
- `GAIA_GEMINI_MODEL`: seleciona o modelo Gemini principal.
- `GAIA_GEMINI_REVIEW_MODEL`: permite escolher um modelo Gemini diferente para
  a revisão final.
- `GAIA_MAX_STEPS`: define um limite numérico ou mantém o modo praticamente
  ilimitado.
- `GAIA_STAGNATION_TOOL_CALLS`: limite de chamadas de ferramenta.
- `GAIA_STAGNATION_SEARCH_CALLS`: limite de buscas de descoberta.
- `GAIA_STAGNATION_DUPLICATES`: tolerância a observações repetidas.
- `GAIA_ASR_MODEL`: permite trocar o modelo usado na transcrição de áudio.
- `GAIA_VISION_MODEL`: permite trocar o modelo usado na análise de imagens.
- `HF_USERNAME`: identifica o usuário quando não há login do Hugging Face.
- `AGENT_CODE_URL`: informa a URL pública do código quando ela não pode ser
  deduzida do ambiente.
- `TELEGRAM_BOT_TOKEN` e `TELEGRAM_ALLOWED_CHAT_ID`: ativam e restringem o bot.
- `TELEGRAM_GEMINI_MODEL`: permite usar outro modelo Gemini somente na conversa
  comum do Telegram.
- `GAIA_MONITOR_INTERVAL_SECONDS`: determina o intervalo de verificação do
  serviço oficial, respeitando o mínimo de 60 segundos.

As chaves devem ser cadastradas como secrets do Codespaces. Elas nunca devem
ser escritas no código, no README, em mensagens públicas ou em commits.

## Limitações conhecidas

O sistema depende da disponibilidade do Gemini, da Hugging Face, da API oficial
do exercício e das páginas consultadas. Um site pode bloquear leitura
automatizada, uma transcrição do YouTube pode não existir e uma imagem pode ser
interpretada incorretamente pelo modelo visual.

O detector de estagnação evita desperdício, mas não transforma evidência
insuficiente em resposta correta. Quando não há material confiável, o melhor
comportamento é não salvar uma resposta inventada e permitir que a questão seja
executada novamente ou revisada manualmente.

O Codespaces é um ambiente de desenvolvimento. Ele pode ser suspenso por
inatividade. Quando isso acontece, a interface, o bot e o monitor deixam de
funcionar até que o ambiente seja iniciado novamente.

## Resumo operacional

Em termos simples, o agente opera assim: recebe uma pergunta, procura primeiro o
caminho mais específico e econômico, coleta evidências com ferramentas
controladas, deixa o Gemini decidir a resposta, reduz essa resposta ao formato
exato, realiza uma revisão final e entrega o resultado ao usuário. O envio
oficial permanece separado e depende de uma ação explícita depois da revisão
das 20 respostas.
