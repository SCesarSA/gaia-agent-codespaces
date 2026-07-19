# GAIA Agent — GitHub Codespaces

Agente educacional para executar e revisar as perguntas da avaliação GAIA com
Gradio, Gemini e ferramentas locais/web.

Para uma explicação completa, em linguagem natural, consulte
[Como o agente GAIA funciona](DOCUMENTACAO_AGENTE.md).

## Executar no GitHub Codespaces

1. No GitHub, abra este repositório.
2. Clique em **Code → Codespaces → Create codespace on main**.
3. Aguarde o comando automático de instalação terminar.
4. No terminal do Codespace, execute:

   ```bash
   python app.py
   ```

5. Abra a porta **7860** quando o Codespaces exibir a notificação. A porta é
   privada por padrão.

### Se aparecer `No module named 'gradio'`

No Codespace atual, instale imediatamente:

```bash
python -m pip install -r requirements.txt
python app.py
```

Depois de receber uma atualização do arquivo `.devcontainer/Dockerfile`, use a
Paleta de Comandos (`Ctrl+Shift+P`) e execute:

```text
Codespaces: Rebuild Container
```

As dependências passam a ser instaladas durante a construção da imagem.

## Secrets necessários

Cadastre em **Settings → Secrets and variables → Codespaces**:

- `GEMINI_API_KEY`: modelo principal e revisão das respostas.
- `HF_TOKEN`: questões e anexos do dataset GAIA quando a API do exercício
  estiver indisponível.

Antes de usar `HF_TOKEN`, aceite as condições de acesso na página do
[dataset GAIA](https://huggingface.co/datasets/gaia-benchmark/GAIA). O token
precisa ter permissão de leitura.

O Codespaces fornece `GITHUB_REPOSITORY` automaticamente. Para identificar o
usuário de outra forma, configure opcionalmente:

- `HF_USERNAME`: nome de usuário usado no envio.
- `AGENT_CODE_URL`: URL pública do código do agente; se omitida no Codespaces,
  o app usa automaticamente a URL do repositório GitHub.

Nunca coloque chaves diretamente no `app.py`, no README ou em arquivos enviados
ao GitHub.

## Executar localmente

```bash
python -m venv .venv
```

No Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
$env:GEMINI_API_KEY="..."
$env:HF_TOKEN="..."
python app.py
```

A interface ficará disponível em `http://127.0.0.1:7860`.

## Falha 500 na API do exercício

Se `/random-question` ou `/questions` retornar erro 500, o app reconstrói
automaticamente o mesmo conjunto de perguntas diretamente do dataset GAIA.
Isso mantém disponíveis:

- o sorteio de uma pergunta;
- a execução individual ou das 20 questões;
- o download dos anexos oficiais.

O envio da pontuação continua dependendo do endpoint oficial `/submit`. Se esse
endpoint também estiver fora do ar, as respostas permanecem na tabela para
revisão e poderão ser enviadas quando o serviço voltar.

## Limitações do Codespaces

O Codespaces é indicado para desenvolvimento e testes. Ele pode parar por
inatividade e não deve ser tratado como hospedagem pública permanente.

## Bot do Telegram

O mesmo processo de `app.py` pode iniciar um bot do Telegram. O bot conversa
usando o modelo configurado e chama diretamente as funções do agente GAIA; ele
não compartilha esta conversa do Codex nem controla uma aba já aberta.

### Segurança obrigatória

Se um token foi publicado em uma conversa ou arquivo, revogue-o imediatamente
com `/revoke` no BotFather e gere outro. Nunca coloque o token no código.

Cadastre o novo valor em **Settings → Secrets and variables → Codespaces**:

- `TELEGRAM_BOT_TOKEN`: novo token gerado pelo BotFather.
- `TELEGRAM_ALLOWED_CHAT_ID`: único chat autorizado a controlar o agente.
- `TELEGRAM_GEMINI_MODEL`: opcional; por padrão usa `gemini-3.5-flash`.

O agente e a conversa do Telegram utilizam exclusivamente o Gemini. Para trocar
o modelo Gemini principal, configure `GAIA_GEMINI_MODEL`; para trocar somente o
modelo da conversa, configure `TELEGRAM_GEMINI_MODEL`.

Por padrão, `GAIA_MAX_STEPS=unlimited`: o agente continua pesquisando até
produzir uma resposta, receber um erro do provedor ou ser interrompido. Como o
`smolagents` exige internamente um inteiro, o aplicativo usa um valor
praticamente inalcançável. Para restaurar uma proteção, configure um número,
por exemplo `GAIA_MAX_STEPS=30`.

O limite numérico de passos não encerra a pesquisa, mas um detector de
estagnação evita ciclos que repetem buscas sem avançar. Ao detectar repetição,
o agente interrompe as ferramentas e pede ao próprio Gemini que consolide a
melhor resposta usando as evidências já coletadas. Os valores podem ser
ajustados com `GAIA_STAGNATION_TOOL_CALLS`,
`GAIA_STAGNATION_SEARCH_CALLS` e `GAIA_STAGNATION_DUPLICATES`.

Perguntas sobre artigos destacados da Wikipédia promovidos em um mês específico
seguem uma rota direta pela API pública da Wikipédia. A ferramenta
`wikipedia_featured_articles` extrai somente as linhas daquele mês, identifica
o tema pelos resumos e categorias e entrega artigo e nomeador ao Gemini. Isso
evita buscas repetidas e páginas inteiras no contexto do modelo.

### Descobrir o `chat_id`

1. Cadastre somente o novo `TELEGRAM_BOT_TOKEN`.
2. Reinicie o Codespace e execute `python app.py`.
3. Abra o bot no Telegram, pressione **Start** e envie `/id`.
4. Cadastre o número retornado como `TELEGRAM_ALLOWED_CHAT_ID`.
5. Reinicie novamente o processo.

Enquanto `TELEGRAM_ALLOWED_CHAT_ID` não estiver configurado, somente `/id`
funciona.

### Comandos

- `/status`: verifica a API oficial e o fallback direto do GAIA.
- `/carregar20`: carrega as questões sem consumir chamadas de LLM.
- `/listar`: lista os Task IDs e uma prévia de cada questão.
- `/sortear`: seleciona uma questão oficial aleatória.
- `/questao TASK_ID`: seleciona uma questão específica.
- `/executar`: executa a questão selecionada.
- `/reexecutar TASK_ID`: executa novamente uma questão específica.
- `/executar20`: executa diretamente todas as questões carregadas.
- `/parar`: interrompe o lote após a conclusão da questão atual.
- `/progresso`: mostra quantas respostas estão prontas.
- `/respostas`: mostra as respostas mantidas na sessão do bot.
- `/definir TASK_ID | RESPOSTA`: permite editar uma resposta.
- `/enviar confirmar`: envia as 20 respostas para a API oficial.
- `/limpar`: remove o histórico e as respostas locais da sessão.

Mensagens comuns, sem `/`, são encaminhadas ao modelo para conversa. O histórico
é curto e mantido somente na memória do processo.

O bot verifica o endpoint oficial a cada cinco minutos e envia uma mensagem
quando ele passa de indisponível para disponível. Esse monitor só funciona
enquanto o Codespace estiver ativo; o Codespaces pode suspender por inatividade.
