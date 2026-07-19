# GAIA Agent — GitHub Codespaces

Agente educacional para executar e revisar as perguntas da avaliação GAIA com
Gradio, Cerebras, Gemini e ferramentas locais/web.

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

- `CEREBRAS_API_KEY`: modelo principal.
- `GEMINI_API_KEY`: fallback e revisão.
- `HF_TOKEN`: anexos, áudio e imagem hospedados no Hugging Face.

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
$env:CEREBRAS_API_KEY="..."
$env:GEMINI_API_KEY="..."
$env:HF_TOKEN="..."
python app.py
```

A interface ficará disponível em `http://127.0.0.1:7860`.

## Limitações fora do Hugging Face Spaces

O Codespaces executa o agente e a interface, mas a API oficial de perguntas,
os anexos do curso, o OAuth e o envio ao leaderboard continuam dependendo do
Hugging Face. Se esses serviços estiverem indisponíveis, mudar a hospedagem não
restaura essas integrações.

O Codespaces é indicado para desenvolvimento e testes. Ele pode parar por
inatividade e não deve ser tratado como hospedagem pública permanente.
