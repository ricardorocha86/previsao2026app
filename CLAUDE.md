# Simulacao-Aplicativo-Streamlit

App Streamlit de previsão da Copa do Mundo 2026.

## Como rodar (IMPORTANTE)

O Python deste PC é **Anaconda**. O alias `python`/`python3` do Windows aponta para o
stub da Microsoft Store e **NÃO funciona** — sempre use o interpretador do Anaconda pelo
caminho absoluto:

```powershell
& "C:\Users\Pichau\anaconda3\python.exe" -m streamlit run app.py --server.port 8501 --server.headless true
```

- Rode a partir da pasta `Simulacao-Aplicativo-Streamlit` (onde está o `app.py`).
- Python 3.10.9, Streamlit 1.58.0 — dependências já instaladas no Anaconda base.
- Health check: `http://localhost:8501/_stcore/health` deve retornar `ok`.
- App: `http://localhost:8501`

## Estrutura de páginas (st.navigation em app.py)

- `pages/Explorador_de_Dados.py` — Conjunto de Dados
- `pages/Indicador_de_Força.py` — Indicador de Força (métricas "Efetivo" + Tabela de Força)
- `pages/Partida.py` — Partida (Seleção 1 vs Seleção 2)
- `pages/Explorador_de_Força.py` — Simulação Copa do Mundo 2026 (**página default**, só simulação, sempre Completa)
- `pages/Simulação_Ao_Vivo.py` — Simulação Ao Vivo da Copa

`utils/forca_core.py` centraliza carga de dados, tabela de força, probabilidades de
partida e o `render_param_sidebar()` (pesos + parâmetros do modelo) compartilhado entre
as páginas Indicador de Força, Partida e Simulação.
