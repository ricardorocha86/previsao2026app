# Scripts de extração de dados

Scripts para reextrair os dados que alimentam o app (Elo, FIFA, mercados).
Todos gravam na pasta `dataset/` com a data no nome do arquivo.

Rodar sempre a partir da pasta `Simulacao-Aplicativo-Streamlit`, com o Python
do Anaconda:

```powershell
& "C:\Users\Pichau\anaconda3\python.exe" extracao\<script>.py
```

## Fluxo principal (re-extração completa)

```powershell
# 1. Atualiza o dataset enriquecido (Elo + FIFA) — o app pega o novo arquivo sozinho
& "C:\Users\Pichau\anaconda3\python.exe" extracao\atualizar_dataset_selecoes.py

# 2. Probabilidades implícitas de mercados de previsão (Kalshi + Polymarket)
& "C:\Users\Pichau\anaconda3\python.exe" extracao\extrair_mercados_predicao.py
```

## Scripts

| Script | Fonte | Saída |
|---|---|---|
| `extrair_elo.py` | eloratings.net (TSVs públicos) | `RankingElo_<data>.xlsx` (244 seleções) |
| `extrair_fifa.py` | inside.fifa.com (API interna `get-rankings`) | `RankingFIFA_<data>.xlsx` (211 seleções) |
| `atualizar_dataset_selecoes.py` | os dois acima | `FIFA_ELO_DadosSeleções_<data>.xlsx` |
| `extrair_mercados_predicao.py` | Kalshi + Polymarket (APIs públicas) | `mercados_predicao_<data>.xlsx` |
| `extrair_valor_mercado.py` | Transfermarkt (`statistik/weltrangliste`) | `valor_mercado_selecoes_<data>.xlsx` |
| `extrair_oddschecker.py` | Oddschecker (HTML salvo do navegador) | `oddschecker_tabela_com_probs_<data>.xlsx` |
| `util_rede.py` | — | contorno de DNS compartilhado |

## Notas importantes

- **FIFA**: o ranking oficial sai ~mensalmente; a página indicava próxima
  atualização em **11/06/2026** (ranking pré-Copa). Rode de novo depois dessa
  data. `--live` usa o ranking ao vivo (não oficial). As colunas
  `FIFA_Highest_Rank`/`FIFA_Lowest_Rank`/etc. não existem mais na API e ficam
  com os valores antigos no dataset.
- **Elo**: `atualizar_dataset_selecoes.py` recalcula também a forma recente
  (`ELO_Forma_Recente` = últimos 5 jogos; contagens e média de gols = últimos
  10), com 1 request por seleção. Use `--sem-forma` para pular.
- **Kalshi/Polymarket**: provedores brasileiros bloqueiam esses domínios no
  DNS. `util_rede.aplicar_contorno_dns()` resolve via DNS-over-HTTPS do Google
  e a extração funciona normalmente. A probabilidade usada é o ponto médio
  bid/ask do contrato "Sim", normalizado para somar 1 por fonte. Atenção: na
  Kalshi os azarões têm passo mínimo de 1 cent, então 0.5% pode aparecer para
  times com chance real menor.
- **Valor de mercado**: vem da página `statistik/weltrangliste` do
  Transfermarkt (valor total do elenco, tamanho e média de idade — durante a
  Copa, das convocações finais). As páginas de competição (`/WM26`) dão 404,
  mas essa lista geral funciona com requests + User-Agent de navegador.
  `--atualizar` preenche as 3 colunas no dataset enriquecido mais recente
  (com backup `.bak`).
- **Oddschecker**: protegido por Cloudflare — não dá para baixar direto.
  Abra https://www.oddschecker.com/football/world-cup/world-cup/winner no
  navegador, salve com Ctrl+S ("somente HTML") e rode:
  `extracao\extrair_oddschecker.py --html "caminho\pagina.html"`.
  Com `--atualizar-app` ele sobrescreve o `oddschecker_tabela_com_probs.xlsx`
  oficial (com backup `.bak`), que é o arquivo que o app e o
  `experimento_calibracao_mercado.py` usam.
- O arquivo `mercados_predicao_<data>.xlsx` tem as colunas `Selecao` e
  `prob_implicita_media_normalizada`, então pode ser passado direto para
  `experimento_calibracao_mercado.load_market_target(path)` como alvo de
  calibração alternativo ao Oddschecker.
