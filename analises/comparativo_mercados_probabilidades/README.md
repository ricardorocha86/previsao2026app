# Comparativo de mercados de probabilidades

Pasta dedicada para guardar a tabela mestre que compara probabilidades normalizadas
de tres fontes:

- Kalshi
- Polymarket
- Oddschecker

## Estrutura

- `resultados/AAAA-MM-DD/`: tabelas mestre finais, em XLSX.
- `fontes/AAAA-MM-DD/`: planilhas-fonte usadas para gerar a tabela mestre da data.
- `scripts/`: scripts relacionados a extracao e geracao da tabela comparativa.

## Arquivo principal

O arquivo mais importante de cada data segue este padrao:

`TABELA_MESTRA_probabilidades_normalizadas_Kalshi_Polymarket_Oddschecker_AAAA-MM-DD.xlsx`

Colunas:

- `Selecao`
- `Kalshi`
- `Polymarket`
- `Oddschecker`
- `Media_3_fontes`

As quatro colunas de probabilidade ficam formatadas como percentual e recebem
escala condicional azul: quanto mais forte o azul, maior a probabilidade.

## Regenerar uma data

Exemplo:

```powershell
& "C:\Users\Pichau\anaconda3\python.exe" `
  ".\scripts\gerar_tabela_mestra_comparativo.py" `
  --data "2026-06-11" `
  --mercados ".\fontes\2026-06-11\mercados_predicao_inicio_da_copa_2026-06-11.xlsx" `
  --oddschecker ".\fontes\2026-06-11\oddschecker_tabela_com_probs_2026-06-11.xlsx"
```
