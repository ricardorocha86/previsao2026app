# -*- coding: utf-8 -*-
"""
Atualiza o dataset enriquecido FIFA_ELO_DadosSeleções_*.xlsx com dados
novos de Elo (eloratings.net) e FIFA (inside.fifa.com).

Lê o arquivo FIFA_ELO_DadosSeleções_*.xlsx mais recente da pasta dataset,
atualiza as colunas ELO_* e FIFA_* (casando por ELO_Codigo / FIFA_Code) e
grava um novo arquivo com a data de hoje. As colunas estáticas (população,
valor de mercado, histórico de copas etc.) são preservadas.

O app escolhe automaticamente o FIFA_ELO_DadosSeleções_*.xlsx mais novo
(utils/forca_core.find_latest_enriched_dataset), então basta rodar isto e
recarregar o app.

Colunas FIFA não disponíveis na API atual (FIFA_Highest_Rank, FIFA_Lowest_Rank,
FIFA_Average_Rank, FIFA_Biggest_Climb/Fall) ficam com os valores antigos.

Uso (a partir da pasta Simulacao-Aplicativo-Streamlit):
  & "C:\\Users\\Pichau\\anaconda3\\python.exe" extracao\\atualizar_dataset_selecoes.py
  ... --sem-forma   # pula a forma recente (mais rápido; ~1 request por seleção a menos)
  ... --fifa-live   # usa o ranking FIFA ao vivo em vez do último oficial
"""
import argparse
import sys
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import extrair_elo
import extrair_fifa

PASTA_DATASET = Path(__file__).resolve().parents[1] / 'dataset'

MAPA_ELO = {
    'Rank': 'ELO_Ranking', 'Rating': 'ELO_Rating',
    'Rank_Max': 'ELO_Rank_Max', 'Rating_Max': 'ELO_Rating_Max',
    'Rank_Avg': 'ELO_Rank_Avg', 'Rating_Avg': 'ELO_Rating_Avg',
    'Rank_Min': 'ELO_Rank_Min', 'Rating_Min': 'ELO_Rating_Min',
    'Rank_Chg_3M': 'ELO_Rank_Chg_3M', 'Chg_3M': 'ELO_Chg_3M',
    'Rank_Chg_6M': 'ELO_Rank_Chg_6M', 'Chg_6M': 'ELO_Chg_6M',
    'Rank_Chg_1A': 'ELO_Rank_Chg_1A', 'Chg_1A': 'ELO_Chg_1A',
    'Rank_Chg_2A': 'ELO_Rank_Chg_2A', 'Chg_2A': 'ELO_Chg_2A',
    'Rank_Chg_5A': 'ELO_Rank_Chg_5A', 'Chg_5A': 'ELO_Chg_5A',
    'Rank_Chg_10A': 'ELO_Rank_Chg_10A', 'Chg_10A': 'ELO_Chg_10A',
    'Total_Jogos': 'ELO_Total_Jogos', 'Jogos_Casa': 'ELO_Jogos_Casa',
    'Jogos_Fora': 'ELO_Jogos_Fora', 'Jogos_Neutro': 'ELO_Jogos_Neutro',
    'Vitorias': 'ELO_Vitorias', 'Derrotas': 'ELO_Derrotas', 'Empates': 'ELO_Empates',
    'Gols_Pro': 'ELO_Gols_Pro', 'Gols_Contra': 'ELO_Gols_Contra',
    'Saldo_Gols': 'ELO_Saldo_Gols', 'Aproveitamento': 'ELO_Aproveitamento',
    'Media_Gols_Pro': 'ELO_Media_Gols_Pro', 'Media_Gols_Contra': 'ELO_Media_Gols_Contra',
    'Forma_Recente': 'ELO_Forma_Recente',
    'Vitorias_Recentes': 'ELO_Vitorias_Recentes',
    'Derrotas_Recentes': 'ELO_Derrotas_Recentes',
    'Empates_Recentes': 'ELO_Empates_Recentes',
    'Media_Gols_Recente': 'ELO_Media_Gols_Recente',
}

COLUNAS_FIFA = [
    'FIFA_Current_Rank', 'FIFA_Current_Points', 'FIFA_Previous_Rank',
    'FIFA_Previous_Points', 'FIFA_Points_Difference', 'FIFA_Flag_URL',
]


def achar_dataset_mais_recente():
    candidatos = sorted(PASTA_DATASET.glob('FIFA_ELO_DadosSeleções_*.xlsx'))
    if not candidatos:
        raise SystemExit(f'Nenhum FIFA_ELO_DadosSeleções_*.xlsx em {PASTA_DATASET}')
    return candidatos[-1]


def main():
    parser = argparse.ArgumentParser(
        description='Atualiza o dataset enriquecido com Elo e FIFA novos')
    parser.add_argument('--sem-forma', action='store_true',
                        help='não recalcula a forma recente (ELO_Forma_Recente etc.)')
    parser.add_argument('--fifa-live', action='store_true',
                        help='usa o ranking FIFA ao vivo em vez do último publicado')
    parser.add_argument('--etapa', default=None,
                        help='rótulo da etapa no nome do arquivo (ex.: "inicio da copa"); '
                             'entra depois da data para manter a ordenação do glob do app')
    args = parser.parse_args()

    origem = achar_dataset_mais_recente()
    print(f'Dataset base: {origem.name}')
    df = pd.read_excel(origem)
    codigos_elo = df['ELO_Codigo'].dropna().astype(str).tolist()

    # --- Elo ---
    print('\n[1/2] Extraindo Elo (eloratings.net)...')
    df_elo = extrair_elo.extrair(incluir_forma=not args.sem_forma,
                                 apenas_codigos=codigos_elo)
    elo_por_codigo = df_elo.set_index('Codigo').to_dict('index')

    atualizadas, sem_dado = 0, []
    for idx, linha in df.iterrows():
        codigo = str(linha.get('ELO_Codigo', '') or '')
        dado = elo_por_codigo.get(codigo)
        if not dado:
            sem_dado.append(linha.get('Seleção'))
            continue
        for col_tsv, col_dataset in MAPA_ELO.items():
            if col_dataset not in df.columns or col_tsv not in dado:
                continue
            valor = dado[col_tsv]
            if valor is not None and not (isinstance(valor, float) and pd.isna(valor)):
                df.loc[idx, col_dataset] = valor
        atualizadas += 1
    print(f'Elo atualizado para {atualizadas}/{len(df)} seleções.'
          + (f' Sem dado: {sem_dado}' if sem_dado else ''))

    # --- FIFA ---
    print('\n[2/2] Extraindo ranking FIFA (inside.fifa.com)...')
    df_fifa, rotulo = extrair_fifa.extrair(live=args.fifa_live)
    fifa_por_codigo = df_fifa.set_index('FIFA_Code').to_dict('index')

    atualizadas, sem_dado = 0, []
    for idx, linha in df.iterrows():
        codigo = str(linha.get('FIFA_Code', '') or '')
        dado = fifa_por_codigo.get(codigo)
        if not dado:
            sem_dado.append(linha.get('Seleção'))
            continue
        for col in COLUNAS_FIFA:
            if col in df.columns and dado.get(col) is not None:
                df.loc[idx, col] = dado[col]
        atualizadas += 1
    print(f'FIFA ({rotulo}) atualizado para {atualizadas}/{len(df)} seleções.'
          + (f' Sem dado: {sem_dado}' if sem_dado else ''))

    rotulo_etapa = ''
    if args.etapa:
        rotulo_etapa = '_' + args.etapa.strip().lower().replace(' ', '_')
    destino = PASTA_DATASET / (
        f'FIFA_ELO_DadosSeleções_{date.today().isoformat()}{rotulo_etapa}.xlsx')
    df.to_excel(destino, index=False)
    print(f'\nSalvo em: {destino}')
    print('O app passa a usar este arquivo automaticamente (é o mais novo do glob).')


if __name__ == '__main__':
    main()
