# -*- coding: utf-8 -*-
"""
Extrai o Ranking Mundial FIFA (masculino) de inside.fifa.com.

A página usa a API interna /api/live-world-ranking/get-rankings com dois modos:
  - mode=schedule + scheduleId  -> ranking oficial de uma data publicada
  - mode=live                   -> ranking "ao vivo" (não oficial, atualiza por jogo)

A lista de datas publicadas vem do JSON __NEXT_DATA__ da página
(props.pageProps.pageData.ranking.allAvailableDates).

Uso (a partir da pasta Simulacao-Aplicativo-Streamlit):
  & "C:\\Users\\Pichau\\anaconda3\\python.exe" extracao\\extrair_fifa.py          # último oficial
  ... extracao\\extrair_fifa.py --live                                            # ranking ao vivo

Saída: dataset/RankingFIFA_AAAA-MM-DD.xlsx
"""
import argparse
import json
import re
import sys
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from util_rede import UA

PASTA_DATASET = Path(__file__).resolve().parents[1] / 'dataset'
PAGINA = 'https://inside.fifa.com/fifa-world-ranking/men'
API = 'https://inside.fifa.com/api/live-world-ranking/get-rankings'


def listar_datas_publicadas():
    """Retorna a lista [(scheduleId, data_iso), ...] da mais nova para a mais antiga."""
    r = requests.get(PAGINA, headers=UA, timeout=30)
    r.raise_for_status()
    m = re.search(r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>',
                  r.text, re.S)
    if not m:
        raise RuntimeError('Não achei o __NEXT_DATA__ na página da FIFA — layout mudou?')
    dados = json.loads(m.group(1))
    datas = (dados['props']['pageProps']['pageData']['ranking']['allAvailableDates'])
    pares = [(d['id'], d['date']) for d in datas if d.get('id') and d.get('date')]
    pares.sort(key=lambda p: p[1], reverse=True)
    return pares


def baixar_ranking(schedule_id=None, live=False):
    """Baixa o ranking completo (211 seleções) e devolve um DataFrame."""
    params = {'gender': '1', 'locale': 'en', 'count': '300'}
    if live:
        params['mode'] = 'live'
    else:
        params['mode'] = 'schedule'
        params['scheduleId'] = schedule_id
    r = requests.get(API, headers=UA, params=params, timeout=30)
    r.raise_for_status()
    itens = r.json().get('rankings', [])
    if not itens:
        raise RuntimeError(f'API da FIFA retornou 0 seleções (params={params})')

    registros = []
    for item in itens:
        registros.append({
            'FIFA_Current_Rank': item.get('rank'),
            'Team': item.get('teamName'),
            'FIFA_Code': item.get('countryCode'),
            'FIFA_Current_Points': item.get('totalPoints'),
            'FIFA_Previous_Rank': item.get('previousRank'),
            'FIFA_Previous_Points': item.get('previousPoints'),
            'Confederacao': item.get('confederationName'),
            'FIFA_Flag_URL': (item.get('flag') or {}).get('src'),
        })
    df = pd.DataFrame(registros)
    df['FIFA_Points_Difference'] = (
        df['FIFA_Current_Points'] - df['FIFA_Previous_Points']).round(2)
    return df


def extrair(live=False):
    """Extrai o ranking FIFA mais recente. Retorna (df, rotulo_da_data)."""
    if live:
        df = baixar_ranking(live=True)
        rotulo = f'live_{date.today().isoformat()}'
    else:
        datas = listar_datas_publicadas()
        schedule_id, data_ranking = datas[0]
        print(f'Ranking oficial mais recente: {data_ranking} (id {schedule_id})')
        df = baixar_ranking(schedule_id=schedule_id)
        rotulo = data_ranking
    df['Data_Ranking'] = rotulo
    df['Data_Extracao'] = datetime.now().strftime('%Y-%m-%d %H:%M')
    return df, rotulo


def main():
    parser = argparse.ArgumentParser(description='Extrai o Ranking Mundial FIFA')
    parser.add_argument('--live', action='store_true',
                        help='usa o ranking ao vivo (não oficial) em vez do último publicado')
    parser.add_argument('--saida', default=None, help='caminho do xlsx de saída')
    args = parser.parse_args()

    df, rotulo = extrair(live=args.live)
    print(f'{len(df)} seleções extraídas (ranking {rotulo}).')
    print(df[['FIFA_Current_Rank', 'Team', 'FIFA_Current_Points']].head(10).to_string(index=False))

    saida = Path(args.saida) if args.saida else (
        PASTA_DATASET / f'RankingFIFA_{date.today().isoformat()}.xlsx')
    saida.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(saida, index=False)
    print(f'Salvo em: {saida}')


if __name__ == '__main__':
    main()
