# -*- coding: utf-8 -*-
"""
Extrai probabilidades implícitas do mercado "Vencedor da Copa 2026" em
mercados de previsão:

  - Kalshi:     série KXMENWORLDCUP (API pública trade-api/v2)
                https://kalshi.com/markets/kxmenworldcup
  - Polymarket: evento world-cup-winner (API pública Gamma)
                https://polymarket.com/event/world-cup-winner

Preço de um contrato "Sim" (0 a 1 dólar) ~ probabilidade implícita.
Para cada fonte usamos o ponto médio bid/ask quando disponível (senão o
último negócio) e normalizamos para somar 1 (remove o overround/viés).

IMPORTANTE: esses domínios costumam ser bloqueados no DNS de provedores
brasileiros; o script contorna isso resolvendo via DNS-over-HTTPS
(ver util_rede.aplicar_contorno_dns).

Uso (a partir da pasta Simulacao-Aplicativo-Streamlit):
  & "C:\\Users\\Pichau\\anaconda3\\python.exe" extracao\\extrair_mercados_predicao.py

Saída: dataset/mercados_predicao_AAAA-MM-DD.xlsx
A coluna 'prob_implicita_media_normalizada' é compatível com
experimento_calibracao_mercado.load_market_target(path).
"""
import argparse
import sys
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from util_rede import UA, aplicar_contorno_dns

PASTA_DATASET = Path(__file__).resolve().parents[1] / 'dataset'

KALSHI_API = 'https://api.elections.kalshi.com/trade-api/v2/markets'
KALSHI_SERIE = 'KXMENWORLDCUP'
POLYMARKET_API = 'https://gamma-api.polymarket.com/events'
POLYMARKET_SLUG = 'world-cup-winner'

# Nomes diferentes entre fontes -> nome padrão (o do Polymarket/dataset)
SINONIMOS = {
    'usa': 'united states',
    'south korea': 'korea republic',
    'czechia': 'czech republic',
    'ivory coast': "côte d'ivoire",
    'cape verde': 'cabo verde',
    'turkey': 'türkiye',
    'iran': 'ir iran',
    'congo dr': 'dr congo',
    'curacao': 'curaçao',
    'bosnia': 'bosnia and herzegovina',
    'bosnia-herzegovina': 'bosnia and herzegovina',
    'turkiye': 'türkiye',
}


def chave_selecao(nome):
    """Chave canônica para casar seleções entre fontes."""
    if not nome:
        return ''
    chave = str(nome).strip().lower()
    return SINONIMOS.get(chave, chave)


def _f(valor):
    try:
        v = float(valor)
        return v
    except (TypeError, ValueError):
        return None


def extrair_kalshi():
    """Retorna {chave_selecao: (nome, preco_yes)} da série KXMENWORLDCUP."""
    mercados, cursor = [], None
    while True:
        params = {'series_ticker': KALSHI_SERIE, 'status': 'open', 'limit': 200}
        if cursor:
            params['cursor'] = cursor
        r = requests.get(KALSHI_API, headers=UA, params=params, timeout=30)
        r.raise_for_status()
        dados = r.json()
        mercados.extend(dados.get('markets', []))
        cursor = dados.get('cursor')
        if not cursor or not dados.get('markets'):
            break

    resultado = {}
    for m in mercados:
        nome = m.get('yes_sub_title') or m.get('title')
        bid = _f(m.get('yes_bid_dollars'))
        ask = _f(m.get('yes_ask_dollars'))
        ultimo = _f(m.get('last_price_dollars'))
        if bid is not None and ask is not None and ask > 0:
            preco = (bid + ask) / 2
        else:
            preco = ultimo
        if nome and preco is not None:
            resultado[chave_selecao(nome)] = (nome, preco)
    return resultado


def extrair_polymarket():
    """Retorna {chave_selecao: (nome, preco_yes)} do evento world-cup-winner."""
    r = requests.get(POLYMARKET_API, headers=UA,
                     params={'slug': POLYMARKET_SLUG}, timeout=30)
    r.raise_for_status()
    eventos = r.json()
    if not eventos:
        raise RuntimeError(f'Polymarket não retornou o evento {POLYMARKET_SLUG}')

    resultado = {}
    for m in eventos[0].get('markets', []):
        if m.get('closed') or not m.get('active'):
            continue
        nome = m.get('groupItemTitle')
        bid = _f(m.get('bestBid'))
        ask = _f(m.get('bestAsk'))
        ultimo = _f(m.get('lastTradePrice'))
        if bid is not None and ask is not None and ask > 0:
            preco = (bid + ask) / 2
        else:
            preco = ultimo
        if nome and preco is not None:
            resultado[chave_selecao(nome)] = (nome, preco)
    return resultado


def _normalizar(df, fontes):
    """(Re)calcula as colunas normalizadas e a média sobre as linhas de df."""
    colunas_norm = []
    for f in fontes:
        bruta = pd.to_numeric(df[f'prob_{f}_extraida'], errors='coerce')
        soma = bruta.sum()
        df[f'prob_{f}_normalizada'] = (bruta / soma) if soma and soma > 0 else None
        colunas_norm.append(f'prob_{f}_normalizada')

    # Média das probabilidades normalizadas das fontes disponíveis por seleção,
    # renormalizada para somar 1
    media = df[colunas_norm].mean(axis=1, skipna=True)
    df['prob_normalizada_media'] = media / media.sum()
    # Apelido compatível com experimento_calibracao_mercado.load_market_target
    df['prob_implicita_media_normalizada'] = df['prob_normalizada_media']
    return df


def montar_tabela(fontes):
    """
    Junta as fontes em um DataFrame com probabilidade extraída (preço do
    contrato "Sim") e normalizada por fonte, mais a média normalizada.
    fontes: {'kalshi': {chave: (nome, preco)}, 'polymarket': {...}}
    """
    chaves = sorted(set().union(*[f.keys() for f in fontes.values()]))
    linhas = []
    for chave in chaves:
        nome_exibicao = next(
            (fontes[f][chave][0] for f in fontes if chave in fontes[f]), chave)
        linha = {'Selecao': nome_exibicao, 'team_key': chave}
        for f in fontes:
            par = fontes[f].get(chave)
            linha[f'prob_{f}_extraida'] = par[1] if par else None
        linhas.append(linha)
    df = pd.DataFrame(linhas)

    df = _normalizar(df, fontes)
    df = df.sort_values('prob_normalizada_media',
                        ascending=False).reset_index(drop=True)
    df['Data_Extracao'] = datetime.now().strftime('%Y-%m-%d %H:%M')
    return df


def alinhar_com_dataset(df, fontes):
    """
    Alinha a tabela com as 48 seleções da Copa (dataset enriquecido mais
    recente do app), renormalizando as probabilidades sobre essas 48.
    Retorna (df_alinhado, selecoes_sem_mercado, mercados_fora_da_copa).
    """
    candidatos = sorted(PASTA_DATASET.glob('FIFA_ELO_DadosSeleções_*.xlsx'))
    if not candidatos:
        raise SystemExit(f'Nenhum FIFA_ELO_DadosSeleções_*.xlsx em {PASTA_DATASET}')
    base = pd.read_excel(candidatos[-1])[['Seleção', 'NomeIngles', 'Grupo']]
    base['team_key'] = base['NomeIngles'].map(chave_selecao)

    alinhado = base.merge(df.drop(columns=['Selecao']), on='team_key', how='left')
    alinhado = alinhado.rename(columns={'NomeIngles': 'Selecao',
                                        'Seleção': 'Selecao_PT'})

    col_extraidas = [f'prob_{f}_extraida' for f in fontes]
    sem_mercado = alinhado[alinhado[col_extraidas].isna().all(axis=1)]['Selecao'].tolist()
    fora_da_copa = sorted(set(df['team_key']) - set(base['team_key']))

    alinhado = _normalizar(alinhado, fontes)
    alinhado = alinhado.sort_values('prob_normalizada_media',
                                    ascending=False).reset_index(drop=True)
    ordem = ['Selecao', 'Selecao_PT', 'Grupo', 'team_key'] + \
            [c for c in alinhado.columns
             if c not in ('Selecao', 'Selecao_PT', 'Grupo', 'team_key')]
    return alinhado[ordem], sem_mercado, fora_da_copa


def main():
    parser = argparse.ArgumentParser(
        description='Extrai probabilidades implícitas de mercados de previsão (Kalshi + Polymarket)')
    parser.add_argument('--saida', default=None, help='caminho do xlsx de saída')
    parser.add_argument('--etapa', default=None,
                        help='rótulo da etapa no nome do arquivo (ex.: "inicio da copa")')
    parser.add_argument('--todas', action='store_true',
                        help='mantém todas as seleções dos mercados em vez de '
                             'alinhar com as 48 da Copa do dataset')
    args = parser.parse_args()

    aplicar_contorno_dns()

    fontes = {}
    print('Baixando Kalshi (KXMENWORLDCUP)...')
    try:
        fontes['kalshi'] = extrair_kalshi()
        print(f'  {len(fontes["kalshi"])} seleções.')
    except Exception as e:
        print(f'  AVISO: Kalshi falhou ({e}); seguindo sem essa fonte.')
    print('Baixando Polymarket (world-cup-winner)...')
    try:
        fontes['polymarket'] = extrair_polymarket()
        print(f'  {len(fontes["polymarket"])} seleções.')
    except Exception as e:
        print(f'  AVISO: Polymarket falhou ({e}); seguindo sem essa fonte.')

    if not fontes:
        raise SystemExit('Nenhuma fonte disponível — abortando.')

    df = montar_tabela(fontes)
    if not args.todas:
        df, sem_mercado, fora_da_copa = alinhar_com_dataset(df, fontes)
        print(f'\nAlinhado com as {len(df)} seleções da Copa.')
        if sem_mercado:
            print(f'  AVISO — sem preço em nenhuma fonte: {sem_mercado}')
        if fora_da_copa:
            print(f'  Ignorados (mercados fora da Copa): {fora_da_copa}')

    colunas_mostrar = ['Selecao'] + \
        [c for c in df.columns if c.startswith('prob_') and c != 'prob_implicita_media_normalizada']
    print('\nTop 10:')
    print(df[colunas_mostrar].head(10).to_string(index=False))

    rotulo_etapa = ''
    if args.etapa:
        rotulo_etapa = '_' + args.etapa.strip().lower().replace(' ', '_')
    saida = Path(args.saida) if args.saida else (
        PASTA_DATASET / f'mercados_predicao{rotulo_etapa}_{date.today().isoformat()}.xlsx')
    saida.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(saida, index=False)
    print(f'\nSalvo em: {saida}')


if __name__ == '__main__':
    main()
