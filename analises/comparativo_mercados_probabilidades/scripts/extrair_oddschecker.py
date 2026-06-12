# -*- coding: utf-8 -*-
"""
Extrai as odds de "Vencedor da Copa 2026" do Oddschecker (agregador de
casas de apostas) e calcula as probabilidades implícitas.

ATENÇÃO: o Oddschecker fica atrás do Cloudflare e bloqueia requisições
automatizadas (403 "Just a moment..."). O script tenta o acesso direto,
mas o caminho confiável é o modo --html:

  1. Abra https://www.oddschecker.com/football/world-cup/world-cup/winner
     no navegador.
  2. Salve a página (Ctrl+S -> "Página da web, somente HTML").
  3. Rode:
     & "C:\\Users\\Pichau\\anaconda3\\python.exe" extracao\\extrair_oddschecker.py --html "caminho\\pagina.html"

Saída: dataset/oddschecker_tabela_com_probs_AAAA-MM-DD.xlsx, no mesmo
formato do dataset/oddschecker_tabela_com_probs.xlsx usado pelo app
(Selecao, odd_N, prob_implicita_N, prob_implicita_media,
prob_implicita_media_normalizada).

Use --atualizar-app para também sobrescrever o arquivo oficial
dataset/oddschecker_tabela_com_probs.xlsx (faz backup .bak antes).
"""
import argparse
import re
import shutil
import sys
from datetime import date
from fractions import Fraction
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from util_rede import UA

PASTA_DATASET = Path(__file__).resolve().parents[1] / 'dataset'
URL = 'https://www.oddschecker.com/football/world-cup/world-cup/winner'
ARQUIVO_OFICIAL = PASTA_DATASET / 'oddschecker_tabela_com_probs.xlsx'
CASAS_PADRAO = [
    'Bet365',
    'William Hill',
    'Unibet',
    'Betfred',
    '888sport',
    'Spreadex',
    'Ladbrokes',
    'BetVictor',
    'BetMGM UK',
    'BOYLE Sports',
    '10bet',
    'Star Sports',
    'PricedUp',
    'Sporting Index',
    'BetGoodwin',
    'Virgin Bet',
    'QuinnBet',
    'Betway',
    'Coral',
    'BetAhoy',
    'BetTom',
    'BresBet',
    'Skybet',
    'Paddy Power',
    'Betfair',
    'Matchbook',
]


def odd_fracionaria_para_prob(odd):
    """Converte odd fracionária ('9/2', 'EVS') ou decimal em probabilidade implícita."""
    texto = str(odd).strip().upper()
    if texto in ('EVS', 'EVENS'):
        return 0.5
    try:
        if '/' in texto:
            fr = Fraction(texto)
            return float(1 / (fr + 1))
        fr = Fraction(texto)
        if fr >= 0:
            return float(1 / (fr + 1))
    except (ValueError, ZeroDivisionError):
        pass
    return None


def baixar_html_direto():
    """Tenta baixar a página direto (geralmente bloqueado pelo Cloudflare)."""
    r = requests.get(URL, headers=UA, timeout=30)
    if r.status_code != 200 or 'Just a moment' in r.text[:2000]:
        raise RuntimeError(
            f'Oddschecker bloqueou o acesso direto (HTTP {r.status_code}). '
            'Salve a página no navegador e use --html (ver docstring).')
    return r.text


def extrair_de_html(html):
    """
    Extrai {selecao: [odds fracionárias]} do HTML da página de winner.
    A tabela tem linhas <tr> com o nome em data-bname/data-name e células
    <td> de odds com data-o (fracionária) e/ou data-odig (decimal).
    """
    tabela = {}
    for trecho in re.split(r'<tr[\s>]', html)[1:]:
        trecho = trecho.split('</tr>')[0]
        m_nome = re.search(r'data-b?name="([^"]+)"', trecho)
        if not m_nome:
            continue
        nome = m_nome.group(1).strip()
        odds = re.findall(r'<td[^>]*\bdata-o="([^"]*)"[^>]*>', trecho)
        if not odds:
            # fallback: usa as decimais
            odds = re.findall(r'<td[^>]*\bdata-odig="([^"]*)"[^>]*>', trecho)
        odds = [o for o in odds if o and o not in ('SP',)]
        if odds and len(odds) >= 3:
            tabela[nome] = odds
    if not tabela:
        raise RuntimeError(
            'Não encontrei a tabela de odds no HTML — o layout pode ter mudado. '
            'Confira se salvou a página certa (winner).')
    return tabela


def extrair_nomes_casas(html):
    """Extrai o mapa codigo -> nome da casa do JavaScript embutido na pagina."""
    nomes = {}
    for codigo, nome in re.findall(r"oc\.bookiesNames\.([A-Z0-9]+)\s*=\s*'([^']+)'", html):
        nomes[codigo] = nome
    return nomes


def extrair_de_html(html):
    """
    Extrai {selecao: [(codigo_casa, nome_casa, odd_fracionaria)]}.
    Usa data-bk em cada celula para evitar deslocar casas quando ha odds vazias.
    """
    nomes_casas = extrair_nomes_casas(html)
    tabela = {}
    for trecho in re.split(r'<tr[\s>]', html)[1:]:
        trecho = trecho.split('</tr>')[0]
        m_nome = re.search(r'data-b?name="([^"]+)"', trecho)
        if not m_nome:
            continue
        nome = m_nome.group(1).strip()
        odds = []
        for td in re.findall(r'<td\b[^>]*>', trecho):
            m_bk = re.search(r'\bdata-bk="([^"]*)"', td)
            m_odd = re.search(r'\bdata-o="([^"]*)"', td)
            if not m_bk or not m_odd:
                continue
            codigo = m_bk.group(1).strip()
            odd = m_odd.group(1).strip()
            if not odd or odd == 'SP':
                continue
            odds.append((codigo, nomes_casas.get(codigo, codigo), odd))
        if odds and len(odds) >= 3:
            tabela[nome] = odds
    if not tabela:
        raise RuntimeError(
            'Nao encontrei a tabela de odds no HTML. '
            'Confira se salvou a pagina certa (winner).')
    return tabela


def montar_tabela(odds_por_selecao):
    """Monta o DataFrame no mesmo esquema do oddschecker_tabela_com_probs.xlsx."""
    n_max = max(len(v) for v in odds_por_selecao.values())
    linhas = []
    for selecao, odds in odds_por_selecao.items():
        linha = {'Selecao': selecao}
        probs = []
        for i in range(n_max):
            odd_info = odds[i] if i < len(odds) else None
            odd = odd_info[2] if isinstance(odd_info, tuple) else odd_info
            linha[f'odd_{i + 1}'] = odd
            prob = odd_fracionaria_para_prob(odd) if odd else None
            linha[f'prob_implicita_{i + 1}'] = prob
            if prob is not None:
                probs.append(prob)
        linha['prob_implicita_media'] = sum(probs) / len(probs) if probs else None
        linhas.append(linha)
    df = pd.DataFrame(linhas)
    soma = df['prob_implicita_media'].sum()
    df['prob_implicita_media_normalizada'] = df['prob_implicita_media'] / soma
    return df.sort_values('prob_implicita_media_normalizada',
                          ascending=False).reset_index(drop=True)


def montar_tabela_por_casa(odds_por_selecao):
    """Monta uma tabela longa para analise por casa de aposta."""
    linhas = []
    for selecao, odds in odds_por_selecao.items():
        for i, odd_info in enumerate(odds):
            if isinstance(odd_info, tuple):
                codigo_casa, casa, odd = odd_info
            else:
                codigo_casa = None
                casa = CASAS_PADRAO[i] if i < len(CASAS_PADRAO) else f'casa_{i + 1}'
                odd = odd_info
            prob = odd_fracionaria_para_prob(odd) if odd else None
            linhas.append({
                'Selecao': selecao,
                'CodigoCasa': codigo_casa,
                'Casa': casa,
                'odd_fracionaria': odd,
                'prob_implicita': prob,
                'prob_implicita_pct': prob * 100 if prob is not None else None,
            })
    return pd.DataFrame(linhas)


def main():
    parser = argparse.ArgumentParser(
        description='Extrai odds do Oddschecker (vencedor da Copa 2026)')
    parser.add_argument('--html', default=None,
                        help='caminho de um HTML salvo da página (contorna o Cloudflare)')
    parser.add_argument('--atualizar-app', action='store_true',
                        help='sobrescreve dataset/oddschecker_tabela_com_probs.xlsx (com backup)')
    parser.add_argument('--saida', default=None, help='caminho do xlsx de saída')
    args = parser.parse_args()

    if args.html:
        html = Path(args.html).read_text(encoding='utf-8', errors='ignore')
    else:
        print('Tentando acesso direto (provavelmente bloqueado pelo Cloudflare)...')
        html = baixar_html_direto()

    odds = extrair_de_html(html)
    print(f'{len(odds)} seleções extraídas.')
    df = montar_tabela(odds)
    print(df[['Selecao', 'prob_implicita_media',
              'prob_implicita_media_normalizada']].head(10).to_string(index=False))

    saida = Path(args.saida) if args.saida else (
        PASTA_DATASET / f'oddschecker_tabela_com_probs_{date.today().isoformat()}.xlsx')
    df.to_excel(saida, index=False)
    print(f'Salvo em: {saida}')

    saida_long = saida.with_name(saida.stem + '_por_casa.xlsx')
    df_long = montar_tabela_por_casa(odds)
    df_long.to_excel(saida_long, index=False)
    df_long.to_csv(saida_long.with_suffix('.csv'), index=False, encoding='utf-8-sig')
    print(f'Tabela por casa salva em: {saida_long}')
    print(f'CSV por casa salvo em: {saida_long.with_suffix(".csv")}')

    if args.atualizar_app:
        if ARQUIVO_OFICIAL.exists():
            shutil.copy2(ARQUIVO_OFICIAL, ARQUIVO_OFICIAL.with_suffix('.xlsx.bak'))
        df.to_excel(ARQUIVO_OFICIAL, index=False)
        print(f'Arquivo oficial atualizado: {ARQUIVO_OFICIAL} (backup .bak criado)')


if __name__ == '__main__':
    main()
