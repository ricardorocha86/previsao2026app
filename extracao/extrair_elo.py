# -*- coding: utf-8 -*-
"""
Extrai o ranking Elo de eloratings.net.

Fontes (TSV públicos do próprio site):
  - https://www.eloratings.net/World.tsv      -> ranking completo (~244 seleções)
  - https://www.eloratings.net/en.teams.tsv   -> mapa código -> nome em inglês
  - https://www.eloratings.net/{Nome}.tsv     -> histórico de jogos por seleção
                                                 (usado para a forma recente)

Uso (a partir da pasta Simulacao-Aplicativo-Streamlit):
  & "C:\\Users\\Pichau\\anaconda3\\python.exe" extracao\\extrair_elo.py
  ... extrair_elo.py --forma   # também busca forma recente (1 request por seleção)

Saída: dataset/RankingElo_AAAA-MM-DD.xlsx
"""
import argparse
import sys
import time
import unicodedata
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from util_rede import UA

PASTA_DATASET = Path(__file__).resolve().parents[1] / 'dataset'
BASE = 'https://www.eloratings.net'

# Ordem das 31 colunas do World.tsv (validada contra o xlsx de 2026-04-15)
COLUNAS_TSV = [
    'Rank', 'Rank_Anterior', 'Codigo', 'Rating',
    'Rank_Max', 'Rating_Max', 'Rank_Avg', 'Rating_Avg', 'Rank_Min', 'Rating_Min',
    'Rank_Chg_3M', 'Chg_3M', 'Rank_Chg_6M', 'Chg_6M', 'Rank_Chg_1A', 'Chg_1A',
    'Rank_Chg_2A', 'Chg_2A', 'Rank_Chg_5A', 'Chg_5A', 'Rank_Chg_10A', 'Chg_10A',
    'Total_Jogos', 'Jogos_Casa', 'Jogos_Fora', 'Jogos_Neutro',
    'Vitorias', 'Derrotas', 'Empates', 'Gols_Pro', 'Gols_Contra',
]


def _num(valor):
    """Converte número do TSV ('+3', '−15', '−', '') para int ou None."""
    if valor is None:
        return None
    texto = str(valor).strip().replace('−', '-').replace('+', '')
    if texto in ('', '-'):
        return None
    try:
        return int(texto)
    except ValueError:
        try:
            return float(texto)
        except ValueError:
            return None


def baixar_ranking():
    """Baixa e estrutura o World.tsv."""
    r = requests.get(f'{BASE}/World.tsv', headers=UA, timeout=30)
    r.raise_for_status()
    r.encoding = 'utf-8'
    linhas = [l.split('\t') for l in r.text.strip().split('\n')]
    registros = []
    for linha in linhas:
        if len(linha) < len(COLUNAS_TSV):
            continue
        registro = dict(zip(COLUNAS_TSV, linha))
        registros.append(registro)
    df = pd.DataFrame(registros)
    for col in df.columns:
        if col != 'Codigo':
            df[col] = df[col].map(_num)
    return df


def baixar_nomes():
    """Baixa o mapa código -> nome em inglês (en.teams.tsv)."""
    r = requests.get(f'{BASE}/en.teams.tsv', headers=UA, timeout=30)
    r.raise_for_status()
    r.encoding = 'utf-8'
    nomes = {}
    for linha in r.text.strip().split('\n'):
        partes = linha.split('\t')
        if len(partes) >= 2:
            nomes[partes[0]] = partes[1]
    return nomes


def baixar_forma_recente(nome_ingles, n_forma=5, n_recente=10):
    """
    Baixa o histórico de jogos da seleção e calcula:
      - forma: string tipo 'W-W-W-D-D' (últimos n_forma jogos, mais recente primeiro)
      - vitórias/derrotas/empates nos últimos n_recente jogos
      - média de gols feitos nos últimos n_recente jogos
    """
    # URLs do site usam ASCII puro (ex.: Curaçao -> Curacao.tsv)
    nome_url = unicodedata.normalize('NFKD', nome_ingles).encode('ascii', 'ignore').decode()
    url = f"{BASE}/{nome_url.replace(' ', '_')}.tsv"
    r = requests.get(url, headers=UA, timeout=30)
    if r.status_code != 200:
        return None
    r.encoding = 'utf-8'

    hoje = date.today()
    jogos = []  # (data, gols_pro, gols_contra)
    for linha in r.text.strip().split('\n'):
        p = linha.split('\t')
        if len(p) < 7:
            continue
        try:
            quando = date(int(p[0]), int(p[1]) or 1, int(p[2]) or 1)
            g1, g2 = int(p[5]), int(p[6])
        except (ValueError, TypeError):
            continue  # jogo futuro ou linha inválida
        if quando > hoje:
            continue
        # p[3]/p[4] são os códigos dos dois times; a perspectiva (gols pró
        # ou contra) é resolvida em calcular_forma() comparando com o código.
        jogos.append((quando, p[3], p[4], g1, g2))
    return jogos


def calcular_forma(jogos, codigo, n_forma=5, n_recente=10):
    """Calcula forma a partir da lista de jogos de baixar_forma_recente."""
    if not jogos:
        return None
    ultimos = sorted(jogos, key=lambda j: j[0])[-max(n_forma, n_recente):]
    resultados = []  # (resultado, gols_pro) do mais antigo ao mais recente
    for _, c1, c2, g1, g2 in ultimos:
        if c1 == codigo:
            gp, gc = g1, g2
        elif c2 == codigo:
            gp, gc = g2, g1
        else:
            continue
        resultado = 'W' if gp > gc else ('L' if gp < gc else 'D')
        resultados.append((resultado, gp))
    if not resultados:
        return None
    recentes = resultados[-n_recente:]
    forma5 = resultados[-n_forma:][::-1]  # mais recente primeiro
    return {
        'Forma_Recente': '-'.join(r for r, _ in forma5),
        'Vitorias_Recentes': sum(1 for r, _ in recentes if r == 'W'),
        'Derrotas_Recentes': sum(1 for r, _ in recentes if r == 'L'),
        'Empates_Recentes': sum(1 for r, _ in recentes if r == 'D'),
        'Media_Gols_Recente': round(sum(g for _, g in recentes) / len(recentes), 2),
    }


def extrair(incluir_forma=False, apenas_codigos=None, pausa=0.5):
    """
    Extrai o ranking Elo completo. Se incluir_forma=True, busca também a
    forma recente (1 request por seleção — use apenas_codigos para limitar).
    """
    df = baixar_ranking()
    nomes = baixar_nomes()
    df['Team'] = df['Codigo'].map(nomes)

    # Colunas derivadas (mesmas fórmulas do dataset enriquecido)
    df['Saldo_Gols'] = df['Gols_Pro'] - df['Gols_Contra']
    df['Aproveitamento'] = (
        (df['Vitorias'] * 3 + df['Empates']) / (df['Total_Jogos'] * 3) * 100
    ).round(2)
    df['Media_Gols_Pro'] = (df['Gols_Pro'] / df['Total_Jogos']).round(2)
    df['Media_Gols_Contra'] = (df['Gols_Contra'] / df['Total_Jogos']).round(2)

    if incluir_forma:
        formas = {}
        alvo = df if apenas_codigos is None else df[df['Codigo'].isin(apenas_codigos)]
        total = len(alvo)
        for i, (_, linha) in enumerate(alvo.iterrows(), 1):
            codigo, nome = linha['Codigo'], linha['Team']
            if not isinstance(nome, str):
                continue
            try:
                jogos = baixar_forma_recente(nome)
                forma = calcular_forma(jogos, codigo) if jogos else None
                if forma:
                    formas[codigo] = forma
                print(f'  [{i}/{total}] forma {nome}: '
                      f"{forma['Forma_Recente'] if forma else 'sem dados'}")
            except Exception as e:
                print(f'  [{i}/{total}] forma {nome}: ERRO {e}')
            time.sleep(pausa)
        for chave in ['Forma_Recente', 'Vitorias_Recentes', 'Derrotas_Recentes',
                      'Empates_Recentes', 'Media_Gols_Recente']:
            df[chave] = df['Codigo'].map(
                lambda c: formas.get(c, {}).get(chave) if c in formas else None)

    df['Data_Extracao'] = datetime.now().strftime('%Y-%m-%d %H:%M')
    # 'Team' e 'Rating' primeiro: compatível com utils/data_loader.carregar_dados_elo
    ordem = ['Rank', 'Team', 'Codigo', 'Rating'] + [
        c for c in df.columns if c not in ('Rank', 'Team', 'Codigo', 'Rating')]
    return df[ordem]


def main():
    parser = argparse.ArgumentParser(description='Extrai o ranking Elo de eloratings.net')
    parser.add_argument('--forma', action='store_true',
                        help='busca também a forma recente de cada seleção (lento)')
    parser.add_argument('--saida', default=None, help='caminho do xlsx de saída')
    args = parser.parse_args()

    print('Baixando ranking Elo (World.tsv)...')
    df = extrair(incluir_forma=args.forma)
    print(f'{len(df)} seleções extraídas.')
    print(df[['Rank', 'Team', 'Rating']].head(10).to_string(index=False))

    saida = Path(args.saida) if args.saida else (
        PASTA_DATASET / f'RankingElo_{date.today().isoformat()}.xlsx')
    saida.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(saida, index=False)
    print(f'Salvo em: {saida}')


if __name__ == '__main__':
    main()
