# -*- coding: utf-8 -*-
"""
Extrai o valor de mercado dos elencos das seleções do Transfermarkt.

Fonte: https://www.transfermarkt.com/statistik/weltrangliste
(ranking FIFA com valor total do elenco, tamanho do elenco e média de
idade por seleção; ~9 páginas de 25). Os valores refletem o elenco atual
cadastrado no Transfermarkt — durante a Copa, as convocações finais.

Uso (a partir da pasta Simulacao-Aplicativo-Streamlit):
  & "C:\\Users\\Pichau\\anaconda3\\python.exe" extracao\\extrair_valor_mercado.py
  ... --atualizar    # também preenche Valor_Mercado_Milhoes_EUR, Media_Idade e
                     # Tamanho_Elenco no FIFA_ELO_DadosSeleções_*.xlsx mais
                     # recente (com backup .bak)

Saída: dataset/valor_mercado_selecoes_AAAA-MM-DD.xlsx
"""
import argparse
import html as html_mod
import re
import shutil
import sys
import time
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from util_rede import UA
from extrair_mercados_predicao import chave_selecao, SINONIMOS

PASTA_DATASET = Path(__file__).resolve().parents[1] / 'dataset'
URL = 'https://www.transfermarkt.com/statistik/weltrangliste'

# Grafias específicas do Transfermarkt -> chave padrão do projeto
SINONIMOS.update({
    'korea, south': 'korea republic',
    "cote d'ivoire": "côte d'ivoire",
})


def _valor_para_milhoes(texto):
    """Converte '€1.52bn' / '€782.50m' / '€950k' em milhões de EUR (float)."""
    m = re.match(r'€?([\d.,]+)\s*(bn|m|k)?', str(texto).strip(), re.I)
    if not m:
        return None
    numero = float(m.group(1).replace(',', ''))
    unidade = (m.group(2) or 'm').lower()
    fator = {'bn': 1000.0, 'm': 1.0, 'k': 0.001}[unidade]
    return round(numero * fator, 2)


def _limpar(celula):
    texto = re.sub(r'<[^>]+>', ' ', celula)
    return html_mod.unescape(texto).replace('\xa0', ' ').strip()


def extrair(pausa=1.0, max_paginas=12):
    """Baixa todas as páginas do ranking e devolve um DataFrame."""
    registros = []
    chaves_vistas = set()
    for pagina in range(1, max_paginas + 1):
        params = {'page': pagina} if pagina > 1 else {}
        r = requests.get(URL, headers=UA, params=params, timeout=30)
        r.raise_for_status()
        m = re.search(r'<table class="items">(.*?)</table>', r.text, re.S)
        if not m:
            break
        linhas = re.findall(r'<tr[^>]*>(.*?)</tr>', m.group(1), re.S)
        novos = 0
        for linha in linhas:
            celulas = re.findall(r'<td[^>]*>(.*?)</td>', linha, re.S)
            if len(celulas) < 7:
                continue
            nome = _limpar(celulas[1])
            valor = _valor_para_milhoes(_limpar(celulas[4]))
            if not nome or valor is None:
                continue
            chave = chave_selecao(nome)
            if chave in chaves_vistas:
                continue  # página além da última repete o conteúdo final
            chaves_vistas.add(chave)
            try:
                elenco = int(_limpar(celulas[2]))
            except ValueError:
                elenco = None
            try:
                idade = float(_limpar(celulas[3]))
            except ValueError:
                idade = None
            registros.append({
                'Selecao': nome,
                'team_key': chave,
                'Tamanho_Elenco': elenco,
                'Media_Idade': idade,
                'Valor_Mercado_Milhoes_EUR': valor,
            })
            novos += 1
        print(f'  página {pagina}: {novos} seleções')
        if novos == 0:
            break
        time.sleep(pausa)
    df = pd.DataFrame(registros).drop_duplicates(subset='team_key')
    df['Data_Extracao'] = datetime.now().strftime('%Y-%m-%d %H:%M')
    return df


def atualizar_dataset(df_valores):
    """Preenche as 3 colunas no FIFA_ELO_DadosSeleções_*.xlsx mais recente."""
    candidatos = sorted(PASTA_DATASET.glob('FIFA_ELO_DadosSeleções_*.xlsx'))
    if not candidatos:
        raise SystemExit(f'Nenhum FIFA_ELO_DadosSeleções_*.xlsx em {PASTA_DATASET}')
    alvo = candidatos[-1]
    df = pd.read_excel(alvo)
    valores = df_valores.set_index('team_key').to_dict('index')

    atualizadas, sem_dado = 0, []
    for idx, linha in df.iterrows():
        dado = valores.get(chave_selecao(linha.get('NomeIngles')))
        if not dado:
            sem_dado.append(linha.get('Seleção'))
            continue
        for col in ['Valor_Mercado_Milhoes_EUR', 'Media_Idade', 'Tamanho_Elenco']:
            if dado.get(col) is not None:
                df.loc[idx, col] = dado[col]
        atualizadas += 1

    shutil.copy2(alvo, alvo.with_suffix('.xlsx.bak'))
    df.to_excel(alvo, index=False)
    print(f'\n{alvo.name}: valor de mercado atualizado em {atualizadas}/{len(df)} '
          f'seleções (backup .bak criado).')
    if sem_dado:
        print(f'  AVISO — sem valor no Transfermarkt: {sem_dado}')


def main():
    parser = argparse.ArgumentParser(
        description='Extrai valor de mercado dos elencos (Transfermarkt)')
    parser.add_argument('--atualizar', action='store_true',
                        help='preenche as colunas no dataset enriquecido mais recente')
    parser.add_argument('--saida', default=None, help='caminho do xlsx de saída')
    args = parser.parse_args()

    print('Baixando ranking de seleções do Transfermarkt...')
    df = extrair()
    print(f'{len(df)} seleções extraídas.')
    print(df[['Selecao', 'Valor_Mercado_Milhoes_EUR']].head(10).to_string(index=False))

    saida = Path(args.saida) if args.saida else (
        PASTA_DATASET / f'valor_mercado_selecoes_{date.today().isoformat()}.xlsx')
    df.to_excel(saida, index=False)
    print(f'Salvo em: {saida}')

    if args.atualizar:
        atualizar_dataset(df)


if __name__ == '__main__':
    main()
