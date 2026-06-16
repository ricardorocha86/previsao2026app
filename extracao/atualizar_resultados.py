# -*- coding: utf-8 -*-
"""
Atualiza os placares dos jogos ENCERRADOS da Copa 2026 a partir de uma API
esportiva estruturada (football-data.org), gravando o mesmo
`resultados_jogos.json` nos dois lugares onde ele vive:

  - Website/assets/resultados_jogos.json            (site React/Vite)
  - Simulacao-Aplicativo-Streamlit/assets/resultados_jogos.json  (app Streamlit)

Ideia central (por que isto é confiável):
  A API só fornece o PLACAR. Quem decide a data e os nomes exatos das seleções
  é o `previsoes_jogos.json` — a fonte de verdade já publicada. Cada jogo
  encerrado da API é casado a uma previsão pelo PAR de seleções (sem ordem) e
  então copiamos a `Data` e os nomes ("Seleção A"/"Seleção B") EXATOS da
  previsão, só preenchendo o placar. Isso elimina de uma vez:
    - erro de fuso (ex.: Austrália x Turquia que a ESPN mostra em 14/06 mas o
      nosso calendário registra em 13/06);
    - erro de string exata (o site/bolão casa por "Seleção A|Seleção B|Data");
    - transcrição manual de placar (que foi o que gerou o Irã 0x1 errado).

Uso (a partir da pasta Simulacao-Aplicativo-Streamlit):

  # 1) Gere uma API key grátis em https://www.football-data.org/client/register
  $env:FOOTBALL_DATA_API_KEY = "sua_key"
  & "C:\\Users\\Pichau\\anaconda3\\python.exe" extracao\\atualizar_resultados.py

  # Só simular (não grava nada):
  ... extracao\\atualizar_resultados.py --dry-run

  # Testar a lógica offline com um JSON salvo da API (sem key):
  ... extracao\\atualizar_resultados.py --from-file extracao\\exemplo_api.json --dry-run

  # Gravar e já commitar+empurrar nos dois repos (dispara deploy Vercel+Streamlit):
  ... extracao\\atualizar_resultados.py --git
"""
import argparse
import json
import os
import subprocess
import sys
import unicodedata
from pathlib import Path

import requests

# --- Caminhos (relativos a este arquivo) ---------------------------------
RAIZ_STREAMLIT = Path(__file__).resolve().parents[1]          # Simulacao-Aplicativo-Streamlit
RAIZ_PROJETO = RAIZ_STREAMLIT.parent                          # PrevisaoEsportiva2026
PREVISOES_PADRAO = RAIZ_STREAMLIT / 'assets' / 'previsoes_jogos.json'
SAIDAS_PADRAO = [
    RAIZ_PROJETO / 'Website' / 'assets' / 'resultados_jogos.json',
    RAIZ_STREAMLIT / 'assets' / 'resultados_jogos.json',
]

# --- API football-data.org -----------------------------------------------
API_URL = 'https://api.football-data.org/v4/competitions/WC/matches'

# --- Mapa de nomes: variantes em inglês/alias -> nome PT canônico ---------
# As chaves são normalizadas (sem acento, minúsculas, só alfanumérico) na hora
# da busca, então "Côte d'Ivoire", "Cote d Ivoire" e "Ivory Coast" colidem.
MAPA_SELECOES = {
    'Alemanha': ['Germany'],
    'Argentina': ['Argentina'],
    'Argélia': ['Algeria'],
    'Arábia Saudita': ['Saudi Arabia'],
    'Austrália': ['Australia'],
    'Brasil': ['Brazil'],
    'Bélgica': ['Belgium'],
    'Bósnia e Herzegovina': ['Bosnia and Herzegovina', 'Bosnia-Herzegovina', 'Bosnia & Herzegovina'],
    'Cabo Verde': ['Cape Verde', 'Cabo Verde'],
    'Canadá': ['Canada'],
    'Catar': ['Qatar'],
    'Colômbia': ['Colombia'],
    'Coreia do Sul': ['Korea Republic', 'South Korea', 'Republic of Korea'],
    'Costa do Marfim': ['Ivory Coast', "Côte d'Ivoire", 'Cote d Ivoire'],
    'Croácia': ['Croatia'],
    'Curaçau': ['Curacao', 'Curaçao'],
    'Egito': ['Egypt'],
    'Equador': ['Ecuador'],
    'Escócia': ['Scotland'],
    'Espanha': ['Spain'],
    'Estados Unidos': ['United States', 'USA', 'United States of America'],
    'França': ['France'],
    'Gana': ['Ghana'],
    'Haiti': ['Haiti'],
    'Holanda': ['Netherlands', 'Holland'],
    'Inglaterra': ['England'],
    'Iraque': ['Iraq'],
    'Irã': ['Iran', 'IR Iran', 'Iran, Islamic Republic of'],
    'Japão': ['Japan'],
    'Jordânia': ['Jordan'],
    'Marrocos': ['Morocco'],
    'México': ['Mexico'],
    'Noruega': ['Norway'],
    'Nova Zelândia': ['New Zealand'],
    'Panamá': ['Panama'],
    'Paraguai': ['Paraguay'],
    'Portugal': ['Portugal'],
    'RD do Congo': ['DR Congo', 'Congo DR', 'Democratic Republic of the Congo',
                    'Congo Democratic Republic'],
    'Senegal': ['Senegal'],
    'Suécia': ['Sweden'],
    'Suíça': ['Switzerland'],
    'Tcheca': ['Czechia', 'Czech Republic'],
    'Tunísia': ['Tunisia'],
    'Turquia': ['Turkey', 'Türkiye', 'Turkiye'],
    'Uruguai': ['Uruguay'],
    'Uzbequistão': ['Uzbekistan'],
    'África do Sul': ['South Africa'],
    'Áustria': ['Austria'],
}


def _norm(nome):
    """Normaliza um nome de seleção: sem acento, minúsculo, só alfanumérico."""
    if nome is None:
        return ''
    sem_acento = ''.join(
        c for c in unicodedata.normalize('NFKD', str(nome))
        if not unicodedata.combining(c))
    return ''.join(c for c in sem_acento.lower() if c.isalnum())


# Índice normalizado -> nome PT canônico (inclui o próprio nome PT e os alias).
_INDICE_NOMES = {}
for _pt, _aliases in MAPA_SELECOES.items():
    _INDICE_NOMES[_norm(_pt)] = _pt
    for _a in _aliases:
        _INDICE_NOMES[_norm(_a)] = _pt


def para_pt(nome):
    """Devolve o nome PT canônico de uma seleção, ou None se desconhecido."""
    return _INDICE_NOMES.get(_norm(nome))


# --- Leitura das previsões (fonte de verdade do casamento) ----------------
def carregar_indice_previsoes(caminho):
    """
    Lê previsoes_jogos.json e devolve um dict:
        frozenset({pt_a, pt_b}) -> {'Seleção A', 'Seleção B', 'Data', 'ordem'}
    'ordem' é o índice na lista (para manter a ordem cronológica ao adicionar).
    Só inclui pares cujas DUAS seleções são reconhecidas (ignora placeholders
    de mata-mata tipo "Vencedor Grupo A").
    """
    dados = json.loads(Path(caminho).read_text(encoding='utf-8'))
    lista = dados if isinstance(dados, list) else next(iter(dados.values()))
    indice = {}
    for ordem, jogo in enumerate(lista):
        a, b = jogo.get('Seleção A'), jogo.get('Seleção B')
        pt_a, pt_b = para_pt(a), para_pt(b)
        if not pt_a or not pt_b:
            continue
        chave = frozenset((pt_a, pt_b))
        indice.setdefault(chave, {
            'Seleção A': a,
            'Seleção B': b,
            'Data': jogo.get('Data'),
            'ordem': ordem,
        })
    return indice


# --- Busca dos jogos encerrados na API ------------------------------------
def buscar_jogos_encerrados_api(api_key):
    """Consulta a football-data.org e devolve jogos com status FINISHED."""
    headers = {'X-Auth-Token': api_key}
    r = requests.get(API_URL, headers=headers, timeout=30)
    r.raise_for_status()
    return _normalizar_resposta_api(r.json())


def buscar_jogos_encerrados_arquivo(caminho):
    """Lê um JSON salvo da API (modo --from-file, para teste offline)."""
    return _normalizar_resposta_api(json.loads(Path(caminho).read_text(encoding='utf-8')))


def _normalizar_resposta_api(payload):
    """
    Extrai dos `matches` da football-data.org só os encerrados, no formato:
        {'home', 'away', 'home_score', 'away_score'}  (nomes em inglês)
    """
    encerrados = []
    for m in payload.get('matches', []):
        if m.get('status') != 'FINISHED':
            continue
        full = (m.get('score') or {}).get('fullTime') or {}
        if full.get('home') is None or full.get('away') is None:
            continue
        encerrados.append({
            'home': (m.get('homeTeam') or {}).get('name'),
            'away': (m.get('awayTeam') or {}).get('name'),
            'home_score': int(full['home']),
            'away_score': int(full['away']),
        })
    return encerrados


# --- Casamento API x previsão ---------------------------------------------
def montar_resultados(jogos_api, indice_previsoes):
    """
    Casa cada jogo encerrado da API a uma previsão e devolve (entradas, avisos).
    Cada entrada usa Data/nomes EXATOS da previsão e o placar da API.
    """
    entradas = []
    avisos = []
    for j in jogos_api:
        pt_home, pt_away = para_pt(j['home']), para_pt(j['away'])
        if not pt_home or not pt_away:
            faltando = j['home'] if not pt_home else j['away']
            avisos.append(f"Seleção não reconhecida no MAPA_SELECOES: {faltando!r} "
                          f"(jogo {j['home']} x {j['away']})")
            continue
        prev = indice_previsoes.get(frozenset((pt_home, pt_away)))
        if not prev:
            avisos.append(f"Sem previsão correspondente para {pt_home} x {pt_away} "
                          f"(ignorado).")
            continue
        # Placar do mandante (home) vai para a seleção certa conforme A/B da previsão.
        if prev['Seleção A'] == pt_home:
            placar_a, placar_b = j['home_score'], j['away_score']
        else:
            placar_a, placar_b = j['away_score'], j['home_score']
        entradas.append({
            'Seleção A': prev['Seleção A'],
            'Seleção B': prev['Seleção B'],
            'Data': prev['Data'],
            'Placar A': placar_a,
            'Placar B': placar_b,
            'Status': 'encerrado',
            '_ordem': prev['ordem'],
        })
    return entradas, avisos


def mesclar(existentes, novas):
    """
    Funde as entradas novas com as já gravadas. Mantém a ordem das existentes
    (diff mínimo) e adiciona as inéditas na ordem do calendário (campo _ordem).
    Devolve (lista_final, n_novas, n_atualizadas).
    """
    def chave(e):
        return (e['Seleção A'], e['Seleção B'], e['Data'])

    indexado = {chave(e): e for e in existentes}
    n_novas = n_atualizadas = 0
    ineditas = []
    for nv in novas:
        k = chave(nv)
        limpo = {x: nv[x] for x in
                 ('Seleção A', 'Seleção B', 'Data', 'Placar A', 'Placar B', 'Status')}
        if k in indexado:
            atual = indexado[k]
            if (atual.get('Placar A'), atual.get('Placar B'), atual.get('Status')) != \
               (limpo['Placar A'], limpo['Placar B'], limpo['Status']):
                indexado[k].update(limpo)
                n_atualizadas += 1
        else:
            ineditas.append((nv['_ordem'], limpo))
            n_novas += 1

    final = list(existentes)  # preserva ordem/append já existentes (atualizados in-place)
    for _, e in sorted(ineditas, key=lambda t: t[0]):
        final.append(e)
    return final, n_novas, n_atualizadas


def carregar_resultados(caminho):
    p = Path(caminho)
    if not p.exists():
        return []
    return json.loads(p.read_text(encoding='utf-8'))


def serializar(lista):
    """Mesmo formato do arquivo atual: indent=2, acentos preservados, \\n final."""
    return json.dumps(lista, ensure_ascii=False, indent=2) + '\n'


# --- git -------------------------------------------------------------------
def commit_e_push(caminho_arquivo, mensagem):
    """Commita o arquivo no repo a que ele pertence e dá push em origin/main."""
    repo = Path(caminho_arquivo).resolve().parents[1]  # .../assets/x.json -> repo
    rel = str(Path(caminho_arquivo).resolve().relative_to(repo))
    def git(*args):
        return subprocess.run(['git', '-C', str(repo), *args],
                              check=True, capture_output=True, text=True)
    git('add', rel)
    # Só commita se houver mudança de fato.
    if subprocess.run(['git', '-C', str(repo), 'diff', '--cached', '--quiet']).returncode == 0:
        print(f'  [{repo.name}] nada para commitar.')
        return
    git('commit', '-m', mensagem)
    git('push', 'origin', 'main')
    print(f'  [{repo.name}] commit + push OK.')


# --- main ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description='Atualiza resultados_jogos.json (jogos encerrados da Copa 2026).')
    parser.add_argument('--api-key', default=os.environ.get('FOOTBALL_DATA_API_KEY'),
                        help='API key da football-data.org (ou env FOOTBALL_DATA_API_KEY).')
    parser.add_argument('--from-file', default=None,
                        help='Lê um JSON salvo da API em vez de chamar a rede (teste offline).')
    parser.add_argument('--previsoes', default=str(PREVISOES_PADRAO),
                        help='Caminho do previsoes_jogos.json (fonte de verdade do casamento).')
    parser.add_argument('--saidas', nargs='*', default=[str(p) for p in SAIDAS_PADRAO],
                        help='Arquivos resultados_jogos.json a atualizar (dois por padrão).')
    parser.add_argument('--dry-run', action='store_true',
                        help='Não grava nada; só mostra o que mudaria.')
    parser.add_argument('--git', action='store_true',
                        help='Após gravar, faz commit + push nos repos (dispara deploys).')
    args = parser.parse_args()

    # 1) Jogos encerrados (API ou arquivo)
    if args.from_file:
        jogos_api = buscar_jogos_encerrados_arquivo(args.from_file)
        print(f'{len(jogos_api)} jogos encerrados lidos de {args.from_file}.')
    else:
        if not args.api_key:
            parser.error('Sem API key. Use --api-key, defina FOOTBALL_DATA_API_KEY, '
                         'ou rode com --from-file para teste offline.')
        jogos_api = buscar_jogos_encerrados_api(args.api_key)
        print(f'{len(jogos_api)} jogos encerrados retornados pela API.')

    # 2) Casa com as previsões (fonte de verdade de data + nomes)
    indice = carregar_indice_previsoes(args.previsoes)
    novas, avisos = montar_resultados(jogos_api, indice)
    for a in avisos:
        print(f'  AVISO: {a}')
    print(f'{len(novas)} jogos casados com o calendário.')

    # 3) Mescla com o que já está gravado (usa o 1o arquivo como base)
    base = carregar_resultados(args.saidas[0])
    final, n_novas, n_atualizadas = mesclar(base, novas)
    print(f'Resultado: {n_novas} novo(s), {n_atualizadas} atualizado(s), '
          f'{len(final)} no total.')

    if n_novas == 0 and n_atualizadas == 0:
        print('Nada mudou — arquivos já estão atualizados.')
        return

    conteudo = serializar(final)
    if args.dry_run:
        print('\n--- DRY-RUN: conteúdo que seria gravado (início) ---')
        print('\n'.join(conteudo.splitlines()[:12]) + '\n  ...')
        return

    # 4) Grava idêntico nos dois lugares
    for caminho in args.saidas:
        Path(caminho).write_text(conteudo, encoding='utf-8')
        print(f'Gravado: {caminho}')

    # 5) (Opcional) commit + push -> deploy automático Vercel/Streamlit
    if args.git:
        msg = f'Atualiza resultados Copa: {n_novas} novo(s), {n_atualizadas} corrigido(s)'
        for caminho in args.saidas:
            commit_e_push(caminho, msg)


if __name__ == '__main__':
    main()
