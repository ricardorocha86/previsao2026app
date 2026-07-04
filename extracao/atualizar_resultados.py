# -*- coding: utf-8 -*-
"""
Atualiza os placares dos jogos ENCERRADOS da Copa 2026 a partir da API pública
de placar da ESPN (site.api.espn.com — JSON estruturado, SEM API key),
gravando o mesmo `resultados_jogos.json` nos dois lugares onde ele vive:

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

Uso (a partir da pasta Simulacao-Aplicativo-Streamlit) — não precisa de key:

  & "C:\\Users\\Pichau\\anaconda3\\python.exe" extracao\\atualizar_resultados.py

  # Só simular (não grava nada):
  ... extracao\\atualizar_resultados.py --dry-run

  # Testar a lógica offline com um JSON salvo da API (sem rede):
  ... extracao\\atualizar_resultados.py --from-file extracao\\exemplo_api.json --dry-run

  # Gravar e já commitar+empurrar nos dois repos (dispara deploy Vercel+Streamlit):
  ... extracao\\atualizar_resultados.py --git

  # Limitar a janela de datas consultadas (padrão: 11/06/2026 -> hoje):
  ... extracao\\atualizar_resultados.py --desde 2026-06-15 --ate 2026-06-16
"""
import argparse
import json
import subprocess
import time
import unicodedata
from datetime import date, datetime, timedelta
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

# --- API pública da ESPN (sem key) ---------------------------------------
ESPN_URL = 'https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world/scoreboard'
ESPN_HEADERS = {'User-Agent': 'Mozilla/5.0'}
INICIO_COPA = date(2026, 6, 11)   # primeiro dia de jogos

MESES_PT = {
    'janeiro': 1,
    'fevereiro': 2,
    'março': 3,
    'marco': 3,
    'abril': 4,
    'maio': 5,
    'junho': 6,
    'julho': 7,
    'agosto': 8,
    'setembro': 9,
    'outubro': 10,
    'novembro': 11,
    'dezembro': 12,
}

WEEKDAYS_PT = (
    'Segunda-feira',
    'Terça-feira',
    'Quarta-feira',
    'Quinta-feira',
    'Sexta-feira',
    'Sábado',
    'Domingo',
)

MONTHS_PT = (
    'janeiro',
    'fevereiro',
    'março',
    'abril',
    'maio',
    'junho',
    'julho',
    'agosto',
    'setembro',
    'outubro',
    'novembro',
    'dezembro',
)

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
def _parse_data_pt(valor):
    """Extrai AAAA-MM-DD de strings como 'Domingo, 28 de junho de 2026'."""
    texto = str(valor or '').lower()
    partes = texto.replace(',', ' ').split()
    try:
        dia = int(next(p for p in partes if p.isdigit() and len(p) <= 2))
        ano = int(next(p for p in partes if p.isdigit() and len(p) == 4))
        mes_nome = partes[partes.index('de') + 1] if 'de' in partes else ''
        mes = MESES_PT.get(mes_nome)
        if not mes:
            return None
        return date(ano, mes, dia)
    except (StopIteration, ValueError, IndexError):
        return None


def _format_data_pt(data):
    return f"{WEEKDAYS_PT[data.weekday()]}, {data.day} de {MONTHS_PT[data.month - 1]} de {data.year}"


def _normalizar_lista_previsoes(payload):
    return payload if isinstance(payload, list) else next(iter(payload.values()))


def carregar_indice_previsoes(caminhos):
    """
    Lê um ou mais arquivos de agenda/previsão e devolve um dict:
        frozenset({pt_a, pt_b}) -> [{'Seleção A', 'Seleção B', 'Data', 'ordem'}]
    'ordem' é o índice na lista (para manter a ordem cronológica ao adicionar).
    Só inclui pares cujas DUAS seleções são reconhecidas (ignora placeholders
    de mata-mata tipo "Vencedor Grupo A").
    """
    if isinstance(caminhos, (str, Path)):
        caminhos = [caminhos]
    indice = {}
    ordem_global = 0
    for caminho in caminhos:
        dados = json.loads(Path(caminho).read_text(encoding='utf-8'))
        lista = _normalizar_lista_previsoes(dados)
        for jogo in lista:
            a, b = jogo.get('Seleção A'), jogo.get('Seleção B')
            pt_a, pt_b = para_pt(a), para_pt(b)
            if not pt_a or not pt_b:
                ordem_global += 1
                continue
            chave = frozenset((pt_a, pt_b))
            indice.setdefault(chave, []).append({
                'Seleção A': a,
                'Seleção B': b,
                'Data': jogo.get('Data'),
                'data_iso': _parse_data_pt(jogo.get('Data')),
                'ordem': ordem_global,
            })
            ordem_global += 1
    return indice


# --- Busca dos jogos encerrados na ESPN (sem key) -------------------------
def buscar_jogos_encerrados_api(desde=None, ate=None):
    """
    Consulta o scoreboard da ESPN dia a dia entre `desde` e `ate` (date) e
    devolve os jogos encerrados. Sem API key. Faz dedup por id do evento (a
    ESPN datas em UTC, então um jogo pode aparecer em dia adjacente).
    """
    desde = desde or INICIO_COPA
    ate = ate or date.today()
    eventos = {}
    dia = desde
    while dia <= ate:
        r = requests.get(ESPN_URL, headers=ESPN_HEADERS,
                         params={'dates': dia.strftime('%Y%m%d')}, timeout=30)
        r.raise_for_status()
        for ev in r.json().get('events', []):
            eventos[ev.get('id')] = ev
        dia += timedelta(days=1)
        time.sleep(0.2)  # educado com a API pública
    return _eventos_para_jogos(eventos.values())


def buscar_jogos_encerrados_arquivo(caminho):
    """Lê um JSON salvo da ESPN ({'events': [...]}) — teste offline, sem rede."""
    payload = json.loads(Path(caminho).read_text(encoding='utf-8'))
    return _eventos_para_jogos(payload.get('events', []))


def _eventos_para_jogos(eventos):
    """
    Converte eventos do scoreboard da ESPN nos encerrados, no formato:
        {'home', 'away', 'home_score', 'away_score'}  (nomes em inglês)
    Só entram os com status.type.completed == True.
    """
    encerrados = []
    for ev in eventos:
        comp = (ev.get('competitions') or [{}])[0]
        if not (comp.get('status', {}).get('type', {}) or {}).get('completed'):
            continue
        lados = {c.get('homeAway'): c for c in comp.get('competitors', [])}
        casa, fora = lados.get('home'), lados.get('away')
        if not casa or not fora or casa.get('score') is None or fora.get('score') is None:
            continue
        encerrados.append({
            'date': str(ev.get('date') or '')[:10],
            'home': (casa.get('team') or {}).get('displayName'),
            'away': (fora.get('team') or {}).get('displayName'),
            'home_score': int(casa['score']),
            'away_score': int(fora['score']),
            'winner': (
                (casa.get('team') or {}).get('displayName') if casa.get('winner')
                else (fora.get('team') or {}).get('displayName') if fora.get('winner')
                else None
            ),
        })
    return encerrados


# --- Casamento API x previsão ---------------------------------------------
def montar_resultados(jogos_api, indice_previsoes, existentes=None):
    """
    Casa cada jogo encerrado da API a uma previsão e devolve (entradas, avisos).
    Cada entrada usa Data/nomes EXATOS da previsão e o placar da API.
    """
    entradas = []
    avisos = []
    indice_existentes = {}
    for ordem, existente in enumerate(existentes or []):
        a = existente.get('Seleção A')
        b = existente.get('Seleção B')
        pt_a, pt_b = para_pt(a), para_pt(b)
        if not pt_a or not pt_b:
            continue
        indice_existentes.setdefault(frozenset((pt_a, pt_b)), []).append({
            'Seleção A': a,
            'Seleção B': b,
            'Data': existente.get('Data'),
            'data_iso': _parse_data_pt(existente.get('Data')),
            'ordem': ordem,
        })
    for j in jogos_api:
        pt_home, pt_away = para_pt(j['home']), para_pt(j['away'])
        if not pt_home or not pt_away:
            faltando = j['home'] if not pt_home else j['away']
            avisos.append(f"Seleção não reconhecida no MAPA_SELECOES: {faltando!r} "
                          f"(jogo {j['home']} x {j['away']})")
            continue
        candidatos = indice_previsoes.get(frozenset((pt_home, pt_away)))
        data_api = datetime.strptime(j['date'], '%Y-%m-%d').date() if j.get('date') else None
        if not candidatos:
            candidatos_existentes = indice_existentes.get(frozenset((pt_home, pt_away)))
            if candidatos_existentes:
                prev = sorted(candidatos_existentes, key=lambda x: x['ordem'])[0]
            elif data_api and data_api >= date(2026, 6, 28):
                prev = {
                    'Seleção A': pt_home,
                    'Seleção B': pt_away,
                    'Data': _format_data_pt(data_api),
                    'ordem': 10_000 + len(entradas),
                }
            else:
                avisos.append(f"Sem previsão correspondente para {pt_home} x {pt_away} "
                              f"(ignorado).")
                continue
        elif data_api:
            candidatos_com_distancia = []
            for candidato in candidatos:
                data_prev = candidato.get('data_iso')
                distancia = abs((data_prev - data_api).days) if data_prev else 999
                candidatos_com_distancia.append((distancia, candidato['ordem'], candidato))
            prev = sorted(candidatos_com_distancia, key=lambda x: (x[0], x[1]))[0][2]
        else:
            prev = sorted(candidatos, key=lambda x: x['ordem'])[0]

        # Placar do mandante (home) vai para a seleção certa conforme A/B da previsão.
        if prev['Seleção A'] == pt_home:
            placar_a, placar_b = j['home_score'], j['away_score']
        else:
            placar_a, placar_b = j['away_score'], j['home_score']
        entrada = {
            'Seleção A': prev['Seleção A'],
            'Seleção B': prev['Seleção B'],
            'Data': prev['Data'],
            'Placar A': placar_a,
            'Placar B': placar_b,
            'Status': 'encerrado',
            '_ordem': prev['ordem'],
        }
        vencedor = para_pt(j.get('winner'))
        if vencedor:
            entrada['Vencedor'] = vencedor
        entradas.append(entrada)
    return entradas, avisos


def mesclar(existentes, novas):
    """
    Funde as entradas novas com as já gravadas. Mantém a ordem das existentes
    (diff mínimo) e adiciona as inéditas na ordem do calendário (campo _ordem).
    Devolve (lista_final, n_novas, n_atualizadas).

    O casamento é pelo PAR de seleções (SEM ordem) + Data — nunca pela tupla
    (Seleção A, Seleção B, Data). Isso é proposital: se a ordem A/B de um jogo
    mudar na previsão entre execuções (ex.: "Croácia x Panamá" -> "Panamá x
    Croácia"), a linha existente é ATUALIZADA e reordenada para a ordem nova, em
    vez de uma 2ª linha do mesmo jogo ser inserida (o que inflava a contagem de
    jogos encerrados, ex.: 49/104 em vez de 48/104).
    """
    def chave(e):
        return (frozenset((e['Seleção A'], e['Seleção B'])), e['Data'])

    indexado = {chave(e): e for e in existentes}
    n_novas = n_atualizadas = 0
    ineditas = []
    campos = ('Seleção A', 'Seleção B', 'Data', 'Placar A', 'Placar B', 'Status', 'Vencedor')
    for nv in novas:
        k = chave(nv)
        limpo = {x: nv[x] for x in campos if x in nv}
        if k in indexado:
            atual = indexado[k]
            # Atualiza (e reordena A/B) se QUALQUER campo divergir, inclusive a
            # ordem das seleções — assim a linha passa a refletir a previsão.
            atual_limpo = {x: atual.get(x) for x in campos if x in atual or x in limpo}
            if atual_limpo != limpo:
                atual.update(limpo)
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
    parser.add_argument('--from-file', default=None,
                        help='Lê um JSON salvo da ESPN em vez de chamar a rede (teste offline).')
    parser.add_argument('--desde', default=None,
                        help='Data inicial AAAA-MM-DD a consultar (padrão: 2026-06-11).')
    parser.add_argument('--ate', default=None,
                        help='Data final AAAA-MM-DD a consultar (padrão: hoje).')
    parser.add_argument('--previsoes', nargs='+', default=[str(PREVISOES_PADRAO)],
                        help='Caminho(s) de agenda/previsões usados como fonte de verdade do casamento.')
    parser.add_argument('--saidas', nargs='*', default=[str(p) for p in SAIDAS_PADRAO],
                        help='Arquivos resultados_jogos.json a atualizar (dois por padrão).')
    parser.add_argument('--dry-run', action='store_true',
                        help='Não grava nada; só mostra o que mudaria.')
    parser.add_argument('--git', action='store_true',
                        help='Após gravar, faz commit + push nos repos (dispara deploys).')
    args = parser.parse_args()

    # 1) Jogos encerrados (ESPN ou arquivo)
    if args.from_file:
        jogos_api = buscar_jogos_encerrados_arquivo(args.from_file)
        print(f'{len(jogos_api)} jogos encerrados lidos de {args.from_file}.')
    else:
        desde = datetime.strptime(args.desde, '%Y-%m-%d').date() if args.desde else None
        ate = datetime.strptime(args.ate, '%Y-%m-%d').date() if args.ate else None
        jogos_api = buscar_jogos_encerrados_api(desde=desde, ate=ate)
        print(f'{len(jogos_api)} jogos encerrados retornados pela ESPN.')

    # 2) Casa com as previsões (fonte de verdade de data + nomes)
    indice = carregar_indice_previsoes(args.previsoes)
    base = carregar_resultados(args.saidas[0])
    novas, avisos = montar_resultados(jogos_api, indice, existentes=base)
    for a in avisos:
        print(f'  AVISO: {a}')
    print(f'{len(novas)} jogos casados com o calendário.')

    # 3) Mescla com o que já está gravado (usa o 1o arquivo como base)
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
