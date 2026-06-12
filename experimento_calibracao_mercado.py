import unicodedata
from pathlib import Path

import pandas as pd

from utils.config import CAMINHO_ODDS


def _achar_arquivo_mercado():
    """
    Probabilidade de referencia usada pelo app e pelos calibradores.

    Prioridade:
    1. Tabela mestra mais recente (Kalshi + Polymarket + Oddschecker),
       usando a coluna Media_3_fontes.
    2. mercados_predicao_*.xlsx mais recente (Kalshi + Polymarket).
    3. Oddschecker antigo.
    """
    base_dir = Path(__file__).resolve().parent
    analises_dir = base_dir / "analises" / "comparativo_mercados_probabilidades" / "resultados"
    tabelas_mestra = sorted(
        analises_dir.glob("**/TABELA_MESTRA_*.xlsx"),
        key=lambda p: p.stat().st_mtime,
    )
    if tabelas_mestra:
        return str(tabelas_mestra[-1])

    pasta = base_dir / "dataset"
    candidatos = sorted(
        pasta.glob("mercados_predicao_*.xlsx"),
        key=lambda p: p.stat().st_mtime,
    )
    return str(candidatos[-1]) if candidatos else CAMINHO_ODDS


ODDS_PATH = _achar_arquivo_mercado()


def _normalizar_texto(texto):
    texto = str(texto).strip().lower()
    texto = "".join(
        c for c in unicodedata.normalize("NFKD", texto)
        if not unicodedata.combining(c)
    )
    texto = texto.replace("`", "'").replace("’", "'").replace("â€™", "'")
    return " ".join(texto.split())


def canonical_team_key(team_name):
    """Padroniza o nome da selecao para busca em diferentes datasets."""
    if not team_name or pd.isna(team_name):
        return ""

    key = _normalizar_texto(team_name)
    replacements = {
        "united states": "usa",
        "south korea": "korea republic",
        "czechia": "czech republic",
        "ivory coast": "cote d'ivoire",
        "cote d ivoire": "cote d'ivoire",
        "cote d'ivoire": "cote d'ivoire",
        "estados unidos": "usa",
        "africa do sul": "south africa",
        "coreia do sul": "korea republic",
        "catar": "qatar",
        "suica": "switzerland",
        "brasil": "brazil",
        "escocia": "scotland",
        "marrocos": "morocco",
        "paraguai": "paraguay",
        "alemanha": "germany",
        "costa do marfim": "cote d'ivoire",
        "curacau": "curacao",
        "equador": "ecuador",
        "holanda": "netherlands",
        "japao": "japan",
        "belgica": "belgium",
        "egito": "egypt",
        "ira": "iran",
        "nova zelandia": "new zealand",
        "arabia saudita": "saudi arabia",
        "cabo verde": "cape verde",
        "espanha": "spain",
        "uruguai": "uruguay",
        "franca": "france",
        "noruega": "norway",
        "argelia": "algeria",
        "austria": "austria",
        "jordania": "jordan",
        "colombia": "colombia",
        "uzbequistao": "uzbekistan",
        "croacia": "croatia",
        "gana": "ghana",
        "inglaterra": "england",
        "iraque": "iraq",
        "rd do congo": "dr congo",
        "rd congo": "dr congo",
        "republica tcheca": "czech republic",
        "tcheca": "czech republic",
        "bosnia e herzegovina": "bosnia and herzegovina",
        "turquia": "turkey",
        "suecia": "sweden",
    }
    return replacements.get(key, key)


def load_market_target(path=None):
    """Carrega as probabilidades de referencia do arquivo XLSX configurado."""
    if path is None:
        path = ODDS_PATH

    df = pd.read_excel(path)

    col_team = "Selecao" if "Selecao" in df.columns else df.columns[0]
    if "Media_3_fontes" in df.columns:
        col_prob = "Media_3_fontes"
    elif "prob_implicita_media_normalizada" in df.columns:
        col_prob = "prob_implicita_media_normalizada"
    elif "prob_implicita_media" in df.columns:
        col_prob = "prob_implicita_media"
    else:
        col_prob = df.columns[-1]

    odds_df = pd.DataFrame()
    odds_df["team_key"] = df[col_team].map(canonical_team_key)
    odds_df["market_prob"] = pd.to_numeric(df[col_prob], errors="coerce").fillna(0)

    total_prob = odds_df["market_prob"].sum()
    if total_prob > 0:
        odds_df["market_prob"] = odds_df["market_prob"] / total_prob

    return odds_df
