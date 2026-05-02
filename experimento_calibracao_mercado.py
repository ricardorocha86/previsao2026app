import pandas as pd
from pathlib import Path
from utils.config import CAMINHO_ODDS

ODDS_PATH = CAMINHO_ODDS

def canonical_team_key(team_name):
    """Padroniza o nome da seleção para busca em diferentes datasets."""
    if not team_name or pd.isna(team_name):
        return ""
    # Remove espaços extras e converte para minúsculo
    key = str(team_name).strip().lower()
    # Adicione aqui mapeamentos específicos se notar divergências entre planilhas
    replacements = {
        'united states': 'usa',
        'south korea': 'korea republic',
        'czechia': 'czech republic',
        'ivory coast': "côte d'ivoire"
    }
    return replacements.get(key, key)

def load_market_target(path=None):
    """Carrega as probabilidades de mercado do arquivo XLSX do Oddschecker."""
    if path is None:
        path = ODDS_PATH
        
    # Carrega o XLSX
    df = pd.read_excel(path)
    
    # Identifica colunas
    # No novo XLSX: 'Selecao' e 'prob_implicita_media_normalizada'
    col_team = 'Selecao' if 'Selecao' in df.columns else df.columns[0]
    col_prob = 'prob_implicita_media_normalizada' if 'prob_implicita_media_normalizada' in df.columns else 'prob_implicita_media'
    
    # Prepara o DataFrame de saída
    odds_df = pd.DataFrame()
    odds_df['team_key'] = df[col_team].map(canonical_team_key)
    odds_df['market_prob'] = pd.to_numeric(df[col_prob], errors='coerce').fillna(0)
    
    # Garante que a soma seja 1.0 (re-normalização de segurança)
    total_prob = odds_df['market_prob'].sum()
    if total_prob > 0:
        odds_df['market_prob'] = odds_df['market_prob'] / total_prob
        
    return odds_df
