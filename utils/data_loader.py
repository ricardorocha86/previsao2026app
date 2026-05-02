# ============ FUNÇÕES DE CARREGAMENTO DE DADOS ============
import streamlit as st
import pandas as pd
import numpy as np
import os

from utils.config import (
    FORCA_PADRAO, PESO_RANKING_FIFA, PESO_PARTICIPACOES, 
    PESO_MELHOR_RESULTADO, MAPEAMENTO_NOMES_ELO,
    CAMINHO_DADOS, CAMINHO_ELO
)
from utils.simulador_oficial import parse_world_cup_score

# Elo padrão para seleções sem dados
ELO_PADRAO = 1500

def extrair_melhor_resultado(texto):
    """Retorna a pontuação numérica do melhor resultado (via simulador_oficial)"""
    return parse_world_cup_score(texto)

@st.cache_data
def carregar_dados(caminho=None):
    """Carrega dados de arquivo CSV ou Excel"""
    if caminho is None:
        caminho = CAMINHO_DADOS
    
    if caminho.endswith('.xlsx') or caminho.endswith('.xls'):
        df = pd.read_excel(caminho)
    else:
        df = pd.read_csv(caminho, sep=';', skiprows=1)
    
    # Processa Pontuação FIFA
    col_fifa = None
    for col in ['FIFA_Current_Points', 'Pontuacao_Ranking_FIFA', 'Pontuacao_FIFA', 'Ranking_FIFA_Nov_2025']:
        if col in df.columns:
            col_fifa = col
            break
    
    if col_fifa:
        df['Pontuacao_FIFA_Num'] = df[col_fifa].apply(
            lambda x: float(str(x).replace('.', '').replace(',', '.')) if pd.notna(x) and str(x) != 'N/A' else FORCA_PADRAO
        )
    else:
        df['Pontuacao_FIFA_Num'] = FORCA_PADRAO
    
    # Processa Participações em Copa
    col_part = None
    for col in ['Participações_Copa_Mundo', 'Participacoes_Copa_Mundo', 'Participacoes_Num']:
        if col in df.columns:
            col_part = col
            break
    
    if col_part:
        df['Participacoes_Num'] = df[col_part].apply(
            lambda x: int(float(x)) if pd.notna(x) and str(x) != 'N/A' else 0
        )
    else:
        df['Participacoes_Num'] = 0
    
    # Processa Melhor Resultado
    col_resultado = None
    for col in ['Melhor_Resultado_Copa_Mundo', 'Melhor_Resultado']:
        if col in df.columns:
            col_resultado = col
            break
    
    if col_resultado:
        df['Melhor_Resultado_Limpo'] = df[col_resultado].apply(extrair_melhor_resultado)
    else:
        df['Melhor_Resultado_Limpo'] = 'Estreante'
    
    if 'Grupo' in df.columns:
        df['Grupo'] = df['Grupo'].str.strip()
    
    return df

@st.cache_data
def carregar_dados_elo(caminho_xlsx=None):
    """Carrega os dados do ranking Elo do arquivo Excel"""
    if caminho_xlsx is None:
        caminho_xlsx = CAMINHO_ELO
    
    try:
        df_elo = pd.read_excel(caminho_xlsx)
        
        # Encontrar coluna de nome do time
        col_team = None
        for possivel_nome in ['Team', 'team', 'Selecao', 'Seleção', 'Country', 'Nation', 'Nome']:
            if possivel_nome in df_elo.columns:
                col_team = possivel_nome
                break
        
        if col_team is None and len(df_elo.columns) >= 2:
            for col in df_elo.columns:
                if df_elo[col].dtype == 'object':
                    sample = df_elo[col].dropna().head(5).tolist()
                    if any(isinstance(s, str) and len(s) > 2 for s in sample):
                        col_team = col
                        break
        
        if col_team and col_team != 'Team_Elo':
            df_elo = df_elo.rename(columns={col_team: 'Team_Elo'})
        
        if 'Team_Elo' not in df_elo.columns:
            st.warning("⚠️ Não foi possível identificar a coluna de times no arquivo Elo")
            return None
        
        return df_elo
    except Exception as e:
        st.warning(f"⚠️ Não foi possível carregar dados Elo: {e}")
        return None

def mapear_selecao_para_elo(selecao, df_elo):
    """Mapeia o nome da seleção em português para o nome no arquivo Elo"""
    if df_elo is None or 'Team_Elo' not in df_elo.columns:
        return None
    
    nomes_possiveis = MAPEAMENTO_NOMES_ELO.get(selecao, [selecao])
    
    for nome in nomes_possiveis:
        try:
            match = df_elo[df_elo['Team_Elo'].astype(str).str.lower() == nome.lower()]
            if not match.empty:
                return match.iloc[0]
            
            match = df_elo[df_elo['Team_Elo'].astype(str).str.lower().str.contains(nome.lower(), na=False)]
            if not match.empty:
                return match.iloc[0]
        except:
            continue
    
    return None

def obter_elo_selecao(selecao, df_elo):
    """Obtém o rating Elo de uma seleção"""
    row_elo = mapear_selecao_para_elo(selecao, df_elo)
    if row_elo is None:
        return ELO_PADRAO
    
    try:
        # Tentar várias colunas possíveis para o rating
        for col in ['Rating', 'Elo', 'ELO', 'rating', 'elo']:
            if col in row_elo.index:
                rating = row_elo[col]
                if pd.notna(rating):
                    return float(rating)
        return ELO_PADRAO
    except:
        return ELO_PADRAO

def preparar_estruturas(df, df_elo=None):
    """
    Prepara estruturas de dados a partir do DataFrame.
    
    Se df_elo for fornecido, usa o Elo como medida de força.
    Caso contrário, usa um valor padrão.
    
    Retorna:
    - selecoes: lista de seleções
    - elo_dict: dicionário {seleção: elo}
    - grupos_dict: dicionário {grupo: [seleções]}
    - bandeiras_dict: dicionário {seleção: url_bandeira}
    - stats_gols_dict: dicionário {seleção: (gols_feitos_media, gols_sofridos_media)}
    """
    selecoes = df['Seleção'].tolist()
    grupos_dict = df.groupby('Grupo')['Seleção'].apply(list).to_dict()
    bandeiras_dict = dict(zip(df['Seleção'], df['Link_Bandeira']))
    
    # Se temos dados de Elo, usar como força
    if df_elo is not None:
        elo_dict = {}
        stats_gols_dict = {}
        for selecao in selecoes:
            elo_dict[selecao] = obter_elo_selecao(selecao, df_elo)
            stats_gols_dict[selecao] = obter_estatisticas_gols(selecao, df_elo)
    else:
        # Fallback: usar valor padrão
        elo_dict = {s: ELO_PADRAO for s in selecoes}
        stats_gols_dict = {s: (1.375, 1.375) for s in selecoes}
    
    return selecoes, elo_dict, grupos_dict, bandeiras_dict, stats_gols_dict

# ============ FUNÇÕES AUXILIARES (MANTIDAS PARA COMPATIBILIDADE) ============

def calcular_poder_ofensivo(row_elo):
    """Calcula o poder ofensivo baseado nas estatísticas"""
    if row_elo is None:
        return 1.0
    try:
        gols_feitos = row_elo.get('goals For', row_elo.get('Goals For', 0))
        total_jogos = row_elo.get('Total matches', row_elo.get('Total Matches', 1))
        if pd.isna(gols_feitos) or pd.isna(total_jogos) or total_jogos == 0:
            return 1.0
        media_gols = gols_feitos / total_jogos
        return media_gols / 1.3
    except:
        return 1.0

def calcular_poder_defensivo(row_elo):
    """Calcula o poder defensivo baseado nas estatísticas"""
    if row_elo is None:
        return 1.0
    try:
        gols_sofridos = row_elo.get('goals Against', row_elo.get('Goals Against', 0))
        total_jogos = row_elo.get('Total matches', row_elo.get('Total Matches', 1))
        if pd.isna(gols_sofridos) or pd.isna(total_jogos) or total_jogos == 0:
            return 1.0
        media_gols_sofridos = gols_sofridos / total_jogos
        return media_gols_sofridos / 1.0
    except:
        return 1.0

def obter_estatisticas_gols(selecao, df_elo):
    """
    Obtém as estatísticas de gols feitos e sofridos por partida de uma seleção.
    
    Retorna:
    - gols_feitos_media: média de gols feitos por partida
    - gols_sofridos_media: média de gols sofridos por partida
    """
    row_elo = mapear_selecao_para_elo(selecao, df_elo)
    
    # Valores padrão (média da liga)
    default_gf = 1.375  # Metade de 2.75
    default_gs = 1.375
    
    if row_elo is None:
        return default_gf, default_gs
    
    try:
        # Tentar obter gols feitos
        gols_feitos = None
        for col in ['goals For', 'Goals For', 'GF', 'Gols_Feitos', 'gf']:
            if col in row_elo.index:
                gols_feitos = row_elo[col]
                break
        
        # Tentar obter gols sofridos
        gols_sofridos = None
        for col in ['goals Against', 'Goals Against', 'GA', 'Gols_Sofridos', 'ga']:
            if col in row_elo.index:
                gols_sofridos = row_elo[col]
                break
        
        # Tentar obter total de jogos
        total_jogos = None
        for col in ['Total matches', 'Total Matches', 'Matches', 'Jogos', 'matches']:
            if col in row_elo.index:
                total_jogos = row_elo[col]
                break
        
        # Calcular médias
        if pd.notna(gols_feitos) and pd.notna(total_jogos) and total_jogos > 0:
            gf_media = float(gols_feitos) / float(total_jogos)
        else:
            gf_media = default_gf
        
        if pd.notna(gols_sofridos) and pd.notna(total_jogos) and total_jogos > 0:
            gs_media = float(gols_sofridos) / float(total_jogos)
        else:
            gs_media = default_gs
        
        return gf_media, gs_media
    except:
        return default_gf, default_gs

def calcular_forca_elo(selecao, df_elo):
    """Calcula a força baseada apenas no rating Elo"""
    return obter_elo_selecao(selecao, df_elo)

def calcular_forca_estatisticas(selecao, df_elo):
    """Calcula a força baseada em estatísticas recentes"""
    row_elo = mapear_selecao_para_elo(selecao, df_elo)
    if row_elo is None:
        return ELO_PADRAO
    
    try:
        vitorias = row_elo.get('Wins', 0) or 0
        empates = row_elo.get('Draws', 0) or 0
        derrotas = row_elo.get('Losses', 0) or 0
        gols_feitos = row_elo.get('goals For', row_elo.get('Goals For', 0)) or 0
        gols_sofridos = row_elo.get('goals Against', row_elo.get('Goals Against', 0)) or 0
        
        total_jogos = vitorias + empates + derrotas
        if total_jogos == 0:
            return ELO_PADRAO
        
        pontos = (vitorias * 3 + empates) / total_jogos
        saldo_gols = (gols_feitos - gols_sofridos) / total_jogos
        forca = 1200 + (pontos * 200) + (saldo_gols * 50)
        
        return max(1000, min(2200, forca))
    except:
        return ELO_PADRAO




