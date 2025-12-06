import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time
from scipy.stats import poisson

# ============ CONFIGURAÇÃO DA PÁGINA ============
st.set_page_config(
    page_title = "Copa 2026 Simulator",
    page_icon = "⚽",
    layout = "wide",
    initial_sidebar_state = "expanded"
)

# ============ CSS CUSTOMIZADO ============
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Outfit:wght@300;400;600;700&display=swap');
    
    :root {
        --bg-dark: #0a0a0f;
        --card-bg: #12121a;
        --accent: #00ff88;
        --accent-dim: #00cc6a;
        --text-primary: #ffffff;
        --text-secondary: #8a8a9a;
        --gold: #ffd700;
        --silver: #c0c0c0;
        --bronze: #cd7f32;
    }
    
    .stApp { background: linear-gradient(135deg, #0a0a0f 0%, #1a1a2e 50%, #0a0a0f 100%); }
    
    .main-title {
        font-family: 'Bebas Neue', sans-serif;
        font-size: 4rem;
        background: linear-gradient(90deg, #00ff88, #00ccff, #ff00ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0;
        letter-spacing: 4px;
    }
    
    .subtitle {
        font-family: 'Outfit', sans-serif;
        color: var(--text-secondary);
        text-align: center;
        font-size: 1.1rem;
        margin-top: -10px;
    }
    
    .stat-card {
        background: linear-gradient(145deg, #1a1a2e, #12121a);
        border: 1px solid rgba(0, 255, 136, 0.2);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
    }
    
    .stat-value {
        font-family: 'Bebas Neue', sans-serif;
        font-size: 3rem;
        color: #00ff88;
        line-height: 1;
    }
    
    .stat-label {
        font-family: 'Outfit', sans-serif;
        color: #8a8a9a;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    
    .team-card {
        background: linear-gradient(145deg, #1e1e2e, #16161f);
        border-left: 4px solid #00ff88;
        padding: 1rem 1.5rem;
        margin: 0.5rem 0;
        border-radius: 0 12px 12px 0;
        display: flex;
        align-items: center;
        gap: 1rem;
    }
    
    .team-rank { font-family: 'Bebas Neue', sans-serif; font-size: 2rem; color: #00ff88; width: 50px; }
    .team-name { font-family: 'Outfit', sans-serif; font-weight: 600; font-size: 1.2rem; color: white; flex: 1; }
    .team-prob { font-family: 'Bebas Neue', sans-serif; font-size: 1.8rem; color: #00ccff; }
    
    .gold { color: #ffd700 !important; border-left-color: #ffd700 !important; }
    .silver { color: #c0c0c0 !important; border-left-color: #c0c0c0 !important; }
    .bronze { color: #cd7f32 !important; border-left-color: #cd7f32 !important; }
    
    .monitor-box {
        background: #0d0d14;
        border: 1px solid #2a2a3a;
        border-radius: 12px;
        padding: 1rem;
        font-family: 'Courier New', monospace;
        font-size: 0.85rem;
        color: #00ff88;
        max-height: 300px;
        overflow-y: auto;
    }
    
    div[data-testid="stMetric"] {
        background: linear-gradient(145deg, #1a1a2e, #12121a);
        border: 1px solid rgba(0, 255, 136, 0.15);
        border-radius: 12px;
        padding: 1rem;
    }
    
    div[data-testid="stMetric"] label { color: #8a8a9a !important; }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] { color: #00ff88 !important; }
    
    .stProgress > div > div { background: linear-gradient(90deg, #00ff88, #00ccff) !important; }
    
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background: #1a1a2e;
        border-radius: 8px;
        color: #8a8a9a;
        border: 1px solid #2a2a3a;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #00ff88, #00ccff) !important;
        color: #0a0a0f !important;
    }
    
    .stSelectbox > div > div { background: #1a1a2e; border-color: #2a2a3a; }
    .stSlider > div > div > div { background: #00ff88; }
    
    .stDataFrame { border-radius: 12px; overflow: hidden; }
</style>
""", unsafe_allow_html = True)

# ============ CONFIGURAÇÕES SIMULAÇÃO ============
MEDIA_GOLS_COPA = 2.75
DIFF_FORCA = 500.0
FORCA_PADRAO = 1350.0

# Pesos para o indicador composto de força
PESO_RANKING_FIFA = 0.60      # 60% ranking FIFA
PESO_PARTICIPACOES = 0.15     # 15% participações em copas
PESO_MELHOR_RESULTADO = 0.25  # 25% melhor resultado histórico

# Mapeamento de melhor resultado para pontuação (0 a 1)
RESULTADO_COPA_PONTOS = {
    'Campeão': 1.0,
    '2º Lugar': 0.85,
    '3º Lugar': 0.75,
    '4º Lugar': 0.65,
    'Quartas de Final': 0.50,
    'Oitavas de Final': 0.35,
    'Fase de Grupos': 0.20,
    'Estreante': 0.10,
    'N/A': 0.10
}

# ============ FUNÇÕES DA SIMULAÇÃO ============
def extrair_melhor_resultado(texto):
    """Extrai o melhor resultado da string (ex: 'Campeão (1978, 1986, 2022)' -> 'Campeão')"""
    if pd.isna(texto) or texto == 'N/A':
        return 'N/A'
    resultado = str(texto).split('(')[0].strip()
    return resultado if resultado in RESULTADO_COPA_PONTOS else 'N/A'

def calcular_forca_composta(row, max_fifa, max_participacoes):
    """Calcula o indicador composto de força da seleção"""
    
    # 1. Componente Ranking FIFA (normalizado)
    pontos_fifa = row['Pontuacao_FIFA_Num']
    fifa_normalizado = pontos_fifa / max_fifa if max_fifa > 0 else 0.5
    
    # 2. Componente Participações (normalizado com escala logarítmica)
    participacoes = row['Participacoes_Num']
    if max_participacoes > 0:
        participacoes_normalizado = np.log1p(participacoes) / np.log1p(max_participacoes)
    else:
        participacoes_normalizado = 0.1
    
    # 3. Componente Melhor Resultado
    melhor_resultado = row['Melhor_Resultado_Limpo']
    resultado_normalizado = RESULTADO_COPA_PONTOS.get(melhor_resultado, 0.10)
    
    # Combina os componentes com os pesos
    forca_normalizada = (
        PESO_RANKING_FIFA * fifa_normalizado +
        PESO_PARTICIPACOES * participacoes_normalizado +
        PESO_MELHOR_RESULTADO * resultado_normalizado
    )
    
    # Converte para escala similar ao ranking FIFA (aprox. 1200-1900)
    forca_final = 1200 + (forca_normalizada * 700)
    
    return forca_final

@st.cache_data
def carregar_dados(caminho_csv):
    df = pd.read_csv(caminho_csv, sep = ';', skiprows = 1)
    
    # Processa Pontuação FIFA
    df['Pontuacao_FIFA_Num'] = df['Pontuacao_Ranking_FIFA'].apply(
        lambda x: float(str(x).replace('.', '').replace(',', '.')) if pd.notna(x) and x != 'N/A' else FORCA_PADRAO
    )
    
    # Processa Participações em Copa
    df['Participacoes_Num'] = df['Participações_Copa_Mundo'].apply(
        lambda x: int(x) if pd.notna(x) and str(x) != 'N/A' else 0
    )
    
    # Processa Melhor Resultado
    df['Melhor_Resultado_Limpo'] = df['Melhor_Resultado_Copa_Mundo'].apply(extrair_melhor_resultado)
    
    # Calcula valores máximos para normalização
    max_fifa = df['Pontuacao_FIFA_Num'].max()
    max_participacoes = df['Participacoes_Num'].max()
    
    # Calcula a força composta
    df['Forca'] = df.apply(lambda row: calcular_forca_composta(row, max_fifa, max_participacoes), axis=1)
    df['Grupo'] = df['Grupo'].str.strip()
    
    return df

def preparar_estruturas(df):
    forca_dict = dict(zip(df['Seleção'], df['Forca']))
    grupos_dict = df.groupby('Grupo')['Seleção'].apply(list).to_dict()
    selecoes = df['Seleção'].tolist()
    bandeiras_dict = dict(zip(df['Seleção'], df['Link_Bandeira']))
    return selecoes, forca_dict, grupos_dict, bandeiras_dict

def get_bandeira_html(selecao, bandeiras_dict, tamanho=24):
    """Retorna HTML da bandeira de uma seleção"""
    url = bandeiras_dict.get(selecao, 'https://flagcdn.com/w320/un.png')
    return f'<img src="{url}" style="width: {tamanho}px; height: auto; border-radius: 3px; vertical-align: middle; margin-right: 8px;">'

def get_bandeira_url(selecao, bandeiras_dict):
    """Retorna URL da bandeira de uma seleção"""
    return bandeiras_dict.get(selecao, 'https://flagcdn.com/w320/un.png')

def simular_jogo(forca_a, forca_b, mata_mata = False):
    diff = (forca_a - forca_b) / DIFF_FORCA
    lambda_a = max(0.1, (MEDIA_GOLS_COPA / 2) * (1 + diff))
    lambda_b = max(0.1, (MEDIA_GOLS_COPA / 2) * (1 - diff))
    gols_a, gols_b = np.random.poisson(lambda_a), np.random.poisson(lambda_b)
    fp_a, fp_b = -np.random.poisson(1.5), -np.random.poisson(1.5)
    if gols_a > gols_b: return 3, 0, gols_a, gols_b, fp_a, fp_b, 0
    if gols_b > gols_a: return 0, 3, gols_a, gols_b, fp_a, fp_b, 1
    if not mata_mata:   return 1, 1, gols_a, gols_b, fp_a, fp_b, -1
    return 0, 0, gols_a, gols_b, fp_a, fp_b, 0 if np.random.random() < (0.5 + diff * 0.1) else 1

def simular_fase_grupos(grupos_dict, forca_dict):
    resultados = []
    for grupo, times in grupos_dict.items():
        stats = {t: [0, 0, 0, 0] for t in times}
        for i in range(len(times)):
            for j in range(i + 1, len(times)):
                t1, t2 = times[i], times[j]
                p1, p2, ga, gb, fp1, fp2, _ = simular_jogo(forca_dict[t1], forca_dict[t2])
                stats[t1][0] += p1; stats[t1][1] += ga - gb; stats[t1][2] += ga; stats[t1][3] += fp1
                stats[t2][0] += p2; stats[t2][1] += gb - ga; stats[t2][2] += gb; stats[t2][3] += fp2
        ranking = sorted(stats.items(), key = lambda x: (x[1][0], x[1][1], x[1][2], x[1][3]), reverse = True)
        for pos, (selecao, stat) in enumerate(ranking):
            resultados.append((selecao, grupo, pos + 1, stat[0], stat[1], stat[2], stat[3]))
    return resultados

def definir_classificados_32(resultados_grupos, forca_dict):
    primeiros = [(t, s[0], s[1], s[2], s[3]) for t, g, p, *s in resultados_grupos if p == 1]
    segundos = [(t, s[0], s[1], s[2], s[3]) for t, g, p, *s in resultados_grupos if p == 2]
    terceiros = sorted([(t, s[0], s[1], s[2], s[3]) for t, g, p, *s in resultados_grupos if p == 3],
                       key = lambda x: (x[1], x[2], x[3], x[4]), reverse = True)[:8]
    todos = primeiros + segundos + terceiros
    return sorted(todos, key = lambda x: (x[1], x[2], x[3], forca_dict[x[0]]), reverse = True)

def rodar_mata_mata(classificados, forca_dict, historico, fase):
    if len(classificados) < 2: return classificados
    vencedores = []
    n = len(classificados)
    for i in range(n // 2):
        t1, t2 = classificados[i][0], classificados[n - 1 - i][0]
        *_, resultado = simular_jogo(forca_dict[t1], forca_dict[t2], mata_mata = True)
        ganhador = t1 if resultado == 0 else t2
        dados = classificados[i] if resultado == 0 else classificados[n - 1 - i]
        vencedores.append(dados)
        historico[ganhador][fase] = 1
    return rodar_mata_mata(vencedores, forca_dict, historico, fase + 1)

def simular_uma_copa(selecoes, forca_dict, grupos_dict):
    historico = {t: [0] * 7 for t in selecoes}
    resultados = simular_fase_grupos(grupos_dict, forca_dict)
    for t, *_ in resultados: historico[t][0] = 1
    classificados = definir_classificados_32(resultados, forca_dict)
    for t, *_ in classificados: historico[t][1] = 1
    rodar_mata_mata(classificados, forca_dict, historico, 2)
    return historico

# ============ HEADER ============
st.markdown('<h1 class="main-title">⚽ COPA 2026 SIMULATOR</h1>', unsafe_allow_html = True)
st.markdown('<p class="subtitle">Simulação Monte Carlo • Análise de Probabilidades • FIFA World Cup USA/CAN/MEX</p>', unsafe_allow_html = True)
st.markdown("---")

# ============ CARREGAR DADOS ============
try:
    df_dados = carregar_dados('dados.csv')
    selecoes, forca_dict, grupos_dict, bandeiras_dict = preparar_estruturas(df_dados)
except:
    st.error("❌ Arquivo `dados.csv` não encontrado!")
    st.stop()

# ============ SIDEBAR ============
with st.sidebar:
    st.markdown("### ⚙️ Configurações")
    n_simulacoes = st.slider("Número de Simulações", min_value = 100, max_value = 100000, value = 10000, step = 100)
    st.markdown(f"**Tempo estimado:** ~{n_simulacoes // 800:.0f}s")
    st.markdown("---")
    
    st.markdown("### 📊 Parâmetros do Modelo")
    media_gols = st.slider("Média de Gols/Jogo", 2.0, 4.0, 2.75, 0.25)
    diff_forca = st.slider("Fator Diferença Força", 300.0, 700.0, 500.0, 50.0)
    
    st.markdown("---")
    st.markdown("### 🎯 Filtros")
    confederacoes = ['Todas'] + sorted(df_dados['Confederação'].unique().tolist())
    filtro_conf = st.selectbox("Confederação", confederacoes)

# ============ TABS PRINCIPAIS ============
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📋 DATASET", "🎮 SIMULAÇÃO", "📊 RESULTADOS", "🎬 AO VIVO", "💰 ODDS IMPLÍCITAS", "⚔️ SIMULADOR PARTIDA"])

# ============ TAB 1: DATASET ============
with tab1:
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("🌍 Seleções", len(df_dados))
    with col2: st.metric("🏆 Grupos", df_dados['Grupo'].nunique())
    with col3: st.metric("⭐ Confederações", df_dados['Confederação'].nunique())
    with col4: st.metric("🆕 Estreantes", len(df_dados[df_dados['Melhor_Resultado_Copa_Mundo'] == 'Estreante']))
    
    # Info sobre o indicador composto
    st.markdown(f"""
    <div style="background: linear-gradient(145deg, #1a1a2e, #12121a); border: 1px solid rgba(0, 255, 136, 0.3); border-radius: 12px; padding: 1rem; margin: 1rem 0;">
        <div style="color: #00ff88; font-size: 1.1rem; font-weight: 600; margin-bottom: 0.5rem;">📊 Indicador Composto de Força</div>
        <div style="color: #8a8a9a; font-size: 0.9rem;">
            A força de cada seleção é calculada combinando três fatores:
            <ul style="margin: 0.5rem 0;">
                <li><b style="color: #00ccff;">Ranking FIFA:</b> {PESO_RANKING_FIFA*100:.0f}% - Pontuação atual no ranking</li>
                <li><b style="color: #ffd700;">Participações em Copa:</b> {PESO_PARTICIPACOES*100:.0f}% - Experiência em mundiais</li>
                <li><b style="color: #ff00ff;">Melhor Resultado:</b> {PESO_MELHOR_RESULTADO*100:.0f}% - Histórico de desempenho</li>
            </ul>
        </div>
    </div>
    """, unsafe_allow_html = True)
    
    st.markdown("### 🗂️ Dataset Completo")
    
    df_display = df_dados.copy()
    if filtro_conf != 'Todas':
        df_display = df_display[df_display['Confederação'] == filtro_conf]
    
    # Renomeia colunas para exibição mais amigável
    df_display = df_display[['Seleção', 'Grupo', 'Confederação', 'Ranking_FIFA_Nov_2025', 'Pontuacao_FIFA_Num', 'Participacoes_Num', 'Melhor_Resultado_Limpo', 'Forca']]
    df_display.columns = ['Seleção', 'Grupo', 'Confederação', 'Ranking FIFA', 'Pts FIFA', 'Participações', 'Melhor Resultado', 'Força Composta']
    df_display = df_display.sort_values('Força Composta', ascending = False)
    df_display['Força Composta'] = df_display['Força Composta'].round(2)
    st.dataframe(df_display, use_container_width = True, height = 400)
    
    # Visualização dos Grupos com Bandeiras
    st.markdown("### 🏟️ Grupos da Copa 2026")
    
    grupos_ordenados = sorted(grupos_dict.keys())
    num_grupos = len(grupos_ordenados)
    
    # Dividir grupos em 2 linhas de 6
    for i in range(0, num_grupos, 6):
        cols = st.columns(6)
        for j, col in enumerate(cols):
            if i + j < num_grupos:
                grupo = grupos_ordenados[i + j]
                times_grupo = grupos_dict[grupo]
                with col:
                    times_list = []
                    for selecao in times_grupo:
                        bandeira_url = get_bandeira_url(selecao, bandeiras_dict)
                        times_list.append(f'<div style="display: flex; align-items: center; gap: 6px; padding: 5px 0; border-bottom: 1px solid #2a2a3a;"><img src="{bandeira_url}" style="width: 24px; height: auto; border-radius: 2px;"><span style="color: white; font-size: 0.75rem;">{selecao}</span></div>')
                    times_html = "".join(times_list)
                    grupo_html = f'<div style="background: linear-gradient(145deg, #1a1a2e, #12121a); border: 1px solid #2a2a3a; border-radius: 10px; padding: 0.8rem; margin-bottom: 0.5rem;"><div style="font-family: Bebas Neue, sans-serif; font-size: 1.1rem; color: #00ccff; text-align: center; border-bottom: 2px solid #00ccff; padding-bottom: 0.4rem; margin-bottom: 0.4rem;">GRUPO {grupo}</div>{times_html}</div>'
                    st.markdown(grupo_html, unsafe_allow_html=True)
    
    st.markdown("### 📊 Estatísticas por Grupo")
    col1, col2 = st.columns(2)
    
    with col1:
        fig_grupos = px.bar(
            df_dados.groupby('Grupo')['Forca'].mean().reset_index(),
            x = 'Grupo', y = 'Forca',
            color = 'Forca', color_continuous_scale = ['#2d5a4a', '#00ff88'],
            title = "Força Média por Grupo"
        )
        fig_grupos.update_layout(
            plot_bgcolor = 'rgba(0,0,0,0)', paper_bgcolor = 'rgba(0,0,0,0)',
            font_color = '#8a8a9a', title_font_color = '#00ff88'
        )
        st.plotly_chart(fig_grupos, use_container_width = True)
    
    with col2:
        fig_conf = px.bar(
            df_dados.groupby('Confederação')['Forca'].mean().reset_index(),
            x = 'Confederação', y = 'Forca',
            color = 'Forca', color_continuous_scale = ['#2d4a5a', '#00ccff'],
            title = "Força Média por Confederação"
        )
        fig_conf.update_layout(
            plot_bgcolor = 'rgba(0,0,0,0)', paper_bgcolor = 'rgba(0,0,0,0)',
            font_color = '#8a8a9a', title_font_color = '#00ff88'
        )
        st.plotly_chart(fig_conf, use_container_width = True)
    
    # Layout: Top 20 seleções à esquerda e gráfico de composição à direita
    st.markdown("### 🏆 Top 20 Seleções por Força")
    
    top20_forca = df_dados.nlargest(20, 'Forca').copy()
    
    col_cards, col_grafico = st.columns([1, 1])
    
    # Coluna da esquerda: Cards das Top 20 seleções
    with col_cards:
        for idx, (_, row) in enumerate(top20_forca.iterrows()):
            pos = idx + 1
            medalha = "🥇" if pos == 1 else ("🥈" if pos == 2 else ("🥉" if pos == 3 else f"#{pos}"))
            cor_borda = '#ffd700' if pos == 1 else ('#c0c0c0' if pos == 2 else ('#cd7f32' if pos == 3 else '#00ff88'))
            st.markdown(f"""
            <div style="background: linear-gradient(145deg, #1e1e2e, #16161f); border-left: 4px solid {cor_borda}; padding: 0.5rem; margin: 0.25rem 0; border-radius: 0 8px 8px 0; display: flex; align-items: center; gap: 0.5rem;">
                <img src="{get_bandeira_url(row['Seleção'], bandeiras_dict)}" style="width: 32px; height: auto; border-radius: 3px; box-shadow: 0 2px 6px rgba(0,0,0,0.3);">
                <div style="flex: 1;">
                    <div style="font-family: 'Outfit', sans-serif; font-weight: 600; font-size: 0.9rem; color: white;">{medalha} {row['Seleção']}</div>
                    <div style="font-size: 0.7rem; color: #8a8a9a;">{row['Confederação']} • Grupo {row['Grupo']}</div>
                </div>
                <div style="text-align: right;">
                    <div style="font-family: 'Bebas Neue', sans-serif; font-size: 1.1rem; color: #00ff88;">{row['Forca']:.0f}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Coluna da direita: Gráfico de composição da força
    with col_grafico:
        max_fifa = df_dados['Pontuacao_FIFA_Num'].max()
        max_part = df_dados['Participacoes_Num'].max()
        
        # Usar todas as seleções ordenadas por força (descendente - maiores primeiro)
        todas_selecoes = df_dados.sort_values('Forca', ascending=False).copy()
        
        # Calcular componentes normalizados
        todas_selecoes['Comp_FIFA'] = (todas_selecoes['Pontuacao_FIFA_Num'] / max_fifa) * PESO_RANKING_FIFA * 100
        todas_selecoes['Comp_Part'] = (np.log1p(todas_selecoes['Participacoes_Num']) / np.log1p(max_part)) * PESO_PARTICIPACOES * 100
        todas_selecoes['Comp_Resultado'] = todas_selecoes['Melhor_Resultado_Limpo'].apply(lambda x: RESULTADO_COPA_PONTOS.get(x, 0.1)) * PESO_MELHOR_RESULTADO * 100
        
        # Lista ordenada de seleções (maiores primeiro)
        ordem_selecoes = todas_selecoes['Seleção'].tolist()
        
        fig_composicao = go.Figure()
        
        fig_composicao.add_trace(go.Bar(
            name = 'Ranking FIFA',
            y = todas_selecoes['Seleção'],
            x = todas_selecoes['Comp_FIFA'],
            orientation = 'h',
            marker_color = '#00ccff'
        ))
        
        fig_composicao.add_trace(go.Bar(
            name = 'Participações Copa',
            y = todas_selecoes['Seleção'],
            x = todas_selecoes['Comp_Part'],
            orientation = 'h',
            marker_color = '#ffd700'
        ))
        
        fig_composicao.add_trace(go.Bar(
            name = 'Melhor Resultado',
            y = todas_selecoes['Seleção'],
            x = todas_selecoes['Comp_Resultado'],
            orientation = 'h',
            marker_color = '#ff00ff'
        ))
        
        fig_composicao.update_layout(
            title = "🧬 Composição da Força - Todas as Seleções",
            barmode = 'stack',
            bargap = 0.2,
            plot_bgcolor = 'rgba(0,0,0,0)', paper_bgcolor = 'rgba(0,0,0,0)',
            font_color = '#8a8a9a', title_font_color = '#00ff88',
            height = 1320,
            xaxis_title = "Contribuição (%)",
            yaxis_title = "",
            yaxis = dict(categoryorder = 'array', categoryarray = ordem_selecoes[::-1]),
            legend = dict(orientation = "h", yanchor = "bottom", y = 1.02, xanchor = "right", x = 1)
        )
        st.plotly_chart(fig_composicao, use_container_width = True)

# ============ TAB 2: SIMULAÇÃO ============
with tab2:
    st.markdown("### 🎮 Executar Simulação Monte Carlo")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{n_simulacoes:,}</div>
            <div class="stat-label">Simulações Configuradas</div>
        </div>
        """, unsafe_allow_html = True)
    
    with col2:
        iniciar = st.button("🚀 INICIAR SIMULAÇÃO", use_container_width = True, type = "primary")
    
    if iniciar:
        st.markdown("### 📡 Monitor de Execução")
        
        MEDIA_GOLS_COPA = media_gols
        DIFF_FORCA = diff_forca
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        monitor = st.empty()
        metricas = st.empty()
        
        idx = {t: i for i, t in enumerate(selecoes)}
        acumulador = np.zeros((len(selecoes), 7), dtype = np.int32)
        
        logs = []
        tempo_inicio = time.time()
        campeoes_parciais = {}
        
        for i in range(n_simulacoes):
            resultado = simular_uma_copa(selecoes, forca_dict, grupos_dict)
            
            for time_sel, stats in resultado.items():
                acumulador[idx[time_sel]] += stats
                if stats[6] == 1:  # campeão
                    campeoes_parciais[time_sel] = campeoes_parciais.get(time_sel, 0) + 1
            
            if (i + 1) % 100 == 0 or i == 0:
                progresso = (i + 1) / n_simulacoes
                progress_bar.progress(progresso)
                
                tempo_decorrido = time.time() - tempo_inicio
                vel = (i + 1) / tempo_decorrido if tempo_decorrido > 0 else 0
                tempo_restante = (n_simulacoes - i - 1) / vel if vel > 0 else 0
                
                status_text.markdown(f"**Simulação {i + 1:,}/{n_simulacoes:,}** • Velocidade: **{vel:.0f} sim/s** • ETA: **{tempo_restante:.1f}s**")
                
                top3 = sorted(campeoes_parciais.items(), key = lambda x: x[1], reverse = True)[:3]
                log_entry = f"[{datetime.now().strftime('%H:%M:%S')}] Sim #{i+1:,} | Top: {top3[0][0] if top3 else 'N/A'} ({top3[0][1]/(i+1)*100:.1f}%)"
                logs.append(log_entry)
                
                monitor.markdown(f'<div class="monitor-box">{chr(10).join(logs[-15:])}</div>', unsafe_allow_html = True)
                
                with metricas:
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("⏱️ Tempo", f"{tempo_decorrido:.1f}s")
                    m2.metric("🚀 Velocidade", f"{vel:.0f}/s")
                    m3.metric("✅ Concluídas", f"{i + 1:,}")
                    if top3:
                        m4.metric("👑 Líder", f"{top3[0][0]} ({top3[0][1]/(i+1)*100:.1f}%)")
        
        progress_bar.progress(1.0)
        tempo_total = time.time() - tempo_inicio
        status_text.markdown(f"✅ **CONCLUÍDO!** {n_simulacoes:,} simulações em **{tempo_total:.2f}s** ({n_simulacoes/tempo_total:.0f} sim/s)")
        
        colunas = ['Fase Grupos', 'Top 32', 'Oitavas', 'Quartas', 'Semis', 'Final', 'Campeão']
        df_resultado = pd.DataFrame(acumulador / n_simulacoes, index = selecoes, columns = colunas)
        df_resultado = df_resultado.sort_values('Campeão', ascending = False)
        
        st.session_state['resultado'] = df_resultado
        st.session_state['n_sims'] = n_simulacoes
        st.session_state['tempo'] = tempo_total
        
        st.success("🎉 Simulação finalizada! Vá para a aba **RESULTADOS** para ver a análise completa.")
        
        # Preview rápido dos resultados
        st.markdown("---")
        st.markdown("### 🏆 Preview - Top 5 Favoritos")
        
        top5 = df_resultado.head(5)
        for idx, (selecao, row) in enumerate(top5.iterrows()):
            medalha = "🥇" if idx == 0 else ("🥈" if idx == 1 else ("🥉" if idx == 2 else f"#{idx+1}"))
            cor = '#ffd700' if idx == 0 else ('#c0c0c0' if idx == 1 else ('#cd7f32' if idx == 2 else '#00ff88'))
            st.markdown(f"""
            <div style="background: linear-gradient(145deg, #1e1e2e, #16161f); border-left: 4px solid {cor}; padding: 0.8rem 1rem; margin: 0.3rem 0; border-radius: 0 8px 8px 0; display: flex; align-items: center; justify-content: space-between;">
                <div style="display: flex; align-items: center; gap: 10px;">
                    <span style="font-size: 1.5rem;">{medalha}</span>
                    <img src="{get_bandeira_url(selecao, bandeiras_dict)}" style="width: 32px; border-radius: 3px;">
                    <span style="color: white; font-size: 1.1rem; font-weight: 600;">{selecao}</span>
                </div>
                <span style="color: #00ff88; font-family: 'Bebas Neue', sans-serif; font-size: 1.5rem;">{row['Campeão']*100:.1f}%</span>
            </div>
            """, unsafe_allow_html = True)
        
        st.info("👉 Acesse a aba **📊 RESULTADOS** para ver análises detalhadas, gráficos e exportar os dados.")

# ============ TAB 3: RESULTADOS ============
with tab3:
    st.markdown("### 📊 Resultados da Simulação Monte Carlo")
    
    if 'resultado' not in st.session_state or st.session_state.get('resultado') is None:
        st.markdown("""
        <div style="text-align: center; padding: 3rem; background: linear-gradient(145deg, #1a1a2e, #12121a); border-radius: 16px; margin: 2rem 0;">
            <div style="font-size: 4rem; margin-bottom: 1rem;">🎮</div>
            <div style="font-family: 'Bebas Neue', sans-serif; font-size: 2rem; color: #00ff88; margin-bottom: 0.5rem;">
                Nenhuma simulação executada ainda!
            </div>
            <div style="color: #8a8a9a; font-size: 1.1rem;">
                Vá para a aba <b style="color: #00ccff;">SIMULAÇÃO</b> e clique em <b style="color: #00ff88;">INICIAR SIMULAÇÃO</b>
            </div>
        </div>
        """, unsafe_allow_html = True)
    else:
        df_res = st.session_state['resultado']
        n_sims = st.session_state.get('n_sims', 0)
        tempo = st.session_state.get('tempo', 0)
        
        # Métricas da simulação
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        with col_m1:
            st.metric("🎲 Simulações", f"{n_sims:,}")
        with col_m2:
            st.metric("⏱️ Tempo", f"{tempo:.2f}s")
        with col_m3:
            st.metric("🚀 Velocidade", f"{n_sims/tempo:.0f}/s" if tempo > 0 else "N/A")
        with col_m4:
            st.metric("🏆 Favorito", df_res.index[0])
        
        st.markdown("---")
        
        st.markdown("### 🏆 Probabilidades de Título")
        
        col1, col2, col3 = st.columns(3)
        top3 = df_res.head(3)
        
        with col1:
            st.markdown(f"""
            <div class="team-card gold">
                <div class="team-rank gold">🥇</div>
                <div class="team-name">{get_bandeira_html(top3.index[0], bandeiras_dict, 32)}{top3.index[0]}</div>
                <div class="team-prob">{top3['Campeão'].iloc[0]*100:.1f}%</div>
            </div>
            """, unsafe_allow_html = True)
        
        with col2:
            st.markdown(f"""
            <div class="team-card silver">
                <div class="team-rank silver">🥈</div>
                <div class="team-name">{get_bandeira_html(top3.index[1], bandeiras_dict, 32)}{top3.index[1]}</div>
                <div class="team-prob">{top3['Campeão'].iloc[1]*100:.1f}%</div>
            </div>
            """, unsafe_allow_html = True)
        
        with col3:
            st.markdown(f"""
            <div class="team-card bronze">
                <div class="team-rank bronze">🥉</div>
                <div class="team-name">{get_bandeira_html(top3.index[2], bandeiras_dict, 32)}{top3.index[2]}</div>
                <div class="team-prob">{top3['Campeão'].iloc[2]*100:.1f}%</div>
            </div>
            """, unsafe_allow_html = True)
        
        st.markdown("### 📊 Top 10 Favoritos")
        
        fig_top10 = go.Figure()
        top10 = df_res.head(10).iloc[::-1]
        
        fig_top10.add_trace(go.Bar(
            y = top10.index, x = top10['Campeão'] * 100,
            orientation = 'h',
            marker = dict(
                color = top10['Campeão'] * 100,
                colorscale = [[0, '#1a1a2e'], [0.5, '#00ccff'], [1, '#00ff88']],
                line = dict(width = 0)
            ),
            text = [f"{v:.1f}%" for v in top10['Campeão'] * 100],
            textposition = 'outside',
            textfont = dict(color = '#00ff88', size = 14)
        ))
        
        fig_top10.update_layout(
            plot_bgcolor = 'rgba(0,0,0,0)', paper_bgcolor = 'rgba(0,0,0,0)',
            font_color = '#8a8a9a', height = 500,
            xaxis_title = "Probabilidade de Título (%)",
            yaxis_title = "",
            margin = dict(l = 120, r = 80)
        )
        st.plotly_chart(fig_top10, use_container_width = True)
        
        st.markdown("### 📈 Progressão por Fase")
        
        col1, col2 = st.columns(2)
        
        with col1:
            selecao_detalhe = st.selectbox("Selecionar Seleção", df_res.index.tolist(), key = "selecao_tab3")
            
            st.markdown(f"""
            <div style="background: linear-gradient(145deg, #1e1e2e, #16161f); border: 2px solid #00ff88; border-radius: 16px; padding: 1.5rem; margin: 1rem 0; text-align: center;">
                <img src="{get_bandeira_url(selecao_detalhe, bandeiras_dict)}" style="width: 80px; border-radius: 8px; box-shadow: 0 4px 15px rgba(0,255,136,0.3);">
                <div style="font-family: 'Bebas Neue', sans-serif; font-size: 2rem; color: white; margin-top: 0.5rem;">{selecao_detalhe}</div>
                <div style="color: #00ff88; font-size: 1.5rem; font-family: 'Bebas Neue', sans-serif;">
                    {df_res.loc[selecao_detalhe]['Campeão']*100:.1f}% de chance de título
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            dados_sel = df_res.loc[selecao_detalhe]
            fases = ['Fase Grupos', 'Top 32', 'Oitavas', 'Quartas', 'Semis', 'Final', 'Campeão']
            
            fig_funil = go.Figure(go.Funnel(
                y = fases,
                x = [dados_sel[f] * 100 for f in fases],
                textinfo = "value+percent total",
                marker = dict(color = ['#00ff88', '#00ddaa', '#00ccbb', '#00bbcc', '#00aadd', '#0099ee', '#0088ff']),
                textfont = dict(size = 14)
            ))
            
            fig_funil.update_layout(
                title = f"Progressão por Fase",
                plot_bgcolor = 'rgba(0,0,0,0)', paper_bgcolor = 'rgba(0,0,0,0)',
                font_color = '#8a8a9a', title_font_color = '#00ff88'
            )
            st.plotly_chart(fig_funil, use_container_width = True)
        
        with col2:
            st.markdown("#### 🔥 Heatmap - Top 15")
            
            top15 = df_res.head(15)
            fig_heat = px.imshow(
                top15.values * 100,
                labels = dict(x = "Fase", y = "Seleção", color = "%"),
                x = top15.columns.tolist(),
                y = top15.index.tolist(),
                color_continuous_scale = 'viridis',
                aspect = "auto"
            )
            fig_heat.update_layout(
                plot_bgcolor = 'rgba(0,0,0,0)', paper_bgcolor = 'rgba(0,0,0,0)',
                font_color = '#8a8a9a', height = 500
            )
            st.plotly_chart(fig_heat, use_container_width = True)
        
        st.markdown("### 📋 Tabela Completa")
        
        df_display_res = df_res.copy()
        df_display_res = df_display_res.round(4)
        for col in df_display_res.columns:
            df_display_res[col] = (df_display_res[col] * 100).round(2).astype(str) + '%'
        
        st.dataframe(df_display_res, use_container_width = True, height = 400)
        
        st.markdown("### 💾 Exportar Resultados")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            csv = df_res.to_csv()
            st.download_button(
                "📥 Download CSV",
                csv, f"copa2026_sim_{n_sims}.csv",
                "text/csv", use_container_width = True
            )
        
        with col2:
            try:
                import io
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine = 'openpyxl') as writer:
                    df_res.to_excel(writer)
                st.download_button(
                    "📥 Download Excel",
                    buffer.getvalue(), 
                    f"copa2026_sim_{n_sims}.xlsx",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width = True
                )
            except Exception as e:
                st.warning("Excel não disponível")
        
        with col3:
            st.metric("📊 Total de Seleções", f"{len(df_res)}")

# ============ TAB 4: SIMULAÇÃO AO VIVO ============
with tab4:
    st.markdown("### 🎬 Simulação Ao Vivo - Jogo a Jogo")
    st.markdown("Acompanhe uma Copa do Mundo completa, vendo cada jogo acontecer em tempo real!")
    
    # CSS adicional para cards de jogo e histórico
    st.markdown("""
    <style>
        .match-card {
            background: linear-gradient(145deg, #1a1a2e, #12121a);
            border: 1px solid #2a2a3a;
            border-radius: 12px;
            padding: 1rem;
            margin: 0.5rem 0;
            text-align: center;
        }
        .match-teams { display: flex; justify-content: space-between; align-items: center; }
        .match-team { font-family: 'Outfit', sans-serif; font-size: 1.1rem; color: white; flex: 1; }
        .match-score { 
            font-family: 'Bebas Neue', sans-serif; 
            font-size: 2.5rem; 
            color: #00ff88;
            padding: 0 1rem;
            min-width: 100px;
        }
        .match-info { font-size: 0.8rem; color: #8a8a9a; margin-top: 0.5rem; }
        .group-header {
            font-family: 'Bebas Neue', sans-serif;
            font-size: 1.5rem;
            color: #00ccff;
            border-bottom: 2px solid #00ccff;
            padding-bottom: 0.3rem;
            margin: 1rem 0;
        }
        .phase-title {
            font-family: 'Bebas Neue', sans-serif;
            font-size: 2rem;
            color: #ff00ff;
            text-align: center;
            margin: 1.5rem 0;
        }
        .winner-card {
            background: linear-gradient(145deg, #2a2a1e, #1a1a0f);
            border: 2px solid #ffd700;
            border-radius: 16px;
            padding: 2rem;
            text-align: center;
            animation: glow 2s ease-in-out infinite alternate;
        }
        @keyframes glow {
            from { box-shadow: 0 0 20px #ffd700; }
            to { box-shadow: 0 0 40px #ffd700, 0 0 60px #ffa500; }
        }
        .champion-name {
            font-family: 'Bebas Neue', sans-serif;
            font-size: 3rem;
            color: #ffd700;
        }
        .history-card {
            background: linear-gradient(145deg, #16162a, #0f0f1a);
            border: 1px solid #3a3a4a;
            border-radius: 12px;
            padding: 1.2rem;
            margin: 0.8rem 0;
            transition: all 0.3s ease;
        }
        .history-card:hover {
            border-color: #00ff88;
            transform: translateX(5px);
        }
        .history-edition {
            font-family: 'Bebas Neue', sans-serif;
            font-size: 1.3rem;
            color: #00ccff;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        .history-champion {
            font-family: 'Outfit', sans-serif;
            font-size: 1.4rem;
            font-weight: 700;
            color: #ffd700;
            margin: 0.5rem 0;
        }
        .history-details {
            font-size: 0.85rem;
            color: #8a8a9a;
            display: flex;
            flex-wrap: wrap;
            gap: 1rem;
            margin-top: 0.5rem;
        }
        .history-badge {
            background: rgba(0, 255, 136, 0.1);
            border: 1px solid rgba(0, 255, 136, 0.3);
            border-radius: 20px;
            padding: 0.2rem 0.8rem;
            font-size: 0.75rem;
            color: #00ff88;
        }
        .hall-of-fame-card {
            background: linear-gradient(145deg, #1a1a0f, #2a2a1e);
            border: 2px solid #ffd700;
            border-radius: 16px;
            padding: 1.5rem;
            text-align: center;
            margin: 0.5rem 0;
        }
        .hall-count {
            font-family: 'Bebas Neue', sans-serif;
            font-size: 3rem;
            color: #ffd700;
        }
        .hall-team {
            font-family: 'Outfit', sans-serif;
            font-weight: 600;
            font-size: 1.1rem;
            color: white;
        }
        .mini-stat {
            background: #1a1a2e;
            border-radius: 8px;
            padding: 0.8rem;
            text-align: center;
        }
        .mini-stat-value {
            font-family: 'Bebas Neue', sans-serif;
            font-size: 1.8rem;
            color: #00ff88;
        }
        .mini-stat-label {
            font-size: 0.75rem;
            color: #8a8a9a;
            text-transform: uppercase;
        }
    </style>
    """, unsafe_allow_html = True)
    
    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([1, 1, 1])
    
    with col_ctrl1:
        velocidade = st.select_slider("⏱️ Velocidade", options = ["Lento", "Normal", "Rápido", "Instantâneo"], value = "Normal")
        delay_map = {"Lento": 1.0, "Normal": 0.3, "Rápido": 0.1, "Instantâneo": 0.0}
        delay = delay_map[velocidade]
    
    with col_ctrl2:
        if st.button("🎲 NOVA COPA", use_container_width = True, type = "primary"):
            st.session_state['live_fase'] = 'grupos'
            st.session_state['live_jogos'] = []
            st.session_state['live_stats'] = {g: {t: [0, 0, 0, 0, 0, 0] for t in times} for g, times in grupos_dict.items()}
            st.session_state['live_running'] = True
            st.session_state['live_grupo_atual'] = 0
            st.session_state['live_classificados'] = []
            st.session_state['live_mata_mata'] = []
            st.session_state['live_campeao'] = None
            st.session_state['live_vice'] = None
            st.session_state['live_semifinalistas'] = []
            st.session_state['live_final_placar'] = None
    
    with col_ctrl3:
        if st.button("⏹️ PARAR", use_container_width = True):
            st.session_state['live_running'] = False
    
    # Inicializar estado
    if 'live_running' not in st.session_state:
        st.session_state['live_running'] = False
        st.session_state['live_fase'] = 'grupos'
        st.session_state['live_jogos'] = []
        st.session_state['live_stats'] = {}
        st.session_state['live_classificados'] = []
        st.session_state['live_mata_mata'] = []
        st.session_state['live_campeao'] = None
        st.session_state['live_vice'] = None
        st.session_state['live_semifinalistas'] = []
        st.session_state['live_final_placar'] = None
    
    # Inicializar histórico de copas
    if 'historico_copas' not in st.session_state:
        st.session_state['historico_copas'] = []
    
    if st.session_state.get('live_running', False):
        
        # ========== FASE DE GRUPOS ==========
        if st.session_state['live_fase'] == 'grupos':
            st.markdown("## 🏟️ FASE DE GRUPOS")
            
            stats = {g: {t: [0, 0, 0, 0, 0, 0] for t in times} for g, times in grupos_dict.items()}
            todos_jogos = []
            
            # Gerar todos os jogos
            for grupo in sorted(grupos_dict.keys()):
                times = grupos_dict[grupo]
                for i in range(len(times)):
                    for j in range(i + 1, len(times)):
                        todos_jogos.append((grupo, times[i], times[j]))
            
            jogos_lista = []
            
            col_jogos, col_tabela = st.columns([3, 2])
            jogos_container = col_jogos.empty()
            tabela_container = col_tabela.empty()
            
            for idx, (grupo, t1, t2) in enumerate(todos_jogos):
                p1, p2, ga, gb, fp1, fp2, resultado = simular_jogo(forca_dict[t1], forca_dict[t2])
                
                # Atualizar stats
                stats[grupo][t1][0] += p1
                stats[grupo][t2][0] += p2
                stats[grupo][t1][4] += ga
                stats[grupo][t1][5] += gb
                stats[grupo][t2][4] += gb
                stats[grupo][t2][5] += ga
                
                if resultado == 0:
                    stats[grupo][t1][1] += 1; stats[grupo][t2][3] += 1
                elif resultado == 1:
                    stats[grupo][t2][1] += 1; stats[grupo][t1][3] += 1
                else:
                    stats[grupo][t1][2] += 1; stats[grupo][t2][2] += 1
                
                jogos_lista.insert(0, {'Grupo': grupo, 'Jogo': f"{idx+1}/{len(todos_jogos)}", 'Time 1': t1, 'Placar': f"{ga} x {gb}", 'Time 2': t2})
                
                # Exibir jogos com bandeiras
                with jogos_container.container():
                    st.markdown(f"""
                    <div style="background: linear-gradient(145deg, #1a1a2e, #12121a); border: 1px solid #2a2a3a; border-radius: 12px; padding: 1rem; margin-bottom: 1rem;">
                        <div style="text-align: center; color: #00ccff; font-size: 0.9rem; margin-bottom: 0.5rem;">GRUPO {grupo}</div>
                        <div style="display: flex; justify-content: center; align-items: center; gap: 1rem;">
                            <div style="display: flex; align-items: center; gap: 8px;">
                                <img src="{get_bandeira_url(t1, bandeiras_dict)}" style="width: 36px; border-radius: 3px;">
                                <span style="color: white; font-size: 1.1rem;">{t1}</span>
                            </div>
                            <div style="font-family: 'Bebas Neue', sans-serif; font-size: 2rem; color: #00ff88; padding: 0 1rem;">{ga} x {gb}</div>
                            <div style="display: flex; align-items: center; gap: 8px;">
                                <span style="color: white; font-size: 1.1rem;">{t2}</span>
                                <img src="{get_bandeira_url(t2, bandeiras_dict)}" style="width: 36px; border-radius: 3px;">
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    df_jogos = pd.DataFrame(jogos_lista[:15])
                    st.dataframe(df_jogos, use_container_width = True, height = 300, hide_index = True)
                
                # Atualizar tabela
                with tabela_container.container():
                    for g in sorted(grupos_dict.keys())[:4]:
                        ranking = sorted(stats[g].items(), key = lambda x: (x[1][0], x[1][4] - x[1][5], x[1][4]), reverse = True)
                        df_tab = pd.DataFrame([
                            {'Pos': i+1, 'Seleção': sel, 'P': s[0], 'V': s[1], 'E': s[2], 'D': s[3], 'SG': s[4] - s[5]}
                            for i, (sel, s) in enumerate(ranking)
                        ])
                        st.markdown(f"**Grupo {g}**")
                        st.dataframe(df_tab, use_container_width = True, height = 140, hide_index = True)
                
                if delay > 0: time.sleep(delay)
                
                if not st.session_state.get('live_running', False): break
            
            # Classificar times
            if st.session_state.get('live_running', False):
                resultados_grupos = []
                for grupo in sorted(grupos_dict.keys()):
                    ranking = sorted(stats[grupo].items(), key = lambda x: (x[1][0], x[1][4] - x[1][5], x[1][4]), reverse = True)
                    for pos, (sel, s) in enumerate(ranking):
                        resultados_grupos.append((sel, grupo, pos + 1, s[0], s[4] - s[5], s[4], 0))
                
                st.session_state['live_resultados_grupos'] = resultados_grupos
                st.session_state['live_fase'] = 'oitavas'
                st.rerun()
        
        # ========== MATA-MATA ==========
        elif st.session_state['live_fase'] in ['oitavas', 'quartas', 'semis', 'final']:
            nomes_fases = {'oitavas': '⚔️ OITAVAS DE FINAL', 'quartas': '🔥 QUARTAS DE FINAL', 'semis': '💥 SEMIFINAIS', 'final': '🏆 FINAL'}
            st.markdown(f"## {nomes_fases[st.session_state['live_fase']]}")
            
            # Definir classificados
            if st.session_state['live_fase'] == 'oitavas':
                resultados = st.session_state['live_resultados_grupos']
                primeiros = [(t, s[0], s[1], s[2], s[3]) for t, g, p, *s in resultados if p == 1]
                segundos = [(t, s[0], s[1], s[2], s[3]) for t, g, p, *s in resultados if p == 2]
                terceiros = sorted([(t, s[0], s[1], s[2], s[3]) for t, g, p, *s in resultados if p == 3],
                                   key = lambda x: (x[1], x[2], x[3], x[4]), reverse = True)[:8]
                todos = primeiros + segundos + terceiros
                classificados = sorted(todos, key = lambda x: (x[1], x[2], x[3], forca_dict[x[0]]), reverse = True)
            else:
                classificados = st.session_state['live_classificados']
            
            # Guardar semifinalistas
            if st.session_state['live_fase'] == 'semis':
                st.session_state['live_semifinalistas'] = [c[0] for c in classificados]
            
            vencedores = []
            perdedores = []
            n = len(classificados)
            jogos_fase = []
            
            col_jogos, col_class = st.columns([3, 2])
            jogos_container = col_jogos.empty()
            class_container = col_class.empty()
            
            for i in range(n // 2):
                t1, t2 = classificados[i][0], classificados[n - 1 - i][0]
                p1, p2, ga, gb, fp1, fp2, resultado = simular_jogo(forca_dict[t1], forca_dict[t2], mata_mata = True)
                
                ganhador = t1 if resultado == 0 else t2
                perdedor = t2 if resultado == 0 else t1
                dados = classificados[i] if resultado == 0 else classificados[n - 1 - i]
                vencedores.append(dados)
                perdedores.append(perdedor)
                
                penaltis = " (pen)" if ga == gb else ""
                
                # Guardar placar da final
                if st.session_state['live_fase'] == 'final':
                    st.session_state['live_final_placar'] = f"{t1} {ga} x {gb} {t2}{penaltis}"
                    st.session_state['live_vice'] = perdedor
                
                jogos_fase.append({'Time 1': t1, 'Placar': f"{ga} x {gb}", 'Time 2': t2, 'Vencedor': f"{ganhador}{penaltis}"})
                
                with jogos_container.container():
                    st.markdown(f"""
                    <div style="background: linear-gradient(145deg, #1a1a2e, #12121a); border: 1px solid #2a2a3a; border-radius: 12px; padding: 1rem; margin-bottom: 1rem;">
                        <div style="display: flex; justify-content: center; align-items: center; gap: 1rem;">
                            <div style="display: flex; align-items: center; gap: 8px;">
                                <img src="{get_bandeira_url(t1, bandeiras_dict)}" style="width: 40px; border-radius: 3px;">
                                <span style="color: white; font-size: 1.1rem;">{t1}</span>
                            </div>
                            <div style="font-family: 'Bebas Neue', sans-serif; font-size: 2.2rem; color: #00ff88; padding: 0 1rem;">{ga} x {gb}</div>
                            <div style="display: flex; align-items: center; gap: 8px;">
                                <span style="color: white; font-size: 1.1rem;">{t2}</span>
                                <img src="{get_bandeira_url(t2, bandeiras_dict)}" style="width: 40px; border-radius: 3px;">
                            </div>
                        </div>
                        <div style="text-align: center; margin-top: 0.5rem; color: #ffd700; font-size: 1rem;">
                            🏆 {ganhador}{penaltis}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    df_fase = pd.DataFrame(jogos_fase)
                    st.dataframe(df_fase, use_container_width = True, hide_index = True)
                
                with class_container.container():
                    st.markdown("### ✅ Classificados")
                    for v in vencedores:
                        st.success(f"🏆 {v[0]}")
                
                if delay > 0: time.sleep(delay * 2)
            
            st.session_state['live_classificados'] = vencedores
            
            proxima = {'oitavas': 'quartas', 'quartas': 'semis', 'semis': 'final', 'final': 'campeao'}
            
            if st.session_state['live_fase'] == 'final':
                st.session_state['live_campeao'] = vencedores[0][0]
                st.session_state['live_fase'] = 'campeao'
            else:
                st.session_state['live_fase'] = proxima[st.session_state['live_fase']]
            
            time.sleep(1)
            st.rerun()
        
        # ========== CAMPEÃO ==========
        elif st.session_state['live_fase'] == 'campeao':
            st.session_state['live_running'] = False
            
            # Salvar no histórico
            nova_copa = {
                'edicao': len(st.session_state['historico_copas']) + 1,
                'campeao': st.session_state['live_campeao'],
                'vice': st.session_state.get('live_vice', 'N/A'),
                'semifinalistas': st.session_state.get('live_semifinalistas', []),
                'final_placar': st.session_state.get('live_final_placar', 'N/A'),
                'timestamp': datetime.now().strftime('%d/%m/%Y %H:%M:%S')
            }
            st.session_state['historico_copas'].append(nova_copa)
            
            st.markdown("---")
            col1, col2, col3 = st.columns([1, 2, 1])
            campeao = st.session_state['live_campeao']
            vice = st.session_state.get('live_vice', 'N/A')
            with col2:
                st.markdown(f"""
                <div style="background: linear-gradient(145deg, #2a2a1e, #1a1a0f); border: 3px solid #ffd700; border-radius: 20px; padding: 3rem; text-align: center; box-shadow: 0 0 30px #ffd700;">
                    <div style="font-size: 5rem;">🏆</div>
                    <img src="{get_bandeira_url(campeao, bandeiras_dict)}" style="width: 120px; height: auto; border-radius: 8px; margin: 1rem 0; box-shadow: 0 4px 20px rgba(255, 215, 0, 0.5);">
                    <div style="font-family: 'Bebas Neue', sans-serif; font-size: 3.5rem; color: #ffd700; margin: 0.5rem 0;">{campeao}</div>
                    <div style="color: #ffd700; font-size: 1.5rem;">CAMPEÃO DA COPA DO MUNDO 2026!</div>
                    <div style="display: flex; justify-content: center; align-items: center; gap: 8px; margin-top: 1rem; color: #c0c0c0; font-size: 1rem;">
                        🥈 Vice: <img src="{get_bandeira_url(vice, bandeiras_dict)}" style="width: 24px; border-radius: 2px;"> {vice}
                    </div>
                    <div style="color: #8a8a9a; font-size: 0.9rem; margin-top: 0.5rem;">Final: {st.session_state.get('live_final_placar', 'N/A')}</div>
                </div>
                """, unsafe_allow_html = True)
            
            st.balloons()
    
    else:
        st.info("👆 Clique em **NOVA COPA** para iniciar uma simulação ao vivo!")
        
        if st.session_state.get('live_campeao'):
            col1, col2, col3 = st.columns([1, 2, 1])
            campeao = st.session_state['live_campeao']
            vice = st.session_state.get('live_vice', 'N/A')
            with col2:
                st.markdown(f"""
                <div style="background: linear-gradient(145deg, #2a2a1e, #1a1a0f); border: 3px solid #ffd700; border-radius: 20px; padding: 3rem; text-align: center; box-shadow: 0 0 30px #ffd700;">
                    <div style="font-size: 5rem;">🏆</div>
                    <img src="{get_bandeira_url(campeao, bandeiras_dict)}" style="width: 120px; height: auto; border-radius: 8px; margin: 1rem 0; box-shadow: 0 4px 20px rgba(255, 215, 0, 0.5);">
                    <div style="font-family: 'Bebas Neue', sans-serif; font-size: 3.5rem; color: #ffd700; margin: 0.5rem 0;">{campeao}</div>
                    <div style="color: #ffd700; font-size: 1.5rem;">CAMPEÃO DA COPA DO MUNDO 2026!</div>
                    <div style="display: flex; justify-content: center; align-items: center; gap: 8px; margin-top: 1rem; color: #c0c0c0; font-size: 1rem;">
                        🥈 Vice: <img src="{get_bandeira_url(vice, bandeiras_dict)}" style="width: 24px; border-radius: 2px;"> {vice}
                    </div>
                </div>
                """, unsafe_allow_html = True)
    
    # ========== HISTÓRICO DE COPAS ==========
    st.markdown("---")
    st.markdown("### 📜 Histórico de Copas Simuladas")
    
    if len(st.session_state.get('historico_copas', [])) == 0:
        st.markdown("""
        <div style="text-align: center; padding: 2rem; color: #8a8a9a;">
            <div style="font-size: 3rem; margin-bottom: 1rem;">🏟️</div>
            <div style="font-size: 1.1rem;">Nenhuma copa simulada ainda!</div>
            <div style="font-size: 0.9rem;">Clique em "NOVA COPA" para começar a construir seu histórico.</div>
        </div>
        """, unsafe_allow_html = True)
    else:
        historico = st.session_state['historico_copas']
        
        # Estatísticas gerais
        col_stats1, col_stats2, col_stats3, col_stats4 = st.columns(4)
        
        # Contar campeões e vices
        campeoes_count = {}
        vices_count = {}
        for copa in historico:
            campeoes_count[copa['campeao']] = campeoes_count.get(copa['campeao'], 0) + 1
            if copa['vice'] != 'N/A':
                vices_count[copa['vice']] = vices_count.get(copa['vice'], 0) + 1
        
        maior_campeao = max(campeoes_count.items(), key=lambda x: x[1]) if campeoes_count else ('N/A', 0)
        
        with col_stats1:
            st.markdown(f"""
            <div class="mini-stat">
                <div class="mini-stat-value">{len(historico)}</div>
                <div class="mini-stat-label">Copas Simuladas</div>
            </div>
            """, unsafe_allow_html = True)
        
        with col_stats2:
            st.markdown(f"""
            <div class="mini-stat">
                <div class="mini-stat-value">{len(campeoes_count)}</div>
                <div class="mini-stat-label">Campeões Únicos</div>
            </div>
            """, unsafe_allow_html = True)
        
        with col_stats3:
            st.markdown(f"""
            <div class="mini-stat">
                <div class="mini-stat-value" style="font-size: 1.3rem;">{maior_campeao[0]}</div>
                <div class="mini-stat-label">Maior Campeão</div>
            </div>
            """, unsafe_allow_html = True)
        
        with col_stats4:
            st.markdown(f"""
            <div class="mini-stat">
                <div class="mini-stat-value">{maior_campeao[1]}x</div>
                <div class="mini-stat-label">Títulos</div>
            </div>
            """, unsafe_allow_html = True)
        
        st.markdown("")
        
        # Tabs do histórico
        tab_lista, tab_hall, tab_grafico = st.tabs(["📋 Lista de Copas", "🏆 Hall da Fama", "📊 Estatísticas"])
        
        with tab_lista:
            # Lista de copas em ordem reversa (mais recente primeiro)
            for copa in reversed(historico):
                semifinalistas_str = ", ".join([s for s in copa['semifinalistas'] if s not in [copa['campeao'], copa['vice']]]) if copa['semifinalistas'] else "N/A"
                campeao_url = get_bandeira_url(copa['campeao'], bandeiras_dict)
                vice_url = get_bandeira_url(copa['vice'], bandeiras_dict)
                
                st.markdown(f"""
                <div class="history-card">
                    <div class="history-edition">
                        <span>🏆 Copa #{copa['edicao']}</span>
                        <span class="history-badge">{copa['timestamp']}</span>
                    </div>
                    <div class="history-champion" style="display: flex; align-items: center; gap: 10px;">
                        <img src="{campeao_url}" style="width: 36px; border-radius: 4px;">
                        🥇 {copa['campeao']}
                    </div>
                    <div class="history-details">
                        <span style="display: flex; align-items: center; gap: 6px;">
                            🥈 Vice: <img src="{vice_url}" style="width: 20px; border-radius: 2px;"> <b>{copa['vice']}</b>
                        </span>
                        <span>⚽ Final: <b>{copa['final_placar']}</b></span>
                    </div>
                    <div style="margin-top: 0.5rem; font-size: 0.8rem; color: #6a6a7a;">
                        🥉 Semifinalistas: {semifinalistas_str}
                    </div>
                </div>
                """, unsafe_allow_html = True)
        
        with tab_hall:
            st.markdown("#### 👑 Ranking de Campeões")
            
            # Ordenar campeões por número de títulos
            ranking_campeoes = sorted(campeoes_count.items(), key=lambda x: x[1], reverse=True)
            
            col_h1, col_h2 = st.columns(2)
            
            with col_h1:
                for i, (time_sel, titulos) in enumerate(ranking_campeoes[:5]):
                    medalha = "🥇" if i == 0 else ("🥈" if i == 1 else ("🥉" if i == 2 else "🏅"))
                    pct = (titulos / len(historico)) * 100
                    border_color = '#ffd700' if i == 0 else ('#c0c0c0' if i == 1 else ('#cd7f32' if i == 2 else '#3a3a4a'))
                    bandeira_url = get_bandeira_url(time_sel, bandeiras_dict)
                    st.markdown(f"""
                    <div class="hall-of-fame-card" style="border-color: {border_color};">
                        <div style="font-size: 2rem;">{medalha}</div>
                        <img src="{bandeira_url}" style="width: 60px; border-radius: 6px; margin: 0.5rem 0;">
                        <div class="hall-team">{time_sel}</div>
                        <div class="hall-count">{titulos}x</div>
                        <div style="color: #8a8a9a; font-size: 0.85rem;">{pct:.1f}% das copas</div>
                    </div>
                    """, unsafe_allow_html = True)
            
            with col_h2:
                if len(ranking_campeoes) > 5:
                    st.markdown("##### Outros Campeões")
                    for time_sel, titulos in ranking_campeoes[5:]:
                        pct = (titulos / len(historico)) * 100
                        bandeira_url = get_bandeira_url(time_sel, bandeiras_dict)
                        st.markdown(f"""
                        <div style="background: #1a1a2e; border-radius: 8px; padding: 0.8rem; margin: 0.3rem 0; display: flex; justify-content: space-between; align-items: center;">
                            <div style="display: flex; align-items: center; gap: 8px;">
                                <img src="{bandeira_url}" style="width: 24px; border-radius: 2px;">
                                <span style="color: white;">{time_sel}</span>
                            </div>
                            <span style="color: #00ff88; font-family: 'Bebas Neue', sans-serif; font-size: 1.3rem;">{titulos}x</span>
                        </div>
                        """, unsafe_allow_html = True)
                
                st.markdown("##### 🥈 Maiores Vices")
                ranking_vices = sorted(vices_count.items(), key=lambda x: x[1], reverse=True)[:5]
                for time_sel, vices in ranking_vices:
                    bandeira_url = get_bandeira_url(time_sel, bandeiras_dict)
                    st.markdown(f"""
                    <div style="background: #1a1a2e; border-radius: 8px; padding: 0.8rem; margin: 0.3rem 0; display: flex; justify-content: space-between; align-items: center;">
                        <div style="display: flex; align-items: center; gap: 8px;">
                            <img src="{bandeira_url}" style="width: 24px; border-radius: 2px;">
                            <span style="color: #c0c0c0;">{time_sel}</span>
                        </div>
                        <span style="color: #c0c0c0; font-family: 'Bebas Neue', sans-serif; font-size: 1.3rem;">{vices}x</span>
                    </div>
                    """, unsafe_allow_html = True)
        
        with tab_grafico:
            if len(historico) >= 2:
                # Gráfico de barras dos campeões
                df_campeoes = pd.DataFrame(list(campeoes_count.items()), columns=['Seleção', 'Títulos'])
                df_campeoes = df_campeoes.sort_values('Títulos', ascending=True)
                
                fig_hist = go.Figure()
                fig_hist.add_trace(go.Bar(
                    y=df_campeoes['Seleção'],
                    x=df_campeoes['Títulos'],
                    orientation='h',
                    marker=dict(
                        color=df_campeoes['Títulos'],
                        colorscale=[[0, '#1a1a2e'], [0.5, '#00ccff'], [1, '#ffd700']],
                    ),
                    text=df_campeoes['Títulos'],
                    textposition='outside',
                    textfont=dict(color='#ffd700', size=14)
                ))
                
                fig_hist.update_layout(
                    title="🏆 Distribuição de Títulos",
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='#8a8a9a',
                    title_font_color='#00ff88',
                    height=max(300, len(df_campeoes) * 50),
                    xaxis_title="Número de Títulos",
                    yaxis_title="",
                    margin=dict(l=120, r=60)
                )
                st.plotly_chart(fig_hist, use_container_width=True)
                
                # Evolução temporal (linha do tempo dos campeões)
                st.markdown("#### 📈 Linha do Tempo dos Campeões")
                timeline_data = [(copa['edicao'], copa['campeao']) for copa in historico]
                df_timeline = pd.DataFrame(timeline_data, columns=['Edição', 'Campeão'])
                
                fig_timeline = px.scatter(
                    df_timeline, 
                    x='Edição', 
                    y='Campeão',
                    color='Campeão',
                    size=[40]*len(df_timeline),
                    color_discrete_sequence=px.colors.qualitative.Set2
                )
                
                fig_timeline.update_traces(marker=dict(symbol='star', line=dict(width=1, color='white')))
                fig_timeline.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='#8a8a9a',
                    height=400,
                    showlegend=False,
                    xaxis=dict(tickmode='linear', tick0=1, dtick=1)
                )
                st.plotly_chart(fig_timeline, use_container_width=True)
                
                # Gráfico de pizza dos campeões
                col_pie1, col_pie2 = st.columns(2)
                
                with col_pie1:
                    fig_pie_champ = go.Figure(go.Pie(
                        labels=list(campeoes_count.keys()),
                        values=list(campeoes_count.values()),
                        hole=0.4,
                        marker=dict(colors=px.colors.qualitative.Set2),
                        textinfo='label+percent',
                        textfont=dict(size=11)
                    ))
                    fig_pie_champ.update_layout(
                        title="🥇 Distribuição de Títulos",
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font_color='#8a8a9a',
                        title_font_color='#ffd700',
                        height=350,
                        showlegend=False
                    )
                    st.plotly_chart(fig_pie_champ, use_container_width=True)
                
                with col_pie2:
                    if vices_count:
                        fig_pie_vice = go.Figure(go.Pie(
                            labels=list(vices_count.keys()),
                            values=list(vices_count.values()),
                            hole=0.4,
                            marker=dict(colors=px.colors.qualitative.Pastel),
                            textinfo='label+percent',
                            textfont=dict(size=11)
                        ))
                        fig_pie_vice.update_layout(
                            title="🥈 Distribuição de Vice-Campeões",
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            font_color='#8a8a9a',
                            title_font_color='#c0c0c0',
                            height=350,
                            showlegend=False
                        )
                        st.plotly_chart(fig_pie_vice, use_container_width=True)
            else:
                st.info("📊 Simule pelo menos 2 copas para ver os gráficos estatísticos!")
        
        # Botão para limpar histórico
        st.markdown("---")
        col_clear1, col_clear2, col_clear3 = st.columns([1, 1, 1])
        with col_clear2:
            if st.button("🗑️ Limpar Histórico", use_container_width=True):
                st.session_state['historico_copas'] = []
                st.rerun()

# ============ TAB 5: PROBABILIDADES IMPLÍCITAS ============
with tab5:
    st.markdown("### 💰 Probabilidades Implícitas das Casas de Apostas")
    st.markdown("""
    <p style="color: #8a8a9a; font-size: 1rem; margin-bottom: 1.5rem;">
        As probabilidades implícitas são calculadas a partir das odds atuais oferecidas pelas casas de apostas.<br>
        Elas representam a <b style="color: #00ff88;">percepção do mercado</b> sobre as chances de cada seleção ser campeã.
    </p>
    """, unsafe_allow_html = True)
    
    # Carregar dados das probabilidades implícitas
    try:
        df_odds = pd.read_csv('probabilidades implicitas.csv', decimal = ',')
        df_odds.columns = ['Seleção', 'Prob_Implicita']
        df_odds['Prob_Implicita'] = pd.to_numeric(df_odds['Prob_Implicita'], errors = 'coerce')
        df_odds['Prob_Percentual'] = df_odds['Prob_Implicita'] * 100
        df_odds['Odds_Decimal'] = 1 / df_odds['Prob_Implicita']
        df_odds = df_odds.sort_values('Prob_Implicita', ascending = False).reset_index(drop = True)
        df_odds['Posição'] = df_odds.index + 1
        
        # Métricas principais
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-value">{len(df_odds)}</div>
                <div class="stat-label">Seleções Cotadas</div>
            </div>
            """, unsafe_allow_html = True)
        
        with col2:
            favorito = df_odds.iloc[0]
            fav_bandeira = get_bandeira_url(favorito['Seleção'], bandeiras_dict)
            st.markdown(f"""
            <div class="stat-card">
                <img src="{fav_bandeira}" style="width: 48px; border-radius: 4px; margin-bottom: 0.5rem;">
                <div class="stat-value" style="font-size: 2rem;">{favorito['Seleção']}</div>
                <div class="stat-label">Favorito das Casas</div>
            </div>
            """, unsafe_allow_html = True)
        
        with col3:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-value">{favorito['Prob_Percentual']:.1f}%</div>
                <div class="stat-label">Prob. do Favorito</div>
            </div>
            """, unsafe_allow_html = True)
        
        with col4:
            soma_probs = df_odds['Prob_Implicita'].sum()
            margem = (soma_probs - 1) * 100
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-value">{margem:.1f}%</div>
                <div class="stat-label">Margem da Casa</div>
            </div>
            """, unsafe_allow_html = True)
        
        st.markdown("---")
        
        # Top 3 Favoritos
        st.markdown("### 🏆 Top 3 Favoritos do Mercado")
        
        col1, col2, col3 = st.columns(3)
        
        top3_odds = df_odds.head(3)
        
        with col1:
            sel1 = top3_odds.iloc[0]['Seleção']
            st.markdown(f"""
            <div class="team-card gold">
                <div class="team-rank gold">🥇</div>
                <div class="team-name">{get_bandeira_html(sel1, bandeiras_dict, 32)}{sel1}</div>
                <div class="team-prob">{top3_odds.iloc[0]['Prob_Percentual']:.1f}%</div>
            </div>
            <div style="text-align: center; color: #8a8a9a; font-size: 0.9rem; margin-top: 0.5rem;">
                Odds: {top3_odds.iloc[0]['Odds_Decimal']:.2f}
            </div>
            """, unsafe_allow_html = True)
        
        with col2:
            sel2 = top3_odds.iloc[1]['Seleção']
            st.markdown(f"""
            <div class="team-card silver">
                <div class="team-rank silver">🥈</div>
                <div class="team-name">{get_bandeira_html(sel2, bandeiras_dict, 32)}{sel2}</div>
                <div class="team-prob">{top3_odds.iloc[1]['Prob_Percentual']:.1f}%</div>
            </div>
            <div style="text-align: center; color: #8a8a9a; font-size: 0.9rem; margin-top: 0.5rem;">
                Odds: {top3_odds.iloc[1]['Odds_Decimal']:.2f}
            </div>
            """, unsafe_allow_html = True)
        
        with col3:
            sel3 = top3_odds.iloc[2]['Seleção']
            st.markdown(f"""
            <div class="team-card bronze">
                <div class="team-rank bronze">🥉</div>
                <div class="team-name">{get_bandeira_html(sel3, bandeiras_dict, 32)}{sel3}</div>
                <div class="team-prob">{top3_odds.iloc[2]['Prob_Percentual']:.1f}%</div>
            </div>
            <div style="text-align: center; color: #8a8a9a; font-size: 0.9rem; margin-top: 0.5rem;">
                Odds: {top3_odds.iloc[2]['Odds_Decimal']:.2f}
            </div>
            """, unsafe_allow_html = True)
        
        st.markdown("---")
        
        # Gráficos lado a lado
        col_graf1, col_graf2 = st.columns(2)
        
        with col_graf1:
            st.markdown("### 📊 Top 15 - Probabilidades Implícitas")
            
            top15_odds = df_odds.head(15).iloc[::-1]
            
            fig_odds_bar = go.Figure()
            fig_odds_bar.add_trace(go.Bar(
                y = top15_odds['Seleção'],
                x = top15_odds['Prob_Percentual'],
                orientation = 'h',
                marker = dict(
                    color = top15_odds['Prob_Percentual'],
                    colorscale = [[0, '#ff6b6b'], [0.3, '#ffd700'], [1, '#00ff88']],
                    line = dict(width = 0)
                ),
                text = [f"{v:.1f}%" for v in top15_odds['Prob_Percentual']],
                textposition = 'outside',
                textfont = dict(color = '#00ff88', size = 12)
            ))
            
            fig_odds_bar.update_layout(
                plot_bgcolor = 'rgba(0,0,0,0)',
                paper_bgcolor = 'rgba(0,0,0,0)',
                font_color = '#8a8a9a',
                height = 500,
                xaxis_title = "Probabilidade Implícita (%)",
                yaxis_title = "",
                margin = dict(l = 100, r = 80)
            )
            st.plotly_chart(fig_odds_bar, use_container_width = True)
        
        with col_graf2:
            st.markdown("### 🎯 Distribuição de Probabilidades")
            
            # Criar categorias de probabilidade
            def categorizar(prob):
                if prob >= 10: return "Alta (≥10%)"
                elif prob >= 5: return "Média-Alta (5-10%)"
                elif prob >= 1: return "Média (1-5%)"
                elif prob >= 0.5: return "Baixa (0.5-1%)"
                else: return "Muito Baixa (<0.5%)"
            
            df_odds['Categoria'] = df_odds['Prob_Percentual'].apply(categorizar)
            cat_counts = df_odds['Categoria'].value_counts().reindex([
                "Alta (≥10%)", "Média-Alta (5-10%)", "Média (1-5%)", "Baixa (0.5-1%)", "Muito Baixa (<0.5%)"
            ]).fillna(0)
            
            fig_pie = go.Figure(go.Pie(
                labels = cat_counts.index,
                values = cat_counts.values,
                hole = 0.5,
                marker = dict(colors = ['#00ff88', '#00ccff', '#ffd700', '#ff9f43', '#ff6b6b']),
                textinfo = 'label+value',
                textfont = dict(size = 11)
            ))
            
            fig_pie.update_layout(
                plot_bgcolor = 'rgba(0,0,0,0)',
                paper_bgcolor = 'rgba(0,0,0,0)',
                font_color = '#8a8a9a',
                height = 500,
                showlegend = False,
                annotations = [dict(text = 'Seleções', x = 0.5, y = 0.5, font_size = 16, showarrow = False, font_color = '#00ff88')]
            )
            st.plotly_chart(fig_pie, use_container_width = True)
        
        st.markdown("---")
        
        # Treemap visual
        st.markdown("### 🗺️ Mapa de Probabilidades")
        
        top20 = df_odds.head(20)
        fig_treemap = px.treemap(
            top20,
            path = ['Seleção'],
            values = 'Prob_Percentual',
            color = 'Prob_Percentual',
            color_continuous_scale = [[0, '#1a1a2e'], [0.3, '#00ccff'], [0.6, '#00ff88'], [1, '#ffd700']],
            hover_data = {'Odds_Decimal': ':.2f'}
        )
        
        fig_treemap.update_layout(
            plot_bgcolor = 'rgba(0,0,0,0)',
            paper_bgcolor = 'rgba(0,0,0,0)',
            font_color = 'white',
            height = 450,
            margin = dict(t = 30, l = 10, r = 10, b = 10)
        )
        fig_treemap.update_traces(textinfo = "label+value", texttemplate = "%{label}<br>%{value:.1f}%")
        st.plotly_chart(fig_treemap, use_container_width = True)
        
        st.markdown("---")
        
        # Tabela completa
        st.markdown("### 📋 Tabela Completa de Probabilidades")
        
        df_display_odds = df_odds[['Posição', 'Seleção', 'Prob_Percentual', 'Odds_Decimal']].copy()
        df_display_odds.columns = ['#', 'Seleção', 'Probabilidade (%)', 'Odds Decimal']
        df_display_odds['Probabilidade (%)'] = df_display_odds['Probabilidade (%)'].apply(lambda x: f"{x:.2f}%")
        df_display_odds['Odds Decimal'] = df_display_odds['Odds Decimal'].apply(lambda x: f"{x:.2f}")
        
        st.dataframe(df_display_odds, use_container_width = True, height = 400, hide_index = True)
        
        # Comparação com simulação (se existir)
        if 'resultado' in st.session_state:
            st.markdown("---")
            st.markdown("### 🔄 Comparação: Mercado vs Simulação Monte Carlo")
            
            df_sim = st.session_state['resultado'].copy()
            df_sim['Seleção'] = df_sim.index
            df_sim['Prob_Simulacao'] = df_sim['Campeão'] * 100
            
            # Merge com odds
            df_compare = df_odds.merge(df_sim[['Seleção', 'Prob_Simulacao']], on = 'Seleção', how = 'inner')
            df_compare['Diferença'] = df_compare['Prob_Simulacao'] - df_compare['Prob_Percentual']
            df_compare = df_compare.sort_values('Prob_Percentual', ascending = False).head(15)
            
            fig_compare = go.Figure()
            
            fig_compare.add_trace(go.Bar(
                name = 'Odds Implícitas',
                x = df_compare['Seleção'],
                y = df_compare['Prob_Percentual'],
                marker_color = '#ff6b6b',
                text = [f"{v:.1f}%" for v in df_compare['Prob_Percentual']],
                textposition = 'outside'
            ))
            
            fig_compare.add_trace(go.Bar(
                name = 'Simulação Monte Carlo',
                x = df_compare['Seleção'],
                y = df_compare['Prob_Simulacao'],
                marker_color = '#00ff88',
                text = [f"{v:.1f}%" for v in df_compare['Prob_Simulacao']],
                textposition = 'outside'
            ))
            
            fig_compare.update_layout(
                barmode = 'group',
                plot_bgcolor = 'rgba(0,0,0,0)',
                paper_bgcolor = 'rgba(0,0,0,0)',
                font_color = '#8a8a9a',
                height = 450,
                xaxis_title = "",
                yaxis_title = "Probabilidade (%)",
                legend = dict(orientation = "h", yanchor = "bottom", y = 1.02, xanchor = "right", x = 1)
            )
            st.plotly_chart(fig_compare, use_container_width = True)
            
            # Value bets
            st.markdown("### 💎 Value Bets (Simulação > Mercado)")
            st.markdown('<p style="color: #8a8a9a;">Seleções onde nossa simulação dá mais chances do que o mercado.</p>', unsafe_allow_html = True)
            
            value_bets = df_compare[df_compare['Diferença'] > 0.5].sort_values('Diferença', ascending = False)
            
            if len(value_bets) > 0:
                for _, row in value_bets.iterrows():
                    sel_name = row['Seleção']
                    st.markdown(f"""
                    <div class="team-card" style="border-left-color: #00ff88;">
                        <div class="team-name">{get_bandeira_html(sel_name, bandeiras_dict, 28)}{sel_name}</div>
                        <div style="color: #00ff88; font-size: 1.2rem;">
                            +{row['Diferença']:.1f}% acima do mercado
                        </div>
                        <div style="color: #8a8a9a; font-size: 0.9rem;">
                            Mercado: {row['Prob_Percentual']:.1f}% | Simulação: {row['Prob_Simulacao']:.1f}%
                        </div>
                    </div>
                    """, unsafe_allow_html = True)
            else:
                st.info("Nenhum value bet encontrado com diferença significativa.")
        
    except FileNotFoundError:
        st.error("❌ Arquivo `probabilidades implicitas.csv` não encontrado!")
    except Exception as e:
        st.error(f"❌ Erro ao carregar dados: {str(e)}")

# ============ TAB 6: SIMULADOR DE PARTIDA ============
with tab6:
    st.markdown("### ⚔️ Simulador de Partida Individual")
    st.markdown("""
    <p style="color: #8a8a9a; font-size: 1rem; margin-bottom: 1.5rem;">
        Selecione duas equipes para simular um confronto direto e ver as probabilidades de cada resultado usando a <b style="color: #00ff88;">distribuição de Poisson</b>.
    </p>
    """, unsafe_allow_html = True)
    
    # Função para calcular probabilidades de Poisson
    def calcular_probabilidades_partida(forca_time1, forca_time2, media_gols=MEDIA_GOLS_COPA, diff_forca=DIFF_FORCA, max_gols=8):
        """Calcula as probabilidades de cada placar usando distribuição de Poisson"""
        diff = (forca_time1 - forca_time2) / diff_forca
        lambda_1 = max(0.1, (media_gols / 2) * (1 + diff))
        lambda_2 = max(0.1, (media_gols / 2) * (1 - diff))
        
        # Matriz de probabilidades de placares
        prob_matrix = np.zeros((max_gols + 1, max_gols + 1))
        
        for gols1 in range(max_gols + 1):
            for gols2 in range(max_gols + 1):
                prob_matrix[gols1, gols2] = poisson.pmf(gols1, lambda_1) * poisson.pmf(gols2, lambda_2)
        
        # Probabilidades de resultado
        prob_vitoria_1 = 0
        prob_empate = 0
        prob_vitoria_2 = 0
        
        for gols1 in range(max_gols + 1):
            for gols2 in range(max_gols + 1):
                if gols1 > gols2:
                    prob_vitoria_1 += prob_matrix[gols1, gols2]
                elif gols1 == gols2:
                    prob_empate += prob_matrix[gols1, gols2]
                else:
                    prob_vitoria_2 += prob_matrix[gols1, gols2]
        
        return prob_matrix, lambda_1, lambda_2, prob_vitoria_1, prob_empate, prob_vitoria_2
    
    # Seleção das equipes
    col_sel1, col_vs, col_sel2 = st.columns([2, 1, 2])
    
    selecoes_ordenadas = sorted(selecoes)
    
    with col_sel1:
        st.markdown("#### 🏠 Time da Casa")
        time1 = st.selectbox("Selecione o time 1", selecoes_ordenadas, index=selecoes_ordenadas.index("Brasil") if "Brasil" in selecoes_ordenadas else 0, key="time1_select")
        if time1:
            st.markdown(f"""
            <div style="text-align: center; padding: 1rem;">
                <img src="{get_bandeira_url(time1, bandeiras_dict)}" style="width: 100px; border-radius: 8px; box-shadow: 0 4px 15px rgba(0,255,136,0.3);">
                <div style="font-family: 'Bebas Neue', sans-serif; font-size: 1.8rem; color: white; margin-top: 0.5rem;">{time1}</div>
                <div style="color: #00ff88; font-size: 1.1rem;">Força: {forca_dict.get(time1, 0):.0f}</div>
            </div>
            """, unsafe_allow_html=True)
    
    with col_vs:
        st.markdown("<div style='text-align: center; padding-top: 4rem;'><span style='font-family: Bebas Neue, sans-serif; font-size: 3rem; color: #ff00ff;'>VS</span></div>", unsafe_allow_html=True)
    
    with col_sel2:
        st.markdown("#### 🏃 Time Visitante")
        time2 = st.selectbox("Selecione o time 2", selecoes_ordenadas, index=selecoes_ordenadas.index("Argentina") if "Argentina" in selecoes_ordenadas else 1, key="time2_select")
        if time2:
            st.markdown(f"""
            <div style="text-align: center; padding: 1rem;">
                <img src="{get_bandeira_url(time2, bandeiras_dict)}" style="width: 100px; border-radius: 8px; box-shadow: 0 4px 15px rgba(0,204,255,0.3);">
                <div style="font-family: 'Bebas Neue', sans-serif; font-size: 1.8rem; color: white; margin-top: 0.5rem;">{time2}</div>
                <div style="color: #00ccff; font-size: 1.1rem;">Força: {forca_dict.get(time2, 0):.0f}</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    if time1 and time2 and time1 != time2:
        # Calcular probabilidades
        forca1 = forca_dict.get(time1, FORCA_PADRAO)
        forca2 = forca_dict.get(time2, FORCA_PADRAO)
        
        prob_matrix, lambda_1, lambda_2, prob_v1, prob_emp, prob_v2 = calcular_probabilidades_partida(forca1, forca2)
        
        # Métricas de expectativa de gols
        st.markdown("### 📊 Análise do Confronto")
        
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        
        with col_m1:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-value" style="color: #00ff88;">{lambda_1:.2f}</div>
                <div class="stat-label">Gols Esperados {time1}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_m2:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-value" style="color: #00ccff;">{lambda_2:.2f}</div>
                <div class="stat-label">Gols Esperados {time2}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_m3:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-value" style="color: #ffd700;">{lambda_1 + lambda_2:.2f}</div>
                <div class="stat-label">Total de Gols Esperado</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_m4:
            diff_forca_times = forca1 - forca2
            cor_diff = '#00ff88' if diff_forca_times > 0 else ('#ff6b6b' if diff_forca_times < 0 else '#ffd700')
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-value" style="color: {cor_diff};">{diff_forca_times:+.0f}</div>
                <div class="stat-label">Diferença de Força</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Probabilidades de Resultado
        st.markdown("### 🎯 Probabilidades de Resultado")
        
        col_prob1, col_prob2, col_prob3 = st.columns(3)
        
        with col_prob1:
            cor_borda1 = '#00ff88' if prob_v1 > prob_v2 and prob_v1 > prob_emp else '#2a2a3a'
            st.markdown(f"""
            <div style="background: linear-gradient(145deg, #1a1a2e, #12121a); border: 3px solid {cor_borda1}; border-radius: 16px; padding: 2rem; text-align: center;">
                <img src="{get_bandeira_url(time1, bandeiras_dict)}" style="width: 60px; border-radius: 6px;">
                <div style="font-family: 'Bebas Neue', sans-serif; font-size: 3rem; color: #00ff88; margin: 0.5rem 0;">{prob_v1*100:.1f}%</div>
                <div style="color: #8a8a9a; font-size: 1rem;">Vitória {time1}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_prob2:
            cor_borda_emp = '#ffd700' if prob_emp > prob_v1 and prob_emp > prob_v2 else '#2a2a3a'
            st.markdown(f"""
            <div style="background: linear-gradient(145deg, #1a1a2e, #12121a); border: 3px solid {cor_borda_emp}; border-radius: 16px; padding: 2rem; text-align: center;">
                <div style="font-size: 2.5rem;">🤝</div>
                <div style="font-family: 'Bebas Neue', sans-serif; font-size: 3rem; color: #ffd700; margin: 0.5rem 0;">{prob_emp*100:.1f}%</div>
                <div style="color: #8a8a9a; font-size: 1rem;">Empate</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_prob3:
            cor_borda2 = '#00ccff' if prob_v2 > prob_v1 and prob_v2 > prob_emp else '#2a2a3a'
            st.markdown(f"""
            <div style="background: linear-gradient(145deg, #1a1a2e, #12121a); border: 3px solid {cor_borda2}; border-radius: 16px; padding: 2rem; text-align: center;">
                <img src="{get_bandeira_url(time2, bandeiras_dict)}" style="width: 60px; border-radius: 6px;">
                <div style="font-family: 'Bebas Neue', sans-serif; font-size: 3rem; color: #00ccff; margin: 0.5rem 0;">{prob_v2*100:.1f}%</div>
                <div style="color: #8a8a9a; font-size: 1rem;">Vitória {time2}</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Barra de probabilidade visual
        st.markdown("")
        st.markdown(f"""
        <div style="background: #1a1a2e; border-radius: 20px; height: 40px; display: flex; overflow: hidden; margin: 1rem 0;">
            <div style="width: {prob_v1*100}%; background: linear-gradient(90deg, #00ff88, #00cc6a); display: flex; align-items: center; justify-content: center; color: #0a0a0f; font-weight: bold; font-size: 0.9rem;">
                {prob_v1*100:.0f}%
            </div>
            <div style="width: {prob_emp*100}%; background: linear-gradient(90deg, #ffd700, #ffaa00); display: flex; align-items: center; justify-content: center; color: #0a0a0f; font-weight: bold; font-size: 0.9rem;">
                {prob_emp*100:.0f}%
            </div>
            <div style="width: {prob_v2*100}%; background: linear-gradient(90deg, #00ccff, #0099cc); display: flex; align-items: center; justify-content: center; color: #0a0a0f; font-weight: bold; font-size: 0.9rem;">
                {prob_v2*100:.0f}%
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Heatmap de Placares + Top 5 Placares lado a lado
        st.markdown("### 🔥 Heatmap de Probabilidade de Placares")
        
        # Criar lista de placares com probabilidades (precisa antes para mostrar no top 5)
        max_gols_display = 6  # Mostrar até 6x6 para ficar mais legível
        prob_display = prob_matrix[:max_gols_display+1, :max_gols_display+1] * 100
        
        placares = []
        for i in range(max_gols_display+1):
            for j in range(max_gols_display+1):
                resultado = "Vitória " + time1 if i > j else ("Empate" if i == j else "Vitória " + time2)
                cor = "#00ff88" if i > j else ("#ffd700" if i == j else "#00ccff")
                placares.append({
                    'placar': f"{i} x {j}",
                    'prob': prob_matrix[i, j] * 100,
                    'resultado': resultado,
                    'cor': cor,
                    'gols1': i,
                    'gols2': j
                })
        
        placares_ordenados = sorted(placares, key=lambda x: x['prob'], reverse=True)
        
        col_heatmap, col_top5 = st.columns([3, 2])
        
        with col_heatmap:
            st.markdown(f"""
            <p style="color: #8a8a9a; font-size: 0.9rem;">
                Linhas = gols de <b style="color: #00ff88;">{time1}</b> | Colunas = gols de <b style="color: #00ccff;">{time2}</b>
            </p>
            """, unsafe_allow_html=True)
            
            # Criar texto para anotações
            annotations_text = [[f"{prob_display[i,j]:.1f}%" for j in range(max_gols_display+1)] for i in range(max_gols_display+1)]
            
            fig_heatmap = go.Figure(data=go.Heatmap(
                z=prob_display,
                x=[str(i) for i in range(max_gols_display+1)],
                y=[str(i) for i in range(max_gols_display+1)],
                colorscale=[
                    [0, '#0a0a14'],
                    [0.2, '#0d1a33'],
                    [0.4, '#1a3a66'],
                    [0.6, '#2a5a99'],
                    [0.8, '#3a7acc'],
                    [1, '#4a9aff']
                ],
                text=annotations_text,
                texttemplate="%{text}",
                textfont={"size": 16, "color": "white"},
                hovertemplate=f"{time1}: %{{y}} x %{{x}} :{time2}<br>Probabilidade: %{{z:.2f}}%<extra></extra>",
                showscale=True,
                colorbar=dict(
                    title="Prob (%)",
                    titlefont=dict(color='#8a8a9a'),
                    tickfont=dict(color='#8a8a9a')
                )
            ))
            
            fig_heatmap.update_layout(
                xaxis=dict(
                    title=f"Gols {time2}",
                    titlefont=dict(color='#00ccff', size=14),
                    tickfont=dict(color='#8a8a9a', size=13),
                    side='bottom'
                ),
                yaxis=dict(
                    title=f"Gols {time1}",
                    titlefont=dict(color='#00ff88', size=14),
                    tickfont=dict(color='#8a8a9a', size=13)
                ),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#8a8a9a',
                height=675,
                margin=dict(l=60, r=20, t=30, b=60)
            )
            
            st.plotly_chart(fig_heatmap, use_container_width=True)
        
        with col_top5:
            st.markdown("#### 🏆 Top 5 Placares Mais Prováveis")
            st.markdown("")
            
            for idx, placar in enumerate(placares_ordenados[:5]):
                medalha = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"][idx]
                st.markdown(f"""
                <div style="background: linear-gradient(145deg, #1e1e2e, #16161f); border-left: 4px solid {placar['cor']}; padding: 1rem 1.2rem; margin: 0.6rem 0; border-radius: 0 12px 12px 0;">
                    <div style="display: flex; align-items: center; justify-content: space-between;">
                        <div style="display: flex; align-items: center; gap: 12px;">
                            <span style="font-size: 1.5rem;">{medalha}</span>
                            <div>
                                <div style="font-family: 'Bebas Neue', sans-serif; font-size: 1.8rem; color: white;">{placar['placar']}</div>
                                <div style="color: {placar['cor']}; font-size: 0.85rem;">{placar['resultado']}</div>
                            </div>
                        </div>
                        <div style="text-align: right;">
                            <div style="font-family: 'Bebas Neue', sans-serif; font-size: 2rem; color: {placar['cor']};">{placar['prob']:.1f}%</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Probabilidades de Over/Under
        st.markdown("### 📈 Probabilidades Over/Under (Total de Gols)")
        
        # Calcular Over/Under
        over_under = {}
        for linha in [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]:
            prob_over = 0
            for i in range(8):
                for j in range(8):
                    if i + j > linha:
                        prob_over += prob_matrix[i, j]
            over_under[linha] = prob_over
        
        col_ou1, col_ou2, col_ou3 = st.columns(3)
        
        linhas_ou = list(over_under.items())
        
        for idx, (linha, prob_over) in enumerate(linhas_ou):
            col = [col_ou1, col_ou2, col_ou3][idx % 3]
            with col:
                prob_under = 1 - prob_over
                st.markdown(f"""
                <div style="background: linear-gradient(145deg, #1a1a2e, #12121a); border: 1px solid #2a2a3a; border-radius: 12px; padding: 1rem; margin: 0.3rem 0; text-align: center;">
                    <div style="color: #8a8a9a; font-size: 0.9rem; margin-bottom: 0.5rem;">Linha {linha}</div>
                    <div style="display: flex; justify-content: space-around;">
                        <div>
                            <div style="font-family: 'Bebas Neue', sans-serif; font-size: 1.5rem; color: #00ff88;">Under</div>
                            <div style="font-size: 1.2rem; color: white;">{prob_under*100:.1f}%</div>
                        </div>
                        <div>
                            <div style="font-family: 'Bebas Neue', sans-serif; font-size: 1.5rem; color: #ff6b6b;">Over</div>
                            <div style="font-size: 1.2rem; color: white;">{prob_over*100:.1f}%</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Ambas Marcam
        st.markdown("### ⚽ Ambas Equipes Marcam (BTTS)")
        
        prob_btts_sim = 0  # Ambas marcam
        prob_btts_nao = 0  # Pelo menos uma não marca
        
        for i in range(8):
            for j in range(8):
                if i > 0 and j > 0:
                    prob_btts_sim += prob_matrix[i, j]
                else:
                    prob_btts_nao += prob_matrix[i, j]
        
        col_btts1, col_btts2 = st.columns(2)
        
        with col_btts1:
            st.markdown(f"""
            <div style="background: linear-gradient(145deg, #1a1a2e, #12121a); border: 2px solid #00ff88; border-radius: 16px; padding: 1.5rem; text-align: center;">
                <div style="font-size: 2rem;">✅</div>
                <div style="font-family: 'Bebas Neue', sans-serif; font-size: 2.5rem; color: #00ff88;">{prob_btts_sim*100:.1f}%</div>
                <div style="color: #8a8a9a; font-size: 1rem;">Ambas Marcam - SIM</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_btts2:
            st.markdown(f"""
            <div style="background: linear-gradient(145deg, #1a1a2e, #12121a); border: 2px solid #ff6b6b; border-radius: 16px; padding: 1.5rem; text-align: center;">
                <div style="font-size: 2rem;">❌</div>
                <div style="font-family: 'Bebas Neue', sans-serif; font-size: 2.5rem; color: #ff6b6b;">{prob_btts_nao*100:.1f}%</div>
                <div style="color: #8a8a9a; font-size: 1rem;">Ambas Marcam - NÃO</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Simulação Monte Carlo Rápida
        st.markdown("### 🎲 Simulação Monte Carlo (1000 jogos)")
        
        if st.button("🚀 Simular 1000 Partidas", use_container_width=True, type="primary"):
            resultados_mc = {'v1': 0, 'emp': 0, 'v2': 0, 'gols1': [], 'gols2': [], 'placares': {}}
            
            progress = st.progress(0)
            
            for i in range(1000):
                *_, ga, gb, _, _, resultado = simular_jogo(forca1, forca2)
                
                if resultado == 0 or ga > gb:
                    resultados_mc['v1'] += 1
                elif resultado == 1 or gb > ga:
                    resultados_mc['v2'] += 1
                else:
                    resultados_mc['emp'] += 1
                
                resultados_mc['gols1'].append(ga)
                resultados_mc['gols2'].append(gb)
                
                placar_key = f"{ga}x{gb}"
                resultados_mc['placares'][placar_key] = resultados_mc['placares'].get(placar_key, 0) + 1
                
                if (i + 1) % 100 == 0:
                    progress.progress((i + 1) / 1000)
            
            progress.progress(1.0)
            
            st.success("✅ Simulação concluída!")
            
            col_res1, col_res2, col_res3 = st.columns(3)
            
            with col_res1:
                st.metric(f"🏆 Vitórias {time1}", f"{resultados_mc['v1']/10:.1f}%", f"Média gols: {np.mean(resultados_mc['gols1']):.2f}")
            
            with col_res2:
                st.metric("🤝 Empates", f"{resultados_mc['emp']/10:.1f}%", f"Total médio: {np.mean(resultados_mc['gols1']) + np.mean(resultados_mc['gols2']):.2f}")
            
            with col_res3:
                st.metric(f"🏆 Vitórias {time2}", f"{resultados_mc['v2']/10:.1f}%", f"Média gols: {np.mean(resultados_mc['gols2']):.2f}")
            
            # Top 5 placares mais frequentes na simulação
            st.markdown("#### 📊 Placares mais frequentes na simulação:")
            top_placares_mc = sorted(resultados_mc['placares'].items(), key=lambda x: x[1], reverse=True)[:5]
            
            for placar, freq in top_placares_mc:
                st.markdown(f"""
                <div style="background: #1a1a2e; border-radius: 8px; padding: 0.5rem 1rem; margin: 0.2rem 0; display: flex; justify-content: space-between;">
                    <span style="color: white; font-size: 1.1rem;">{placar.replace('x', ' x ')}</span>
                    <span style="color: #00ff88; font-family: 'Bebas Neue', sans-serif; font-size: 1.2rem;">{freq/10:.1f}%</span>
                </div>
                """, unsafe_allow_html=True)
    
    elif time1 == time2:
        st.warning("⚠️ Por favor, selecione duas equipes diferentes para simular o confronto.")
    else:
        st.info("👆 Selecione as duas equipes acima para ver a análise do confronto.")

# ============ FOOTER ============
st.markdown("---")
st.markdown("""
<p style="text-align: center; color: #4a4a5a; font-size: 0.85rem;">
    ⚽ Copa 2026 Simulator • Monte Carlo Simulation Engine • Built with Streamlit
</p>
""", unsafe_allow_html = True)

