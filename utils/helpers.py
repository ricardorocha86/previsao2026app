# ============ FUNÇÕES AUXILIARES ============
import streamlit as st
import numpy as np

def get_brasil_style(selecao, cor_padrao='#209927'):
    """Retorna estilos especiais se a seleção for Brasil"""
    if selecao == 'Brasil':
        return {
            'borda': '#009c3b',
            'bg': 'linear-gradient(145deg, #009c3b22, #ffd70015)',
            'glow': 'box-shadow: 0 0 15px rgba(0, 156, 59, 0.4), 0 0 25px rgba(255, 215, 0, 0.2);',
            'badge': '🇧🇷',
            'extra_class': 'brasil-glow',
            'is_brasil': True
        }
    return {
        'borda': cor_padrao,
        'bg': 'linear-gradient(145deg, rgba(255,255,255,0.045), rgba(104,231,15,0.035))',
        'glow': '',
        'badge': '',
        'extra_class': '',
        'is_brasil': False
    }

def get_bandeira_html(selecao, bandeiras_dict, tamanho=24):
    """Retorna HTML da bandeira de uma seleção"""
    url = bandeiras_dict.get(selecao, 'https://flagcdn.com/w320/un.png')
    return f'<img src="{url}" style="width: {tamanho}px; height: auto; border-radius: 3px; vertical-align: middle; margin-right: 8px;">'

def get_bandeira_url(selecao, bandeiras_dict):
    """Retorna URL da bandeira de uma seleção"""
    return bandeiras_dict.get(selecao, 'https://flagcdn.com/w320/un.png')

def calcular_entropia(probabilidades):
    """Calcula a entropia de Shannon (incerteza do torneio)"""
    probs = probabilidades[probabilidades > 0]
    return -np.sum(probs * np.log2(probs))

def calcular_numero_efetivo_candidatos(probabilidades):
    """Número efetivo de candidatos (inverso do índice Herfindahl)"""
    return 1 / np.sum(probabilidades ** 2)

def calcular_indice_gini(probabilidades):
    """Índice de Gini para medir concentração"""
    sorted_probs = np.sort(probabilidades)
    n = len(sorted_probs)
    cumulative = np.cumsum(sorted_probs)
    return (2 * np.sum((np.arange(1, n + 1) * sorted_probs)) - (n + 1) * np.sum(sorted_probs)) / (n * np.sum(sorted_probs))

# ============ CSS CUSTOMIZADO — IDENTIDADE VISUAL PREVISÃO ESPORTIVA ============
def inject_custom_css():
    """Injeta o CSS customizado conforme manual de identidade visual Previsão Esportiva"""
    st.markdown("""
    <style>
        /* ===== PALETA OFICIAL PREVISÃO ESPORTIVA ===== */
        :root {
            --verde-claro: #68E70F;
            --verde-valor: #7BC242;
            --verde-escuro: #209927;
            --amarelo: #FFCF26;
            --azul: #035C88;
            --cinza-claro: #F1F1F1;
            --cinza-escuro: #2E2E2E;

            --bg-primary: #12150F;
            --bg-card: #1A1F19;
            --bg-card-strong: #222a21;
            --text-primary: #E0E4DE;
            --text-secondary: #aeb6ad;
            --text-muted: #828b81;
            --border-light: rgba(224, 228, 222, 0.08);
            --shadow-sm: 0 4px 12px rgba(0, 0, 0, 0.2);
            --shadow-md: 0 8px 24px rgba(0, 0, 0, 0.3);
        }
        
        /* ===== FUNDO GLOBAL ===== */
        .stApp {
            background-color: var(--bg-primary) !important;
            background-image: none !important;
            color: var(--text-primary) !important;
        }
        .block-container {
            padding-top: 2rem !important;
        }
        
        /* ===== SIDEBAR ===== */
        section[data-testid="stSidebar"] {
            background-color: #0E120C !important;
            background-image: none !important;
            border-right: 1px solid rgba(104, 231, 15, 0.1);
        }
        section[data-testid="stSidebar"] * {
            color: #e8efe8 !important;
        }

        /* Customização dos Subheaders da Sidebar */
        section[data-testid="stSidebar"] h4 {
            color: var(--text-primary) !important;
            font-family: 'Exo 2', sans-serif !important;
            font-weight: 700 !important;
            font-size: 0.9rem !important;
            letter-spacing: 0.5px !important;
            text-transform: uppercase !important;
            border-bottom: 1px solid rgba(104, 231, 15, 0.15) !important;
            padding-bottom: 6px !important;
            margin-top: 1.8rem !important;
            margin-bottom: 0.8rem !important;
        }
        
        /* Customização dos Labels dos Widgets na Sidebar */
        section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
            color: var(--text-secondary) !important;
            font-weight: 600 !important;
            font-size: 0.8rem !important;
            letter-spacing: 0.2px !important;
            margin-bottom: 2px !important;
        }

        /* Customização dos Sliders (Global) */
        .stSlider {
            padding-bottom: 0.5rem !important;
        }
        .stSlider [data-baseweb="slider"] > div > div {
            height: 4px !important;
            background: linear-gradient(90deg, var(--verde-escuro), var(--verde-claro)) !important;
            border-radius: 2px !important;
        }
        .stSlider [data-baseweb="slider"] [role="slider"] {
            background: #ffffff !important;
            border: 2px solid var(--verde-claro) !important;
            box-shadow: 0 0 6px rgba(104, 231, 15, 0.4) !important;
            width: 12px !important;
            height: 12px !important;
        }
        
        /* Marcas numéricas do Slider */
        .stSlider [data-testid="stWidgetLabel"] + div div {
            color: var(--text-muted) !important;
            font-size: 0.75rem !important;
        }

        /* Botão de Reset na Sidebar */
        .reset-btn-container button {
            background-color: rgba(255, 207, 38, 0.04) !important;
            border: 1px solid rgba(255, 207, 38, 0.25) !important;
            color: var(--amarelo) !important;
            font-weight: 700 !important;
            font-size: 0.85rem !important;
            border-radius: 8px !important;
            padding: 0.5rem 1rem !important;
            transition: all 0.25s ease !important;
            margin-top: 1rem !important;
        }
        .reset-btn-container button:hover {
            background-color: rgba(255, 207, 38, 0.12) !important;
            border-color: var(--amarelo) !important;
            box-shadow: 0 0 8px rgba(255, 207, 38, 0.25) !important;
            color: #ffffff !important;
        }
        
        /* Sliders e Checkboxes - REMOVER VERMELHO */
        [style*="rgb(255, 75, 75)"], [style*="#ff4b4b"], [style*="rgb(255,75,75)"],
        [data-testid="stCheckbox"] [style*="rgb(255, 75, 75)"],
        [data-testid="stCheckbox"] [style*="#ff4b4b"] {
            background: var(--verde-escuro) !important;
            background-color: var(--verde-escuro) !important;
            border-color: var(--verde-escuro) !important;
        }

        /* Sliders globais definidos acima */
        
        /* ===== TIPOGRAFIA ===== */
        h1, h2, h3 {
            font-family: 'Exo 2', sans-serif !important;
            font-weight: 800 !important;
            color: var(--text-primary) !important;
            letter-spacing: -0.5px !important;
        }
        h2::after, h3::after {
            content: "";
            display: block;
            width: 50px;
            height: 3px;
            margin-top: 0.5rem;
            background: var(--verde-escuro);
            border-radius: 10px;
        }

        /* ===== CARDS ===== */
        .stat-card, .team-card, .match-card, .history-card {
            background: var(--bg-card);
            border: 1px solid var(--border-light);
            border-left: 4px solid var(--verde-escuro);
            border-radius: 12px;
            padding: 1.2rem;
            margin: 0.5rem 0;
            box-shadow: var(--shadow-sm);
        }
        
        .stat-value {
            font-size: 2.5rem;
            font-weight: 800;
            color: var(--verde-valor);
        }
        
        /* ===== MÉTRICAS STREAMLIT ===== */
        div[data-testid="stMetric"] {
            background: var(--bg-card);
            border: 1px solid var(--border-light);
            border-left: 3px solid var(--verde-escuro);
            border-radius: 10px;
            padding: 1rem;
        }
        
        div[data-testid="stMetric"] label,
        div[data-testid="stMetric"] label * {
            font-family: 'Exo 2', sans-serif !important;
            font-weight: 700 !important;
            color: var(--text-secondary) !important;
        }
        div[data-testid="stMetric"] div[data-testid="stMetricValue"],
        div[data-testid="stMetric"] div[data-testid="stMetricValue"] * {
            font-family: 'Exo 2', sans-serif !important;
            font-weight: 900 !important;
            line-height: 1 !important;
            color: var(--verde-valor) !important;
        }
        
        /* ===== TABS ===== */
        .stTabs [role="tablist"] {
            background-color: rgba(255, 255, 255, 0.03) !important;
            padding: 6px !important;
            border-radius: 12px !important;
            border: 1px solid rgba(224, 228, 222, 0.05) !important;
            gap: 6px !important;
        }
        .stTabs [role="tab"] {
            background-color: transparent !important;
            color: var(--text-secondary) !important;
            font-family: 'Exo 2', sans-serif !important;
            font-weight: 600 !important;
            font-size: 0.88rem !important;
            padding: 8px 16px !important;
            border-radius: 8px !important;
            border: none !important;
            transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
            height: auto !important;
        }
        .stTabs [role="tab"]:hover {
            color: #ffffff !important;
            background-color: rgba(255, 255, 255, 0.05) !important;
        }
        .stTabs [aria-selected="true"] {
            background-color: var(--verde-escuro) !important;
            color: #ffffff !important;
            font-weight: 700 !important;
            box-shadow: 0 4px 12px rgba(32, 153, 39, 0.25) !important;
        }
        .stTabs [data-baseweb="tab-highlight-bar"] {
            display: none !important;
        }
        
        /* ===== BOTÕES ===== */
        .stButton > button, .stDownloadButton > button {
            font-weight: 700 !important;
            border-radius: 8px !important;
            background: rgba(255,255,255,0.05) !important;
            color: var(--text-primary) !important;
            border: 1px solid var(--border-light) !important;
            transition: all 0.2s ease !important;
        }
        .stButton > button:hover, .stDownloadButton > button:hover {
            border-color: var(--verde-escuro) !important;
            background: rgba(104,231,15,0.05) !important;
        }
        /* Botão Primário (Verde Vibrante) */
        .stButton > button[kind="primary"] {
            background: linear-gradient(135deg, var(--verde-escuro), var(--verde-claro)) !important;
            color: #0d1f0d !important;
            font-weight: 800 !important;
            border: none !important;
            box-shadow: 0 4px 14px rgba(104, 231, 15, 0.35) !important;
        }
        .stButton > button[kind="primary"]:hover {
            background: linear-gradient(135deg, #1f8f26, var(--verde-claro)) !important;
            box-shadow: 0 6px 20px rgba(104, 231, 15, 0.55) !important;
            transform: translateY(-1px);
        }
        .stButton > button[kind="primary"]:active {
            transform: translateY(0);
        }
        
        /* ===== DATAFRAME ===== */
        .stDataFrame {
            border-radius: 8px;
            overflow: hidden;
        }
    </style>
    """, unsafe_allow_html=True)
