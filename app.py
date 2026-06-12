import streamlit as st

# ============ CONFIGURAÇÃO DA PÁGINA ============
st.set_page_config(
    page_title="Previsão Esportiva — Copa 2026",
    page_icon="assets/bola_previsao.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============ DEFINIÇÃO DAS PÁGINAS ============
datasets = st.Page("pages/Explorador_de_Dados.py", title="Conjunto de Dados", icon="📁")
indicador_forca = st.Page("pages/Indicador_de_Força.py", title="Indicador de Força", icon="📊")
partida = st.Page("pages/Partida.py", title="Probabilidade de uma Partida", icon="🆚", default=True)
explorador_forca = st.Page("pages/Explorador_de_Força.py", title="Simulação Copa do Mundo 2026", icon="🏆")
ao_vivo = st.Page("pages/Simulação_Ao_Vivo.py", title="Simulação Ao Vivo da Copa", icon="⚽")

# ============ NAVEGAÇÃO ============
pg = st.navigation([partida, ao_vivo, explorador_forca, indicador_forca, datasets])

# ============ EXECUTAR PÁGINA SELECIONADA ============
pg.run()
