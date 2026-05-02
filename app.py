import streamlit as st

# ============ CONFIGURAÇÃO DA PÁGINA ============
st.set_page_config(
    page_title="Previsão Esportiva — Copa 2026",
    page_icon="assets/bola_previsao.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============ LOGO NA SIDEBAR ============
st.logo(
    "assets/logo fundo preto.png",
    icon_image="assets/bola_previsao.png",
    size="large",
)

# ============ DEFINIÇÃO DAS PÁGINAS ============
datasets = st.Page("pages/Explorador_de_Dados.py", title="Conjunto de Dados", icon="🗄️", default=True)
ao_vivo = st.Page("pages/Simulação_Ao_Vivo.py", title="Simulação Ao Vivo da Copa", icon="⚽")
explorador_forca = st.Page("pages/Explorador_de_Força.py", title="Simulação Copa do Mundo 2026", icon="🏆")

# ============ NAVEGAÇÃO ============
pg = st.navigation(
    {
        "Previsão Esportiva": [datasets, explorador_forca, ao_vivo],
    }
)

# ============ EXECUTAR PÁGINA SELECIONADA ============
pg.run()
