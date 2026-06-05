from __future__ import annotations

import os
import sys

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.helpers import inject_custom_css
from utils.forca_core import build_combined, load_force_dataframe, render_param_sidebar


inject_custom_css()

st.markdown("## Indicador de Força")
st.markdown(
    """
<p style="font-size: 1rem; margin-bottom: 1.5rem;">
Combine os índices normalizados de <b>FIFA</b>,
<b>ELO</b>, o <b>momento recente</b>, <b>valor de mercado</b>,
o <b>histórico em Copas</b> e a condição de <b>anfitrião</b>, ajuste <b>elasticidade</b>
e <b>offset</b> na barra lateral e veja a força resultante de cada seleção.
</p>
""",
    unsafe_allow_html=True,
)

params = render_param_sidebar()
base_df = load_force_dataframe()
combined_df, weight_sum = build_combined(base_df, params)

if weight_sum <= 0:
    st.warning("A soma dos pesos está zerada. Ajuste ao menos um peso para construir a força resultante.")

effective_fifa = params.weight_fifa / weight_sum if weight_sum > 0 else 0.0
effective_elo = params.weight_elo / weight_sum if weight_sum > 0 else 0.0
effective_momentum = params.weight_momentum / weight_sum if weight_sum > 0 else 0.0
effective_market = params.weight_market / weight_sum if weight_sum > 0 else 0.0
effective_history = params.weight_history / weight_sum if weight_sum > 0 else 0.0
effective_host = params.weight_host / weight_sum if weight_sum > 0 else 0.0

col_m1, col_m2, col_m3, col_m4, col_m5, col_m6 = st.columns(6)

with col_m1:
    st.metric("Efetivo FIFA", f"{effective_fifa:.1%}")
with col_m2:
    st.metric("Efetivo ELO", f"{effective_elo:.1%}")
with col_m3:
    st.metric("Efetivo Momento", f"{effective_momentum:.1%}")
with col_m4:
    st.metric("Efetivo Mercado", f"{effective_market:.1%}")
with col_m5:
    st.metric("Efetivo História", f"{effective_history:.1%}")
with col_m6:
    st.metric("Efetivo Anfitrião", f"{effective_host:.1%}")

st.markdown("### Tabela de Força")

display_table = combined_df[
    [
        "Seleção",
        "fifa_force_01",
        "elo_force_01",
        "momentum_force_01",
        "market_force_01",
        "world_cup_history_01",
        "is_host",
        "forca_resultante_01",
        "forca_com_offset",
        "market_prob",
    ]
].rename(
    columns={
        "Seleção": "Seleção",
        "fifa_force_01": "Fifa",
        "elo_force_01": "Elo",
        "momentum_force_01": "Momento",
        "market_force_01": "Mercado",
        "world_cup_history_01": "Historico",
        "is_host": "Anfitrião",
        "forca_resultante_01": "Força",
        "forca_com_offset": "Força Ajustada",
        "market_prob": "Prob Implicita",
    }
)

display_table = display_table.copy()
display_table.insert(0, "Rank", range(1, len(display_table) + 1))

st.dataframe(
    display_table,
    width="stretch",
    height=520,
    hide_index=True,
    column_config={
        "Rank": st.column_config.NumberColumn("Rank", width=50, format="%d"),
        "Fifa": st.column_config.NumberColumn(format="%.3f"),
        "Elo": st.column_config.NumberColumn(format="%.3f"),
        "Momento": st.column_config.NumberColumn(format="%.3f"),
        "Mercado": st.column_config.NumberColumn(format="%.3f"),
        "Historico": st.column_config.NumberColumn(format="%.3f"),
        "Anfitrião": st.column_config.NumberColumn(format="%.0f"),
        "Força": st.column_config.NumberColumn(format="%.3f"),
        "Força Ajustada": st.column_config.NumberColumn(format="%.3f"),
        "Prob Implicita": st.column_config.NumberColumn(format="%.4f"),
    },
)
