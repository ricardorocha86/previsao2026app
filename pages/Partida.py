from __future__ import annotations

import os
import sys

import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.helpers import inject_custom_css
from utils.forca_core import (
    build_combined,
    compute_match_probabilities,
    ensure_selected_teams,
    load_force_dataframe,
    render_param_sidebar,
)


inject_custom_css()

st.markdown("## Probabilidade de uma Partida")
st.markdown(
    """
<p style="font-size: 1rem; margin-bottom: 1.5rem;">
Escolha duas seleções e veja as probabilidades de vitória, empate e derrota,
os gols esperados e a matriz de placares — tudo a partir do indicador de força e
dos parâmetros do modelo definidos na barra lateral.
</p>
""",
    unsafe_allow_html=True,
)

params = render_param_sidebar()
base_df = load_force_dataframe()
combined_df, weight_sum = build_combined(base_df, params)

media_gols = params.media_gols
usar_dixon_coles = params.usar_dixon_coles
rho_dixon_coles = params.rho_dixon_coles

st.markdown(
    """
<style>
    .match-flag-frame {
        width: 100%;
        aspect-ratio: 3 / 2;
        border-radius: 8px;
        overflow: hidden;
        background: #0d120d;
        display: flex;
        align-items: center;
        justify-content: center;
    }

    .match-flag-frame img {
        width: 100%;
        height: 100%;
        object-fit: cover;
        display: block;
    }

    .match-stat-card,
    .match-prob-card {
        background: #ffffff;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        font-family: 'Exo 2', sans-serif;
    }

    .match-stat-card {
        border: 1px solid #e0e0e0;
        border-radius: 12px;
        padding: 0.85rem;
    }

    .match-prob-card {
        border-radius: 14px;
        padding: 1.2rem;
        box-shadow: 0 2px 12px rgba(0,0,0,0.12);
    }

    .match-prob-card--draw {
        padding: 1rem 0.8rem;
    }

    .match-card-label {
        font-size: 0.95rem;
        color: #5a5a6a;
        line-height: 1.25;
        font-weight: 700;
    }

    .match-stat-value {
        font-size: 2.15rem;
        font-weight: 900;
        line-height: 1;
        margin-top: 0.35rem;
    }

    .match-prob-value {
        font-size: 3.25rem;
        font-weight: 900;
        line-height: 1;
        margin-top: 0.55rem;
    }

    .match-prob-value--home {
        text-align: left;
    }

    .match-prob-value--away {
        text-align: right;
    }

    .match-team-label {
        font-family: 'Montserrat', 'Exo 2', sans-serif;
        font-size: 1rem;
        font-weight: 900;
        line-height: 1.15;
        letter-spacing: 0;
    }

    .match-team-label--home {
        text-align: left;
    }

    .match-team-label--away {
        text-align: right;
    }

    .match-draw-label {
        color: #7d7d86;
        font-size: 0.82rem;
        font-weight: 600;
    }

    .match-draw-value {
        font-size: 2.25rem;
        text-align: center;
    }
</style>
""",
    unsafe_allow_html=True,
)

team_options = combined_df["Seleção"].tolist()
ensure_selected_teams(team_options)

col_left, col_right = st.columns(2)

with col_left:
    col_home_sel, col_spacer_sel, col_away_sel = st.columns([5, 1, 5])
    with col_home_sel:
        home_team = st.selectbox(
            "Seleção 1",
            team_options,
            key="explorador_home_team",
        )
    with col_away_sel:
        away_team = st.selectbox(
            "Seleção 2",
            team_options,
            key="explorador_away_team",
        )

    home_flag = combined_df.loc[combined_df["Seleção"] == home_team, "Link_Bandeira"].iloc[0]
    away_flag = combined_df.loc[combined_df["Seleção"] == away_team, "Link_Bandeira"].iloc[0]

    col_home_flag, col_vs_mid, col_away_flag = st.columns([5, 1, 5])
    with col_home_flag:
        st.markdown(
            f"""
<div style="text-align: center; padding: 0.4rem 0;">
    <div class="match-flag-frame" style="box-shadow: 0 4px 20px rgba(32,153,39,0.25);">
        <img src="{home_flag}" alt="Bandeira {home_team}">
    </div>
</div>
""",
            unsafe_allow_html=True,
        )
    with col_vs_mid:
        st.markdown(
            """
<div style="text-align: center; padding-top: 2rem;">
    <span style="font-size: 1.6rem; font-weight: 800; color: #FFCF26;">VS</span>
</div>
""",
            unsafe_allow_html=True,
        )
    with col_away_flag:
        st.markdown(
            f"""
<div style="text-align: center; padding: 0.4rem 0;">
    <div class="match-flag-frame" style="box-shadow: 0 4px 20px rgba(3,92,136,0.25);">
        <img src="{away_flag}" alt="Bandeira {away_team}">
    </div>
</div>
""",
            unsafe_allow_html=True,
        )

if home_team == away_team:
    with col_left:
        st.info("Escolha duas seleções diferentes para calcular as probabilidades da partida.")
else:
    home_row = combined_df.loc[combined_df["Seleção"] == home_team].iloc[0]
    away_row = combined_df.loc[combined_df["Seleção"] == away_team].iloc[0]

    match = compute_match_probabilities(
        force_a=float(home_row["forca_com_offset"]),
        force_b=float(away_row["forca_com_offset"]),
        media_gols=media_gols,
        usar_dixon_coles=usar_dixon_coles,
        rho_dixon_coles=rho_dixon_coles,
    )

    with col_left:
        st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)

        col_hm1, col_hm2, col_spacer_m, col_am1, col_am2 = st.columns([2.5, 2.5, 0.5, 2.5, 2.5])
        with col_hm1:
            st.markdown(
                f"""
<div class="match-stat-card" style="border-left: 3px solid #209927;">
    <div class="match-card-label" style="font-size: 0.82rem; font-weight: 600;">Força</div>
    <div class="match-stat-value" style="color: #209927;">{float(home_row['forca_com_offset']):.3f}</div>
</div>
""",
                unsafe_allow_html=True,
            )
        with col_hm2:
            st.markdown(
                f"""
<div class="match-stat-card" style="border-left: 3px solid #209927;">
    <div class="match-card-label" style="font-size: 0.82rem; font-weight: 600;">Gols esp.</div>
    <div class="match-stat-value" style="color: #209927;">{float(match['lambda_a']):.2f}</div>
</div>
""",
                unsafe_allow_html=True,
            )
        with col_am1:
            st.markdown(
                f"""
<div class="match-stat-card" style="border-left: 3px solid #035C88;">
    <div class="match-card-label" style="font-size: 0.82rem; font-weight: 600;">Força</div>
    <div class="match-stat-value" style="color: #035C88;">{float(away_row['forca_com_offset']):.3f}</div>
</div>
""",
                unsafe_allow_html=True,
            )
        with col_am2:
            st.markdown(
                f"""
<div class="match-stat-card" style="border-left: 3px solid #035C88;">
    <div class="match-card-label" style="font-size: 0.82rem; font-weight: 600;">Gols esp.</div>
    <div class="match-stat-value" style="color: #035C88;">{float(match['lambda_b']):.2f}</div>
</div>
""",
                unsafe_allow_html=True,
            )

        st.markdown("<div style='height: 1.2rem;'></div>", unsafe_allow_html=True)

        col_prob_1, col_prob_2, col_prob_3 = st.columns([3, 2, 3])

        with col_prob_1:
            st.markdown(
                f"""
<div class="match-prob-card" style="border: 2px solid #209927; box-shadow: 0 2px 12px rgba(32,153,39,0.12);">
    <div class="match-team-label match-team-label--home" style="color: #209927;">{home_team}</div>
    <div class="match-prob-value match-prob-value--home" style="color: #209927;">{float(match['win_a']):.1%}</div>
</div>
""",
                unsafe_allow_html=True,
            )

        with col_prob_2:
            st.markdown(
                f"""
<div class="match-prob-card match-prob-card--draw" style="border: 2px solid #9e9e9e; box-shadow: 0 2px 12px rgba(158,158,158,0.12);">
    <div class="match-card-label match-draw-label">Empate</div>
    <div class="match-prob-value match-draw-value" style="color: #9e9e9e;">{float(match['draw']):.1%}</div>
</div>
""",
                unsafe_allow_html=True,
            )

        with col_prob_3:
            st.markdown(
                f"""
<div class="match-prob-card" style="border: 2px solid #035C88; box-shadow: 0 2px 12px rgba(3,92,136,0.12);">
    <div class="match-team-label match-team-label--away" style="color: #035C88;">{away_team}</div>
    <div class="match-prob-value match-prob-value--away" style="color: #035C88;">{float(match['win_b']):.1%}</div>
</div>
""",
                unsafe_allow_html=True,
            )

        st.markdown(
            f"""
<div style="background: #e0e0e0; border-radius: 20px; height: 36px; display: flex; overflow: hidden; margin: 1rem 0 1.5rem 0; box-shadow: inset 0 1px 3px rgba(0,0,0,0.1);">
    <div style="width: {float(match['win_a']) * 100:.2f}%; background: #209927;"></div>
    <div style="width: {float(match['draw']) * 100:.2f}%; background: linear-gradient(90deg, #d8d8d8, #b8b8b8);"></div>
    <div style="width: {float(match['win_b']) * 100:.2f}%; background: #035C88;"></div>
</div>
""",
            unsafe_allow_html=True,
        )

    with col_right:
        max_gols_display = 7
        prob_display = match["matrix"][: max_gols_display + 1, : max_gols_display + 1] * 100.0
        annotations_text = [
            [f"{prob_display[i, j]:.1f}%" for j in range(max_gols_display + 1)]
            for i in range(max_gols_display + 1)
        ]

        fig_heatmap = go.Figure(
            data=go.Heatmap(
                z=prob_display,
                x=[str(i) for i in range(max_gols_display + 1)],
                y=[str(i) for i in range(max_gols_display + 1)],
                zmin=0,
                zmax=float(prob_display.max()),
                colorscale=[
                    [0.00, "#010301"],
                    [1.00, "#55B81E"],
                ],
                text=annotations_text,
                texttemplate="%{text}",
                textfont={"size": 16, "color": "#F1F1F1"},
                hovertemplate=(
                    f"{home_team}: %{{y}} x %{{x}}: {away_team}"
                    "<br>Probabilidade: %{z:.2f}%<extra></extra>"
                ),
                showscale=False,
            )
        )
        fig_heatmap.update_layout(
            title=dict(text="Probabilidade de Placares", x=0.5, xanchor="center", font=dict(size=20)),
            xaxis=dict(
                title=dict(text=away_team, standoff=18, font=dict(size=18)),
                tickfont=dict(size=13),
                automargin=True,
            ),
            yaxis=dict(
                title=dict(text=home_team, standoff=18, font=dict(size=18)),
                tickfont=dict(size=13),
                automargin=True,
            ),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#C9D1C9",

            height=598,
            margin=dict(l=72, r=20, t=60, b=70),
        )
        st.plotly_chart(fig_heatmap, width='stretch')
