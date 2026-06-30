from __future__ import annotations

import os
import sys

import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.helpers import inject_custom_css
from utils.forca_core import (
    build_combined,
    compute_knockout_probabilities,
    compute_match_probabilities,
    ensure_selected_teams,
    load_force_dataframe,
    render_param_sidebar,
    team_with_flag,
)


inject_custom_css()

st.markdown("## Probabilidade de uma Partida")
st.markdown(
    """
<p style="font-size: 1rem; margin-bottom: 1.5rem;">
Escolha duas seleções e veja as probabilidades de vitória, empate e derrota,
os gols esperados e a matriz de placares — tudo a partir do indicador de força e
dos parâmetros do modelo definidos na barra lateral. Ative o
<strong>modo mata-mata</strong> para ver, sem empate, a chance de cada seleção
avançar (incluindo prorrogação e pênaltis), como na simulação da Copa.
</p>
""",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("#### Tipo de Jogo")
    modo_mata_mata = st.toggle(
        "Modo mata-mata",
        value=True,
        key="partida_modo_mata_mata",
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

    .knockout-bar-stack {
        font-family: 'Exo 2', sans-serif;
        margin: 1rem 0 1.2rem 0;
    }

    .knockout-bars {
        display: flex;
        flex-direction: column;
        gap: 0.34rem;
    }

    .knockout-mini-bar,
    .knockout-path-bar {
        background: #e0e0e0;
        display: flex;
        overflow: hidden;
        box-shadow: inset 0 1px 3px rgba(0,0,0,0.1);
    }

    .knockout-bar-segment {
        align-items: center;
        color: rgba(255,255,255,0.94);
        display: flex;
        font-size: 0.68rem;
        font-weight: 900;
        justify-content: center;
        line-height: 1;
        min-width: 0;
        overflow: hidden;
        text-shadow: 0 1px 2px rgba(0,0,0,0.3);
        white-space: nowrap;
    }

    .knockout-mini-bar {
        border-radius: 999px;
        height: 18px;
    }

    .knockout-mini-bar--advance {
        height: 18px;
    }

    .knockout-marker-stage {
        height: 54px;
        margin: 0.05rem 0 -0.04rem 0;
        position: relative;
    }

    .knockout-marker {
        border-left: 1px solid;
        border-right: 1px solid;
        border-top: 1px solid;
        position: absolute;
    }

    .knockout-marker::after {
        border-left: 4px solid transparent;
        border-right: 4px solid transparent;
        content: "";
        left: calc(50% - 4px);
        position: absolute;
        top: 13px;
    }

    .knockout-marker--draw {
        border-color: #9a9a9a;
        top: 4px;
        height: 34px;
        z-index: 1;
    }

    .knockout-marker--draw::after {
        border-top: 6px solid #9a9a9a;
    }

    .knockout-marker--penalty {
        border-color: #54a8c8;
        top: 24px;
        height: 28px;
        z-index: 2;
    }

    .knockout-marker--penalty::after {
        display: none;
    }

    .knockout-marker--penalty .knockout-marker-label {
        box-sizing: border-box;
        left: 50%;
        min-width: max-content;
        text-align: center;
        transform: translateX(-50%);
        width: auto;
    }

    .knockout-marker-label {
        background: #ffffff;
        border: 1px solid currentColor;
        border-radius: 999px;
        font-size: 0.67rem;
        font-weight: 900;
        left: 50%;
        line-height: 1;
        padding: 0.18rem 0.42rem;
        position: absolute;
        top: -8px;
        transform: translateX(-50%);
        white-space: nowrap;
    }

    .knockout-path-bar {
        border-radius: 20px;
        height: 38px;
        margin: 0;
    }

    .knockout-path-segment {
        align-items: center;
        color: rgba(255,255,255,0.95);
        display: flex;
        font-size: 0.72rem;
        font-weight: 900;
        justify-content: center;
        line-height: 1;
        min-width: 0;
        overflow: hidden;
        position: relative;
        text-shadow: 0 1px 2px rgba(0,0,0,0.35);
        white-space: nowrap;
    }

    .knockout-conditional {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 0.55rem;
        margin: 0.62rem 0 1.1rem 0;
        font-family: 'Exo 2', sans-serif;
    }

    .knockout-conditional-card {
        background: #ffffff;
        border: 1px solid #e4e4e4;
        border-radius: 8px;
        color: #555760;
        font-size: 0.74rem;
        font-weight: 700;
        line-height: 1.32;
        padding: 0.58rem 0.68rem;
    }

    .knockout-conditional-title {
        color: #34343c;
        font-size: 0.76rem;
        font-weight: 900;
        margin-bottom: 0.22rem;
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
            format_func=team_with_flag,
        )
    with col_away_sel:
        away_team = st.selectbox(
            "Seleção 2",
            team_options,
            key="explorador_away_team",
            format_func=team_with_flag,
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

        if modo_mata_mata:
            knockout = compute_knockout_probabilities(match)
            regular_win_a = float(match["win_a"])
            regular_draw = float(match["draw"])
            regular_win_b = float(match["win_b"])
            extra_win_a = float(knockout["extra_win_a"])
            extra_draw = float(knockout["extra_draw"])
            extra_win_b = float(knockout["extra_win_b"])
            penalty_a = float(knockout.get("penalty_a", 0.4 + 0.2 * float(match["share_a"])))
            penalty_b = float(knockout.get("penalty_b", 0.4 + 0.2 * float(match["share_b"])))
            unconditional_extra_a = regular_draw * extra_win_a
            unconditional_extra_draw = regular_draw * extra_draw
            unconditional_extra_b = regular_draw * extra_win_b
            unconditional_penalty_a = unconditional_extra_draw * penalty_a
            unconditional_penalty_b = unconditional_extra_draw * penalty_b
            if "penalty_a" in knockout and "penalty_b" in knockout:
                advance_a = float(knockout["advance_a"])
                advance_b = float(knockout["advance_b"])
            else:
                advance_a = regular_win_a + unconditional_extra_a + unconditional_penalty_a
                advance_b = regular_win_b + unconditional_extra_b + unconditional_penalty_b

            col_adv_1, col_adv_2 = st.columns(2)
            with col_adv_1:
                st.markdown(
                    f"""
<div class="match-prob-card" style="border: 2px solid #209927; box-shadow: 0 2px 12px rgba(32,153,39,0.12);">
    <div class="match-team-label match-team-label--home" style="color: #209927;">{home_team}</div>
    <div class="match-prob-value match-prob-value--home" style="color: #209927;">{advance_a:.1%}</div>
</div>
""",
                    unsafe_allow_html=True,
                )
            with col_adv_2:
                st.markdown(
                    f"""
<div class="match-prob-card" style="border: 2px solid #035C88; box-shadow: 0 2px 12px rgba(3,92,136,0.12);">
    <div class="match-team-label match-team-label--away" style="color: #035C88;">{away_team}</div>
    <div class="match-prob-value match-prob-value--away" style="color: #035C88;">{advance_b:.1%}</div>
</div>
""",
                    unsafe_allow_html=True,
                )

            st.markdown(
                f"""
<div class="knockout-bar-stack">
    <div class="knockout-bars">
        <div class="knockout-marker-stage">
            <div class="knockout-marker knockout-marker--draw" style="left: {regular_win_a * 100:.2f}%; width: {regular_draw * 100:.2f}%;">
                <span class="knockout-marker-label" style="color:#777;">Empate {regular_draw:.1%}</span>
            </div>
            <div class="knockout-marker knockout-marker--penalty" style="left: {(regular_win_a + unconditional_extra_a) * 100:.2f}%; width: {unconditional_extra_draw * 100:.2f}%;">
                <span class="knockout-marker-label knockout-marker-label--penalty" style="color:#1682b7;">Pênaltis {unconditional_extra_draw:.1%}</span>
            </div>
        </div>
        <div class="knockout-path-bar" title="Caminhos de avanço no mata-mata">
            <div class="knockout-path-segment" title="{home_team} nos 90 min: {regular_win_a:.1%}" style="width: {regular_win_a * 100:.2f}%; background: #16751f;">{regular_win_a:.1%}</div>
            <div class="knockout-path-segment" title="{home_team} na prorrogação: {unconditional_extra_a:.1%}" style="width: {unconditional_extra_a * 100:.2f}%; background: #32b53a;">{unconditional_extra_a:.1%}</div>
            <div class="knockout-path-segment" title="{home_team} nos pênaltis: {unconditional_penalty_a:.1%}" style="width: {unconditional_penalty_a * 100:.2f}%; background: #8dde76;">{unconditional_penalty_a:.1%}</div>
            <div class="knockout-path-segment" title="{away_team} nos pênaltis: {unconditional_penalty_b:.1%}" style="width: {unconditional_penalty_b * 100:.2f}%; background: #7bc4e8;">{unconditional_penalty_b:.1%}</div>
            <div class="knockout-path-segment" title="{away_team} na prorrogação: {unconditional_extra_b:.1%}" style="width: {unconditional_extra_b * 100:.2f}%; background: #1682b7;">{unconditional_extra_b:.1%}</div>
            <div class="knockout-path-segment" title="{away_team} nos 90 min: {regular_win_b:.1%}" style="width: {regular_win_b * 100:.2f}%; background: #034c73;">{regular_win_b:.1%}</div>
        </div>
        <div class="knockout-mini-bar knockout-mini-bar--advance" title="Probabilidade final de avanço">
            <div class="knockout-bar-segment" title="{home_team} avança: {advance_a:.1%}" style="width: {advance_a * 100:.2f}%; background: #209927;">{advance_a:.1%}</div>
            <div class="knockout-bar-segment" title="{away_team} avança: {advance_b:.1%}" style="width: {advance_b * 100:.2f}%; background: #035C88;">{advance_b:.1%}</div>
        </div>
    </div>
<div class="knockout-conditional">
    <div class="knockout-conditional-card">
        <div class="knockout-conditional-title">Dado empate nos 90 min</div>
        {home_team} vence na prorrogação: {extra_win_a:.1%}<br>
        Pênaltis: {extra_draw:.1%}<br>
        {away_team} vence na prorrogação: {extra_win_b:.1%}
    </div>
    <div class="knockout-conditional-card">
        <div class="knockout-conditional-title">Dado que foi para pênaltis</div>
        {home_team} avança: {penalty_a:.1%}<br>
        {away_team} avança: {penalty_b:.1%}
    </div>
</div>
</div>
""",
                unsafe_allow_html=True,
            )
        else:
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
        if modo_mata_mata:
            titulo_placares = (
                "Probabilidade de Placares"
                "<br><span style='font-size:13px; color:#9aa39a;'>no tempo regular (90 min)</span>"
            )
        else:
            titulo_placares = "Probabilidade de Placares"

        fig_heatmap.update_layout(
            title=dict(text=titulo_placares, x=0.5, xanchor="center", font=dict(size=20)),
            xaxis=dict(
                title=dict(text="", standoff=18, font=dict(size=18)),
                tickfont=dict(size=13),
                tickmode="linear",
                dtick=1,
                automargin=False,
            ),
            yaxis=dict(
                title=dict(text=home_team, standoff=18, font=dict(size=18)),
                tickfont=dict(size=13),
                automargin=True,
            ),
            annotations=[
                dict(
                    text=away_team,
                    xref="paper",
                    yref="paper",
                    x=0.5,
                    y=-0.085,
                    xanchor="center",
                    yanchor="top",
                    showarrow=False,
                    font=dict(size=18, color="#C9D1C9"),
                )
            ],
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#C9D1C9",

            height=598,
            margin=dict(l=72, r=20, t=78 if modo_mata_mata else 60, b=95),
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)
