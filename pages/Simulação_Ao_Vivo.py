from __future__ import annotations

import html
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_loader import carregar_dados
from utils.helpers import get_bandeira_url, inject_custom_css
from utils.forca_core import ModelParams, render_param_sidebar
from utils.live_model import (
    DefaultModelParams,
    apply_group_result,
    build_default_force_table,
    build_round_of_32,
    group_stage_records,
    new_group_table,
    rank_group,
    simulate_match,
)


PHASE_LABELS = {
    "groups": "Fase de grupos",
    "round32": "Top 32",
    "round16": "Oitavas",
    "quarters": "Quartas",
    "semis": "Semifinais",
    "third": "Disputa de 3º lugar",
    "final": "Final",
    "champion": "Campeão",
}

NEXT_PHASE = {
    "round32": "round16",
    "round16": "quarters",
    "quarters": "semis",
    "semis": "third",
    "third": "final",
    "final": "champion",
}

DELAY_BY_SPEED = {
    "Lento": 0.85,
    "Normal": 0.25,
    "Rápido": 0.08,
    "Instantâneo": 0.0,
}

KNOCKOUT_PHASE_ORDER = ["round32", "round16", "quarters", "semis", "third", "final"]


def inject_live_css() -> None:
    st.markdown(
        """
<style>
    h1, h2, h3 { letter-spacing: 0 !important; }
    .block-container { padding-top: 1.35rem !important; max-width: 1500px; }

    .live-hero {
        background: linear-gradient(135deg, rgba(17,22,17,0.98), rgba(3,92,136,0.30));
        border: 1px solid rgba(241,241,241,0.10);
        border-radius: 8px;
        padding: 1rem 1.15rem;
        margin-bottom: 0.8rem;
    }

    .live-kicker {
        color: #68E70F;
        font-size: 0.78rem;
        font-weight: 800;
        text-transform: uppercase;
    }

    .live-title {
        color: #E0E4DE;
        font-family: 'Exo 2', sans-serif;
        font-size: 2rem;
        line-height: 1.05;
        font-weight: 900;
        margin: 0.25rem 0 0.4rem;
    }

    .live-subtitle {
        color: #aeb6ad;
        font-size: 0.95rem;
        margin: 0;
    }

    .live-metrics {
        display: grid;
        grid-template-columns: repeat(4, minmax(120px, 1fr));
        gap: 0.55rem;
        margin-top: 0.85rem;
    }

    .live-metric {
        background: rgba(255,255,255,0.055);
        border: 1px solid rgba(241,241,241,0.08);
        border-radius: 8px;
        padding: 0.65rem;
    }

    .live-metric-label {
        color: #92a092;
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
    }

    .live-metric-value {
        color: #F1F1F1;
        font-size: 1.25rem;
        font-weight: 900;
        line-height: 1.1;
        margin-top: 0.2rem;
    }

    .top-live-grid {
        display: grid;
        grid-template-columns: minmax(0, 1.35fr) minmax(320px, 0.85fr);
        gap: 0.75rem;
        align-items: start;
        margin-bottom: 0.8rem;
    }

    .current-match {
        background: #111611;
        border: 1px solid rgba(104,231,15,0.18);
        border-left: 4px solid #68E70F;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 0.8rem;
        min-height: 275px;
    }

    .current-stage {
        color: #FFCF26;
        font-size: 0.8rem;
        font-weight: 900;
        text-transform: uppercase;
        text-align: center;
    }

    .score-row {
        display: grid;
        grid-template-columns: minmax(0, 1fr) auto minmax(0, 1fr);
        align-items: center;
        gap: 1rem;
        margin-top: 0.9rem;
    }

    .team-side {
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 0.65rem;
        min-width: 0;
        color: #F1F1F1;
        font-weight: 900;
        font-size: 1.15rem;
        line-height: 1.15;
        text-align: center;
    }

    .team-side.right { justify-content: center; text-align: center; }
    .team-name { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }

    .flag {
        width: 28px;
        height: 19px;
        object-fit: cover;
        border-radius: 3px;
        flex: 0 0 auto;
        box-shadow: 0 1px 4px rgba(0,0,0,0.28);
    }

    .current-flag {
        width: min(100%, 190px);
        aspect-ratio: 3 / 2;
        object-fit: cover;
        border-radius: 6px;
        flex: 0 0 auto;
        box-shadow: 0 8px 24px rgba(0,0,0,0.36);
    }

    .score {
        color: #68E70F;
        font-size: 3.6rem;
        font-weight: 950;
        line-height: 1;
        min-width: 150px;
        text-align: center;
    }

    .match-note {
        color: #c9d1c9;
        font-size: 0.82rem;
        text-align: center;
        margin-top: 0.45rem;
    }

    .groups-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 0.55rem;
    }

    .group-card,
    .mini-panel,
    .history-card {
        background: #111611;
        border: 1px solid rgba(241,241,241,0.08);
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 4px 12px rgba(0,0,0,0.18);
    }

    .group-title {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: rgba(255,255,255,0.045);
        color: #F1F1F1;
        font-weight: 900;
        padding: 0.45rem 0.55rem;
    }

    .group-title span {
        color: #68E70F;
        font-size: 0.78rem;
        text-transform: uppercase;
    }

    .standings {
        width: 100%;
        border-collapse: collapse;
        table-layout: fixed;
        font-size: 0.72rem;
    }

    .standings th {
        color: #92a092;
        font-size: 0.68rem;
        font-weight: 800;
        text-transform: uppercase;
        padding: 0.34rem 0.28rem;
        border-bottom: 1px solid rgba(241,241,241,0.06);
    }

    .standings td {
        color: #e8efe8;
        padding: 0.34rem 0.22rem;
        border-bottom: 1px solid rgba(241,241,241,0.045);
        text-align: center;
    }

    .standings tr:last-child td { border-bottom: none; }
    .standings .team-cell { text-align: left; width: 48%; }
    .standings .qualified td:first-child { color: #68E70F; font-weight: 900; }

    .compact-match {
        display: grid;
        grid-template-columns: minmax(0, 1fr) 54px minmax(0, 1fr);
        gap: 0.35rem;
        align-items: center;
        padding: 0.5rem;
        border-bottom: 1px solid rgba(241,241,241,0.06);
        color: #e8efe8;
        font-size: 0.78rem;
    }

    .compact-match:last-child { border-bottom: none; }
    .compact-score { color: #FFCF26; font-weight: 900; text-align: center; }
    .compact-team { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .compact-team.right { text-align: right; }

    .knockout-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 0.55rem;
    }

    .knockout-phase-title {
        color: #FFCF26;
        font-family: 'Exo 2', sans-serif;
        font-size: 1.05rem;
        font-weight: 900;
        text-transform: uppercase;
        letter-spacing: 0.02em;
        margin: 1rem 0 0.5rem;
        padding-bottom: 0.3rem;
        border-bottom: 1px solid rgba(255,207,38,0.25);
    }

    .knockout-phase-title:first-child { margin-top: 0; }

    .winner-chip {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        color: #68E70F;
        font-size: 0.75rem;
        font-weight: 900;
        margin-top: 0.4rem;
    }

    @media (max-width: 1250px) {
        .groups-grid { grid-template-columns: repeat(3, minmax(0, 1fr)); }
    }

    @media (max-width: 900px) {
        .top-live-grid { grid-template-columns: 1fr; }
        .live-metrics { grid-template-columns: repeat(2, minmax(120px, 1fr)); }
        .score { font-size: 2.4rem; min-width: 100px; }
        .current-flag { width: min(100%, 150px); }
        .groups-grid { grid-template-columns: 1fr; }
    }
</style>
""",
        unsafe_allow_html=True,
    )


def esc(value: object) -> str:
    return html.escape(str(value))


def flag(team: str, bandeiras: dict[str, str], css_class: str = "flag") -> str:
    return f'<img class="{css_class}" src="{esc(get_bandeira_url(team, bandeiras))}" alt="{esc(team)}">'


def build_groups(force_df: pd.DataFrame) -> dict[str, list[str]]:
    ordered = force_df.sort_values(["Grupo", "rank_forca"], ascending=[True, True])
    return ordered.groupby("Grupo")["Seleção"].apply(list).to_dict()


def build_group_fixtures(groups: dict[str, list[str]]) -> list[dict]:
    fixtures = []
    for group in sorted(groups):
        teams = groups[group]
        for index_a, team_a in enumerate(teams):
            for team_b in teams[index_a + 1 :]:
                fixtures.append({"phase": "groups", "group": group, "team_a": team_a, "team_b": team_b})
    return fixtures


def phase_progress() -> tuple[int, int]:
    phase = st.session_state.get("live_phase", "groups")
    if phase == "groups":
        return st.session_state.get("live_group_index", 0), len(st.session_state.get("live_group_fixtures", []))
    if phase in NEXT_PHASE:
        current_round = st.session_state.get("live_current_round", [])
        played = len(st.session_state.get("live_current_phase_matches", []))
        return played, max(1, len(current_round) // 2)
    return 1, 1


def render_hero(current_match: dict | None, total_matches: int, top_team: str) -> None:
    phase = st.session_state.get("live_phase", "groups")
    played, phase_total = phase_progress()
    champion = st.session_state.get("live_campeao") or "-"
    latest = current_match
    title = "Simulação Ao Vivo"
    subtitle = "Modelo padrão: força composta, média de gols 3.00 e Dixon-Coles ativo."
    if latest:
        subtitle = f"Último jogo: {latest['team_a']} {latest['goals_a']} x {latest['goals_b']} {latest['team_b']}"

    st.markdown(
        f"""
<div class="live-hero">
    <div class="live-kicker">{esc(PHASE_LABELS.get(phase, phase))}</div>
    <div class="live-title">{esc(title)}</div>
    <p class="live-subtitle">{esc(subtitle)}</p>
    <div class="live-metrics">
        <div class="live-metric">
            <div class="live-metric-label">Progresso da fase</div>
            <div class="live-metric-value">{played}/{phase_total}</div>
        </div>
        <div class="live-metric">
            <div class="live-metric-label">Jogos simulados</div>
            <div class="live-metric-value">{total_matches}</div>
        </div>
        <div class="live-metric">
            <div class="live-metric-label">Favorito do modelo</div>
            <div class="live-metric-value">{esc(top_team)}</div>
        </div>
        <div class="live-metric">
            <div class="live-metric-label">Campeão</div>
            <div class="live-metric-value">{esc(champion)}</div>
        </div>
    </div>
</div>
""",
        unsafe_allow_html=True,
    )


def render_current_match(match: dict | None, bandeiras: dict[str, str]) -> None:
    if match is None:
        st.markdown(
            """
<div class="current-match">
    <div class="current-stage">Pronto para começar</div>
    <div class="score-row">
        <div class="team-side"><div class="current-flag" style="background: rgba(255,255,255,0.06);"></div><span class="team-name">Seleção A</span></div>
        <div class="score">0 x 0</div>
        <div class="team-side right"><div class="current-flag" style="background: rgba(255,255,255,0.06);"></div><span class="team-name">Seleção B</span></div>
    </div>
    <div class="match-note">Clique em Nova Copa na sidebar para acompanhar a simulação no painel completo.</div>
</div>
""",
            unsafe_allow_html=True,
        )
        return

    note = match.get("group")
    if note:
        note = f"Grupo {note}"
    elif match.get("penalty_winner"):
        note = f"{match['penalty_winner']} avançou nos pênaltis"
    elif match.get("winner"):
        note = f"{match['winner']} avançou"
    else:
        note = PHASE_LABELS.get(match.get("phase", ""), "")

    st.markdown(
        f"""
<div class="current-match">
    <div class="current-stage">{esc(PHASE_LABELS.get(match.get("phase"), match.get("phase", "")))}</div>
    <div class="score-row">
        <div class="team-side">{flag(match['team_a'], bandeiras, "current-flag")}<span class="team-name">{esc(match['team_a'])}</span></div>
        <div class="score">{match['goals_a']} x {match['goals_b']}</div>
        <div class="team-side right">{flag(match['team_b'], bandeiras, "current-flag")}<span class="team-name">{esc(match['team_b'])}</span></div>
    </div>
    <div class="match-note">{esc(note)}</div>
</div>
""",
        unsafe_allow_html=True,
    )


def render_group_cards(group_tables: dict[str, dict[str, dict]], bandeiras: dict[str, str]) -> None:
    cards = []
    for group in sorted(group_tables):
        rows = []
        for position, record in enumerate(rank_group(group_tables[group]), start=1):
            klass = "qualified" if position <= 2 else ""
            rows.append(
                f"""
<tr class="{klass}">
    <td>{position}</td>
    <td class="team-cell">{flag(record['team'], bandeiras)}{esc(record['team'])}</td>
    <td>{record['played']}</td>
    <td><b>{record['points']}</b></td>
    <td>{record['goal_diff']:+d}</td>
    <td>{record['goals_for']}</td>
</tr>
"""
            )
        cards.append(
            f"""
<div class="group-card">
    <div class="group-title">Grupo {esc(group)} <span>{sum(r['played'] for r in group_tables[group].values()) // 2}/6</span></div>
    <table class="standings">
        <thead>
            <tr><th>#</th><th class="team-cell">Seleção</th><th>J</th><th>Pts</th><th>SG</th><th>GP</th></tr>
        </thead>
        <tbody>{''.join(rows)}</tbody>
    </table>
</div>
"""
        )
    st.markdown(f'<div class="groups-grid">{"".join(cards)}</div>', unsafe_allow_html=True)


def render_recent_matches(matches: list[dict], bandeiras: dict[str, str], limit: int = 6) -> None:
    recent = list(reversed(matches[-limit:]))
    if not recent:
        st.markdown('<div class="mini-panel"><div class="group-title">Jogos recentes <span>0</span></div></div>', unsafe_allow_html=True)
        return

    rows = []
    for match in recent:
        suffix = ""
        if match.get("penalty_winner"):
            suffix = " pen"
        rows.append(
            f"""
<div class="compact-match">
    <div class="compact-team">{flag(match['team_a'], bandeiras)}{esc(match['team_a'])}</div>
    <div class="compact-score">{match['goals_a']} x {match['goals_b']}{suffix}</div>
    <div class="compact-team right">{esc(match['team_b'])}{flag(match['team_b'], bandeiras)}</div>
</div>
"""
        )
    st.markdown(
        f"""
<div class="mini-panel">
    <div class="group-title">Jogos recentes <span>{len(matches)}</span></div>
    {''.join(rows)}
</div>
""",
        unsafe_allow_html=True,
    )


def render_knockout(matches: list[dict], bandeiras: dict[str, str]) -> None:
    if not matches:
        return

    grouped: dict[str, list[dict]] = {}
    for match in matches:
        grouped.setdefault(match.get("phase"), []).append(match)

    blocks = []
    # Renderiza da fase mais avançada para a mais antiga, de modo que a chave
    # vá sendo construída "para cima" (Final no topo, Top 32 embaixo) conforme
    # a simulação avança.
    for phase in reversed(KNOCKOUT_PHASE_ORDER):
        phase_matches = grouped.get(phase)
        if not phase_matches:
            continue

        cards = []
        for match in phase_matches:
            winner = match.get("winner") or "-"
            cards.append(
                f"""
<div class="group-card">
    <div class="group-title">{esc(PHASE_LABELS.get(match.get('phase'), match.get('phase', '')))} <span>{esc(match.get('slot', ''))}</span></div>
    <div class="compact-match">
        <div class="compact-team">{flag(match['team_a'], bandeiras)}{esc(match['team_a'])}</div>
        <div class="compact-score">{match['goals_a']} x {match['goals_b']}</div>
        <div class="compact-team right">{esc(match['team_b'])}{flag(match['team_b'], bandeiras)}</div>
    </div>
    <div style="padding: 0 0.55rem 0.55rem;">
        <span class="winner-chip">{flag(winner, bandeiras)}{esc(winner)}{esc(' (pen)' if match.get('penalty_winner') else '')}</span>
    </div>
</div>
"""
            )
        blocks.append(
            f'<div class="knockout-phase-title">{esc(PHASE_LABELS.get(phase, phase))}</div>'
            f'<div class="knockout-grid">{"".join(cards)}</div>'
        )

    st.markdown("".join(blocks), unsafe_allow_html=True)


def build_champions_chart(campeoes: dict[str, int], bandeiras: dict[str, str]) -> go.Figure:
    """Gráfico de barras verticais com os campeões mais frequentes e a bandeira no eixo x."""
    items = sorted(campeoes.items(), key=lambda kv: (-kv[1], kv[0]))[:12]
    teams = [team for team, _ in items]
    counts = [count for _, count in items]
    indices = list(range(len(teams)))

    fig = go.Figure(
        go.Bar(
            x=indices,
            y=counts,
            marker_color="#68E70F",
            marker_line_color="#209927",
            marker_line_width=1,
            text=counts,
            textposition="outside",
            textfont=dict(color="#F1F1F1", size=13),
            hovertext=teams,
            hovertemplate="%{hovertext}: %{y} título(s)<extra></extra>",
        )
    )

    # Bandeiras posicionadas logo abaixo de cada barra, no lugar dos rótulos do eixo x.
    for i, team in enumerate(teams):
        fig.add_layout_image(
            dict(
                source=get_bandeira_url(team, bandeiras),
                xref="x",
                yref="paper",
                x=i,
                y=-0.04,
                sizex=0.7,
                sizey=0.13,
                xanchor="center",
                yanchor="top",
                layer="above",
            )
        )

    max_count = max(counts) if counts else 1
    fig.update_layout(
        height=440,
        margin=dict(l=10, r=10, t=30, b=80),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#C9D1C9",
        bargap=0.35,
        xaxis=dict(
            showticklabels=False,
            showgrid=False,
            zeroline=False,
            range=[-0.6, len(teams) - 0.4] if teams else [-0.6, 0.6],
        ),
        yaxis=dict(
            title="Títulos",
            gridcolor="rgba(255,255,255,0.08)",
            dtick=1 if max_count <= 10 else None,
            rangemode="tozero",
        ),
    )
    return fig


def render_history(bandeiras: dict[str, str]) -> None:
    historico = st.session_state.get("historico_copas", [])
    if not historico:
        st.info("Nenhuma copa simulada ainda.")
        return

    campeoes = {}
    for copa in historico:
        campeoes[copa["campeao"]] = campeoes.get(copa["campeao"], 0) + 1

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Copas", len(historico))
    col_b.metric("Campeões únicos", len(campeoes))
    maior = max(campeoes.items(), key=lambda item: item[1])
    col_c.metric("Maior campeão", f"{maior[0]} ({maior[1]}x)")

    col_results, col_chart = st.columns([1, 2])

    with col_results:
        st.markdown("##### Todos os resultados")
        df_hist = pd.DataFrame(
            [
                {
                    "#": copa["edicao"],
                    "Bandeira": get_bandeira_url(copa["campeao"], bandeiras),
                    "Campeão": copa["campeao"],
                    "Vice": copa["vice"],
                    "3º": copa.get("terceiro", "N/A"),
                    "Final": copa["final_placar"],
                    "Quando": copa["timestamp"],
                }
                for copa in reversed(historico)
            ]
        )
        st.dataframe(
            df_hist,
            hide_index=True,
            height=440,
            width="stretch",
            column_config={
                "#": st.column_config.NumberColumn(width=50),
                "Bandeira": st.column_config.ImageColumn(""),
            },
        )

    with col_chart:
        st.markdown("##### Top campeões")
        st.plotly_chart(build_champions_chart(campeoes, bandeiras), width="stretch")


def initialize_new_cup(groups: dict[str, list[str]], strengths: dict[str, float]) -> None:
    st.session_state["live_running"] = True
    st.session_state["live_phase"] = "groups"
    st.session_state["live_group_tables"] = new_group_table(groups, strengths)
    st.session_state["live_group_fixtures"] = build_group_fixtures(groups)
    st.session_state["live_group_index"] = 0
    st.session_state["live_matches"] = []
    st.session_state["live_knockout_matches"] = []
    st.session_state["live_current_phase_matches"] = []
    st.session_state["live_current_round"] = []
    st.session_state["live_campeao"] = None
    st.session_state["live_vice"] = None
    st.session_state["live_terceiro"] = None
    st.session_state["live_terceiro_disputa"] = []
    st.session_state["live_semifinalistas"] = []
    st.session_state["live_final_placar"] = None
    st.session_state["live_champion_popup_done"] = False
    st.session_state["live_seed"] = int(time.time_ns() % 2**32)


def ensure_state(groups: dict[str, list[str]], strengths: dict[str, float]) -> None:
    if "historico_copas" not in st.session_state:
        st.session_state["historico_copas"] = []
    if "live_running" not in st.session_state:
        st.session_state["live_running"] = False
    if "live_group_tables" not in st.session_state:
        st.session_state["live_group_tables"] = new_group_table(groups, strengths)
    if "live_matches" not in st.session_state:
        st.session_state["live_matches"] = []
    if "live_knockout_matches" not in st.session_state:
        st.session_state["live_knockout_matches"] = []


def render_all(
    overview_slot,
    current_slot,
    groups_slot,
    side_slot,
    knockout_slot,
    bandeiras: dict[str, str],
    top_team: str,
) -> None:
    matches = st.session_state.get("live_matches", [])
    current_match = matches[-1] if matches else None
    with overview_slot.container():
        render_hero(current_match, len(matches), top_team)
    with current_slot.container():
        render_current_match(current_match, bandeiras)
    with groups_slot.container():
        render_group_cards(st.session_state.get("live_group_tables", {}), bandeiras)
    with side_slot.container():
        render_recent_matches(matches, bandeiras)
    with knockout_slot.container():
        render_knockout(st.session_state.get("live_knockout_matches", []), bandeiras)


def run_group_stage(
    params: ModelParams,
    strengths: dict[str, float],
    bandeiras: dict[str, str],
    delay: float,
    slots: tuple,
    top_team: str,
) -> None:
    rng = np.random.default_rng(st.session_state["live_seed"] + len(st.session_state.get("live_matches", [])))
    fixtures = st.session_state["live_group_fixtures"]
    group_tables = st.session_state["live_group_tables"]

    for index in range(st.session_state["live_group_index"], len(fixtures)):
        fixture = fixtures[index]
        match = simulate_match(
            fixture["team_a"],
            fixture["team_b"],
            strengths,
            rng,
            params,
            knockout=False,
        )
        match.update({"phase": "groups", "group": fixture["group"], "slot": f"{index + 1}/{len(fixtures)}"})
        apply_group_result(group_tables[fixture["group"]], match, rng)
        st.session_state["live_group_index"] = index + 1
        st.session_state["live_matches"].append(match)

        if delay > 0 or index == len(fixtures) - 1 or index % 8 == 0:
            render_all(*slots, bandeiras, top_team)
        if delay > 0:
            time.sleep(delay)

    records = group_stage_records(group_tables)
    st.session_state["live_group_records"] = records
    st.session_state["live_current_round"] = build_round_of_32(records, strengths)
    st.session_state["live_current_phase_matches"] = []
    st.session_state["live_phase"] = "round32"
    st.rerun()


def run_knockout_stage(
    params: ModelParams,
    strengths: dict[str, float],
    bandeiras: dict[str, str],
    delay: float,
    slots: tuple,
    top_team: str,
) -> None:
    phase = st.session_state["live_phase"]
    rng = np.random.default_rng(st.session_state["live_seed"] + len(st.session_state.get("live_matches", [])) * 97)

    # Disputa de 3º lugar: jogo único entre os dois perdedores das semifinais.
    if phase == "third":
        disputa = st.session_state.get("live_terceiro_disputa", [])
        if len(disputa) >= 2:
            left, right = disputa[0], disputa[1]
            match = simulate_match(left["team"], right["team"], strengths, rng, params, knockout=True)
            winner_record = left if match["winner"] == left["team"] else right
            match.update({"phase": "third", "slot": "1/1", "winner": winner_record["team"]})
            st.session_state["live_current_phase_matches"] = [match]
            st.session_state["live_knockout_matches"].append(match)
            st.session_state["live_matches"].append(match)
            st.session_state["live_terceiro"] = winner_record["team"]
            render_all(*slots, bandeiras, top_team)
            if delay > 0:
                time.sleep(delay * 1.7)
        st.session_state["live_phase"] = NEXT_PHASE["third"]
        st.session_state["live_current_phase_matches"] = []
        st.rerun()
        return

    current_round = st.session_state.get("live_current_round", [])
    winners = []
    losers = []
    phase_matches = []

    if phase == "semis":
        st.session_state["live_semifinalistas"] = [row["team"] for row in current_round]

    for slot_index in range(0, len(current_round), 2):
        left = current_round[slot_index]
        right = current_round[slot_index + 1]
        match = simulate_match(left["team"], right["team"], strengths, rng, params, knockout=True)
        winner_record = left if match["winner"] == left["team"] else right
        loser_record = right if winner_record is left else left
        winners.append(winner_record)
        losers.append(loser_record)

        match.update(
            {
                "phase": phase,
                "slot": f"{(slot_index // 2) + 1}/{len(current_round) // 2}",
                "winner": winner_record["team"],
            }
        )
        phase_matches.append(match)
        st.session_state["live_current_phase_matches"] = phase_matches
        st.session_state["live_knockout_matches"].append(match)
        st.session_state["live_matches"].append(match)

        if phase == "final":
            suffix = " (pen)" if match.get("penalty_winner") else ""
            st.session_state["live_final_placar"] = (
                f"{match['team_a']} {match['goals_a']} x {match['goals_b']} {match['team_b']}{suffix}"
            )
            st.session_state["live_vice"] = loser_record["team"]

        render_all(*slots, bandeiras, top_team)
        if delay > 0:
            time.sleep(delay * 1.7)

    st.session_state["live_current_round"] = winners
    # Guarda os perdedores das semifinais para a disputa de 3º lugar.
    if phase == "semis":
        st.session_state["live_terceiro_disputa"] = losers
    if phase == "final":
        st.session_state["live_campeao"] = winners[0]["team"]
        st.session_state["live_phase"] = "champion"
    else:
        st.session_state["live_phase"] = NEXT_PHASE[phase]
    st.session_state["live_current_phase_matches"] = []
    st.rerun()


@st.dialog("Fim da Copa 2026")
def show_champion_dialog(bandeiras: dict[str, str]) -> None:
    champion = st.session_state.get("live_campeao", "-")
    vice = st.session_state.get("live_vice", "-")
    terceiro = st.session_state.get("live_terceiro", "-")
    placar = st.session_state.get("live_final_placar", "-")
    st.markdown(
        f"""
<div style="text-align:center;">
    <div style="color:#FFCF26; font-size:0.85rem; font-weight:900; text-transform:uppercase; letter-spacing:0.04em;">Campeão da Copa 2026</div>
    <div style="margin:0.7rem 0;">{flag(champion, bandeiras, "current-flag")}</div>
    <div style="color:#68E70F; font-size:2.1rem; font-weight:900; line-height:1.1;">{esc(champion)}</div>
    <p style="color:#c9d1c9; font-size:0.95rem; margin-top:0.7rem;">
        🥈 Vice: <b>{esc(vice)}</b><br>
        🥉 3º lugar: <b>{esc(terceiro)}</b><br>
        Final: {esc(placar)}
    </p>
    <p style="color:#92a092; font-size:0.8rem; margin-top:0.9rem;">Clique fora desta janela para ver a simulação completa.</p>
</div>
""",
        unsafe_allow_html=True,
    )


def finish_cup(bandeiras: dict[str, str]) -> None:
    st.session_state["live_running"] = False
    champion = st.session_state.get("live_campeao")
    if champion and not st.session_state.get("live_saved_result"):
        st.session_state["historico_copas"].append(
            {
                "edicao": len(st.session_state.get("historico_copas", [])) + 1,
                "campeao": champion,
                "vice": st.session_state.get("live_vice", "N/A"),
                "terceiro": st.session_state.get("live_terceiro", "N/A"),
                "semifinalistas": st.session_state.get("live_semifinalistas", []),
                "final_placar": st.session_state.get("live_final_placar", "N/A"),
                "timestamp": datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
            }
        )
        st.session_state["live_saved_result"] = True

    if champion:
        vice = st.session_state.get("live_vice", "N/A")
        st.markdown(
            f"""
<div class="live-hero" style="text-align:center; border-color: rgba(255,207,38,0.45);">
    <div class="live-kicker">Campeão da Copa 2026</div>
    <div style="margin: 0.8rem 0;">{flag(champion, bandeiras, "flag")}</div>
    <div class="live-title">{esc(champion)}</div>
    <p class="live-subtitle">Vice: {esc(vice)} · 3º lugar: {esc(st.session_state.get('live_terceiro', 'N/A'))} · Final: {esc(st.session_state.get('live_final_placar', 'N/A'))}</p>
</div>
""",
            unsafe_allow_html=True,
        )
        st.balloons()
        if not st.session_state.get("live_champion_popup_done"):
            st.session_state["live_champion_popup_done"] = True
            show_champion_dialog(bandeiras)


inject_custom_css()
inject_live_css()

st.markdown("## Simulação Ao Vivo da Copa")

col_ctrl1, col_ctrl2, col_ctrl3 = st.columns(3, vertical_alignment="bottom")
with col_ctrl1:
    speed = st.pills(
        "Velocidade",
        options=list(DELAY_BY_SPEED),
        default="Normal",
        required=True,
        key="live_speed",
    )
with col_ctrl2:
    start_new_cup = st.button("Nova Copa", type="primary", width='stretch')
with col_ctrl3:
    clear_history = st.button("Limpar histórico", width='stretch')

params = render_param_sidebar()


try:
    raw_df = carregar_dados()
    force_df = build_default_force_table(raw_df, params)
except Exception as error:
    st.error(f"Erro ao carregar dados da simulação ao vivo: {error}")
    st.stop()

groups = build_groups(force_df)
strengths = dict(zip(force_df["Seleção"], force_df["forca_com_offset"]))
bandeiras_dict = dict(zip(force_df["Seleção"], force_df["Link_Bandeira"]))
top_team = str(force_df.iloc[0]["Seleção"])

ensure_state(groups, strengths)

if start_new_cup:
    initialize_new_cup(groups, strengths)
    st.session_state["live_saved_result"] = False

if clear_history:
    st.session_state["historico_copas"] = []
    st.rerun()

overview_slot = st.empty()
top_left_col, top_right_col = st.columns([1.35, 0.85])
with top_left_col:
    current_slot = st.empty()
with top_right_col:
    side_slot = st.empty()

st.markdown("### Mata-mata")
knockout_slot = st.empty()
st.markdown("### Grupos")
groups_slot = st.empty()

slots = (overview_slot, current_slot, groups_slot, side_slot, knockout_slot)
render_all(*slots, bandeiras_dict, top_team)

if st.session_state.get("live_running"):
    delay_value = DELAY_BY_SPEED[speed]
    if st.session_state["live_phase"] == "groups":
        run_group_stage(params, strengths, bandeiras_dict, delay_value, slots, top_team)
    elif st.session_state["live_phase"] in NEXT_PHASE:
        run_knockout_stage(params, strengths, bandeiras_dict, delay_value, slots, top_team)
    elif st.session_state["live_phase"] == "champion":
        finish_cup(bandeiras_dict)

st.markdown("---")
st.markdown("### Histórico")
render_history(bandeiras_dict)
