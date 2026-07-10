"""Simulação Minuto a Minuto — linha do tempo animada da partida.

As probabilidades usam o mesmo modelo calibrado do resto do app
(``compute_match_probabilities`` com Dixon-Coles; prorrogação ``lambda * 0.3``;
pênaltis via ``penalty_win_probability_from_share``), então o ponto de 0' e o
"avança %" batem com as páginas Partida e Simulação da Copa. O gráfico é animado
no navegador (Plotly), começa no 0' e o ▶ Play revela o jogo minuto a minuto.
"""

from __future__ import annotations

import html
import os
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.forca_core import (
    build_combined,
    compute_match_probabilities,
    ensure_selected_teams,
    load_force_dataframe,
    poisson_matrix,
    render_param_sidebar,
    team_with_flag,
)
from utils.helpers import get_bandeira_url, inject_custom_css
from utils.simulador_oficial import penalty_win_probability_from_share

# ───────────────────────── Constantes ─────────────────────────

# Duração de cada frame (ms) da animação Plotly disparada pelo botão ▶ Play.
SPEED_MS = {"Lento": 550, "Normal": 210, "Rápido": 70, "Instantâneo": 18}

# Barra de ferramentas do Plotly: manter só o download (câmera). O "tela cheia"
# é botão do próprio Streamlit e continua disponível.
PLOTLY_CONFIG = {
    "displaylogo": False,
    "modeBarButtonsToRemove": ["zoom2d", "pan2d", "select2d", "lasso2d",
                               "zoomIn2d", "zoomOut2d", "autoScale2d", "resetScale2d"],
}

EXTRA_TIME_FACTOR = 0.3  # prorrogação = lambda * 0.3, igual ao motor oficial
SHOOTOUT_CONVERSION = 0.75  # conversão por cobrança, só para ilustrar a disputa

HOME_COLOR = "#209927"
AWAY_COLOR = "#035C88"
DRAW_COLOR = "#9CA3AF"

TEAM_DOMINANT_COLORS = {
    "França": "#1F4AA8", "Espanha": "#C8102E", "Brasil": "#0E8C3A",
    "Argentina": "#4B9CD3", "Inglaterra": "#C8102E", "Portugal": "#D0001B",
    "Alemanha": "#333333", "Holanda": "#F36C21", "Bélgica": "#C9A100",
    "Uruguai": "#3FA9D6", "Colômbia": "#D4A800", "México": "#006847",
    "Estados Unidos": "#3C3B6E", "Canadá": "#D80621", "Marrocos": "#C1272D",
    "Croácia": "#D7141A", "Japão": "#BC002D", "Coreia do Sul": "#CD2E3A",
    "Suíça": "#D52B1E", "Senegal": "#00853F", "Noruega": "#BA0C2F",
}

STAGE_FILL = {
    "1T": "rgba(104,231,15,0.045)",
    "2T": "rgba(3,92,136,0.060)",
    "Prorrogação 1T": "rgba(255,207,38,0.055)",
    "Prorrogação 2T": "rgba(255,207,38,0.085)",
}
STAGE_END_LABEL = {"1T": "45'", "2T": "90'", "Prorrogação 1T": "105'", "Prorrogação 2T": "120'"}


# ───────────────────────── Utilitários ─────────────────────────

def esc(value: object) -> str:
    return html.escape(str(value))


def team_color(team: str, fallback: str) -> str:
    if team in TEAM_DOMINANT_COLORS:
        return TEAM_DOMINANT_COLORS[team]
    palette = ["#209927", "#035C88", "#C8102E", "#F59E0B", "#7C3AED", "#0EA5E9"]
    return palette[sum(map(ord, team)) % len(palette)] if team else fallback


def flag_img(team: str, bandeiras: dict[str, str], css_class: str = "minute-flag") -> str:
    url = get_bandeira_url(team, bandeiras)
    return f'<img class="{css_class}" src="{esc(url)}" alt="{esc(team)}">'


def stoppage_label(base_minute: int, added: int) -> str:
    return f"{base_minute}'" if added <= 0 else f"{base_minute}+{added}'"


def build_minute_schedule(half1_added: int, half2_added: int) -> list[dict]:
    schedule: list[dict] = []
    for minute in range(1, 46):
        schedule.append({"stage": "1T", "minute": minute, "label": f"{minute}'"})
    for added in range(1, half1_added + 1):
        schedule.append({"stage": "1T", "minute": 45 + added, "label": stoppage_label(45, added)})
    for minute in range(46, 91):
        schedule.append({"stage": "2T", "minute": minute, "label": f"{minute}'"})
    for added in range(1, half2_added + 1):
        schedule.append({"stage": "2T", "minute": 90 + added, "label": stoppage_label(90, added)})
    return schedule


def build_extra_schedule(extra1_added: int, extra2_added: int) -> list[dict]:
    schedule: list[dict] = []
    for minute in range(91, 106):
        schedule.append({"stage": "Prorrogação 1T", "minute": minute, "label": f"{minute}'"})
    for added in range(1, extra1_added + 1):
        schedule.append({"stage": "Prorrogação 1T", "minute": 105 + added, "label": stoppage_label(105, added)})
    for minute in range(106, 121):
        schedule.append({"stage": "Prorrogação 2T", "minute": minute, "label": f"{minute}'"})
    for added in range(1, extra2_added + 1):
        schedule.append({"stage": "Prorrogação 2T", "minute": 120 + added, "label": stoppage_label(120, added)})
    return schedule


# ───────────────────── Modelo (unificado ao oficial) ─────────────────────

def conditional_probs(
    score_a: int,
    score_b: int,
    remaining_minutes: int,
    rate_a: float,
    rate_b: float,
    usar_dixon_coles: bool,
    rho: float,
) -> dict[str, float]:
    """P(vitória/empate/derrota) dado o placar atual e o tempo restante.

    Modela os gols futuros como Poisson(rate · minutos restantes). Em ``t = 0``
    com o tempo regular inteiro isto reproduz ``compute_match_probabilities``.
    """
    lambda_a = max(0.0, rate_a * max(0, remaining_minutes))
    lambda_b = max(0.0, rate_b * max(0, remaining_minutes))
    max_goals = max(10, int(np.ceil(max(lambda_a, lambda_b) + 6)))
    matrix = poisson_matrix(lambda_a, lambda_b, max_goals=max_goals,
                            usar_dixon_coles=usar_dixon_coles, rho_dixon_coles=rho)

    win_a = draw = win_b = 0.0
    for future_a in range(matrix.shape[0]):
        for future_b in range(matrix.shape[1]):
            probability = float(matrix[future_a, future_b])
            final_a, final_b = score_a + future_a, score_b + future_b
            if final_a > final_b:
                win_a += probability
            elif final_b > final_a:
                win_b += probability
            else:
                draw += probability
    return {"win_a": win_a, "draw": draw, "win_b": win_b}


def make_state_probs(
    rate_a: float,
    rate_b: float,
    share_a: float,
    extra_minutes: int,
    usar_dixon_coles: bool,
    rho: float,
):
    """Fábrica de funções de probabilidade que fixam os parâmetros do confronto.

    Devolve ``(regular_state, extra_state)`` — a probabilidade de avanço usa a
    prorrogação (Poisson(lambda·0.3), sem Dixon-Coles, exatamente como o motor)
    e o desempate por pênaltis via ``penalty_win_probability_from_share``.
    """
    penalty_a = penalty_win_probability_from_share(share_a)
    penalty_b = penalty_win_probability_from_share(1.0 - share_a)

    # Constante do confronto: resultado de uma prorrogação começando 0-0.
    extra_rate_a = rate_a * 90.0 * EXTRA_TIME_FACTOR / max(1, extra_minutes)
    extra_rate_b = rate_b * 90.0 * EXTRA_TIME_FACTOR / max(1, extra_minutes)
    extra0 = conditional_probs(0, 0, extra_minutes, extra_rate_a, extra_rate_b, False, rho)
    adv_if_draw_a = extra0["win_a"] + extra0["draw"] * penalty_a
    adv_if_draw_b = extra0["win_b"] + extra0["draw"] * penalty_b

    def regular_state(score_a: int, score_b: int, remaining: int) -> dict[str, float]:
        reg = conditional_probs(score_a, score_b, remaining, rate_a, rate_b, usar_dixon_coles, rho)
        return {
            **reg,
            "adv_a": reg["win_a"] + reg["draw"] * adv_if_draw_a,
            "adv_b": reg["win_b"] + reg["draw"] * adv_if_draw_b,
        }

    def extra_state(score_a: int, score_b: int, remaining: int) -> dict[str, float]:
        ex = conditional_probs(score_a, score_b, remaining, extra_rate_a, extra_rate_b, False, rho)
        return {
            **ex,
            "adv_a": ex["win_a"] + ex["draw"] * penalty_a,
            "adv_b": ex["win_b"] + ex["draw"] * penalty_b,
        }

    return regular_state, extra_state, penalty_a, extra_rate_a, extra_rate_b


# ───────────────────── Pênaltis (ilustração coerente) ─────────────────────

def penalty_terminal(score_a: int, score_b: int, kicks_a: int, kicks_b: int) -> int | None:
    if kicks_a <= 5 and kicks_b <= 5:
        if score_a > score_b + (5 - kicks_b):
            return 1
        if score_b > score_a + (5 - kicks_a):
            return 2
    if kicks_a >= 5 and kicks_b >= 5 and kicks_a == kicks_b and score_a != score_b:
        return 1 if score_a > score_b else 2
    return None


def penalty_next_team(kicks_a: int, kicks_b: int, first_team: int) -> int:
    return first_team if (kicks_a + kicks_b) % 2 == 0 else (2 if first_team == 1 else 1)


def _one_shootout(rng: np.random.Generator, first_team: int) -> tuple[list[dict], int]:
    score_a = score_b = kicks_a = kicks_b = 0
    kicks: list[dict] = []
    for kick_index in range(1, 41):
        team = penalty_next_team(kicks_a, kicks_b, first_team)
        made = float(rng.random()) <= SHOOTOUT_CONVERSION
        if team == 1:
            kicks_a += 1
            score_a += int(made)
        else:
            kicks_b += 1
            score_b += int(made)
        winner = penalty_terminal(score_a, score_b, kicks_a, kicks_b)
        kicks.append({"kick_index": kick_index, "team_code": team, "made": made,
                      "score_a": score_a, "score_b": score_b,
                      "kicks_a": kicks_a, "kicks_b": kicks_b})
        if winner is not None:
            return kicks, winner
    return kicks, 1 if score_a >= score_b else 2


def illustrative_shootout(rng: np.random.Generator, decided_winner: int) -> list[dict]:
    """Sequência de cobranças coerente com o vencedor já decidido pelo modelo.

    O vencedor vem de ``penalty_win_probability_from_share`` (calibrado); aqui só
    geramos uma disputa plausível que termina nesse vencedor, por rejeição.
    """
    for _ in range(200):
        first_team = int(rng.integers(1, 3))
        kicks, winner = _one_shootout(rng, first_team)
        if winner == decided_winner:
            return kicks
    return kicks  # fallback improvável


# ───────────────────── Simulação da trajetória ─────────────────────

def simulate_trajectory(
    home_team: str,
    away_team: str,
    lambda_a: float,
    lambda_b: float,
    share_a: float,
    knockout: bool,
    usar_dixon_coles: bool,
    rho: float,
    seed: int,
) -> dict:
    rng = np.random.default_rng(seed)
    rate_a, rate_b = lambda_a / 90.0, lambda_b / 90.0

    regular_schedule = build_minute_schedule(int(rng.integers(3, 10)), int(rng.integers(3, 10)))
    extra_schedule = build_extra_schedule(int(rng.integers(1, 4)), int(rng.integers(1, 4)))

    regular_state, extra_state, penalty_a, extra_rate_a, extra_rate_b = make_state_probs(
        rate_a, rate_b, share_a, len(extra_schedule), usar_dixon_coles, rho
    )

    points: list[dict] = []
    events: list[dict] = []
    step = 0
    score_a = score_b = 0

    def record(label, stage, probs, note):
        points.append({"step": step, "label": label, "stage": stage,
                       "score_a": int(score_a), "score_b": int(score_b),
                       "win_a": probs.get("win_a", 0.0), "draw": probs.get("draw", 0.0),
                       "win_b": probs.get("win_b", 0.0),
                       "adv_a": probs.get("adv_a", probs.get("win_a", 0.0)),
                       "adv_b": probs.get("adv_b", probs.get("win_b", 0.0)), "note": note})

    def register_goals(minute, n, team):
        for _ in range(n):
            events.append({"step": step, "label": minute["label"], "stage": minute["stage"],
                           "team": team, "kind": "Gol"})

    # Ponto pré-jogo (0') — bate com a página Partida.
    initial = regular_state(0, 0, len(regular_schedule)) if knockout else \
        conditional_probs(0, 0, len(regular_schedule), rate_a, rate_b, usar_dixon_coles, rho)
    record("0'", "Pré-jogo", initial, "Início da partida")

    total_reg = len(regular_schedule)
    for index, minute in enumerate(regular_schedule, start=1):
        step += 1
        goals_a, goals_b = int(rng.poisson(rate_a)), int(rng.poisson(rate_b))
        score_a += goals_a
        score_b += goals_b
        register_goals(minute, goals_a, home_team)
        register_goals(minute, goals_b, away_team)
        remaining = total_reg - index
        probs = regular_state(score_a, score_b, remaining) if knockout else \
            conditional_probs(score_a, score_b, remaining, rate_a, rate_b, usar_dixon_coles, rho)
        record(minute["label"], minute["stage"], probs, "Tempo normal")

    winner_code: int | None = None
    went_extra = went_penalties = False
    penalty_kicks: list[dict] = []

    if score_a > score_b:
        winner_code = 1
    elif score_b > score_a:
        winner_code = 2
    elif not knockout:
        winner_code = 0
    else:
        went_extra = True
        total_extra = len(extra_schedule)
        for index, minute in enumerate(extra_schedule, start=1):
            step += 1
            goals_a, goals_b = int(rng.poisson(extra_rate_a)), int(rng.poisson(extra_rate_b))
            score_a += goals_a
            score_b += goals_b
            register_goals(minute, goals_a, home_team)
            register_goals(minute, goals_b, away_team)
            record(minute["label"], minute["stage"],
                   extra_state(score_a, score_b, total_extra - index), "Prorrogação")

        if score_a > score_b:
            winner_code = 1
        elif score_b > score_a:
            winner_code = 2
        else:
            went_penalties = True
            winner_code = 1 if float(rng.random()) < penalty_a else 2
            penalty_kicks = illustrative_shootout(rng, winner_code)
            for kick in penalty_kicks:
                step += 1
                team = home_team if kick["team_code"] == 1 else away_team
                kind = "Pênalti convertido" if kick["made"] else "Pênalti perdido"
                events.append({"step": step, "label": f"P{kick['kick_index']}",
                               "stage": "Pênaltis", "team": team, "kind": kind})

    return {
        "home_team": home_team, "away_team": away_team,
        "lambda_a": lambda_a, "lambda_b": lambda_b, "rate_a": rate_a, "rate_b": rate_b,
        "share_a": share_a, "knockout": knockout, "seed": int(seed),
        "score_a": score_a, "score_b": score_b, "winner_code": winner_code,
        "went_extra": went_extra, "went_penalties": went_penalties,
        "penalty_a": penalty_a, "penalty_kicks": penalty_kicks,
        "points": points, "events": events,
    }


# ───────────────────────── Gráfico (animação client-side) ─────────────────────────

def build_probability_figure(result: dict, bandeiras: dict[str, str], speed: str,
                             chart_kind: str) -> go.Figure:
    """Gráfico animado no NAVEGADOR (um único render — não pisca).

    Começa no 0' (linha vazia) e, ao clicar ▶ Play, a linha cresce minuto a
    minuto revelando gols e a probabilidade atual até o placar final. A animação
    é do próprio Plotly (client-side), então roda suave e sem re-render do
    Streamlit. Sem Play, o gráfico fica parado no início (não entrega o final).
    """
    points = [p for p in result["points"] if p["stage"] != "Pênaltis"]
    frame = pd.DataFrame(points)
    n = len(frame)
    steps = frame["step"].tolist()
    home = result["home_team"]
    away = result["away_team"]
    home_color = team_color(home, HOME_COLOR)
    away_color = team_color(away, AWAY_COLOR)

    # séries: (coluna, legenda, cor, seleção-para-bandeira) — None = sem bandeira (empate)
    if chart_kind == "avanco":
        series = [("adv_a", f"{team_with_flag(home)} avança", home_color, home),
                  ("adv_b", f"{team_with_flag(away)} avança", away_color, away)]
    else:
        series = [("win_a", f"{team_with_flag(home)} vence", home_color, home),
                  ("draw", "Empate", DRAW_COLOR, None),
                  ("win_b", f"{team_with_flag(away)} vence", away_color, away)]

    custom = np.stack([frame["label"], frame["score_a"], frame["score_b"], frame["stage"]], axis=-1)
    hover = ("%{customdata[3]} %{customdata[0]}<br>Placar: %{customdata[1]} x %{customdata[2]}"
             "<br>%{y:.1f}%<extra></extra>")

    x_span = max(float(max(steps)) - float(min(steps)), 20.0)
    margin_x = x_span * 0.06
    flag_x, flag_y = x_span * 0.05, 10.0  # tamanho da bandeira que segue a ponta da linha

    def line_scatter(col: str, color: str, name: str, upto: int) -> go.Scatter:
        return go.Scatter(
            x=steps[: upto + 1], y=(frame[col].iloc[: upto + 1] * 100).tolist(),
            mode="lines", line=dict(color=color, width=3), name=name,
            customdata=custom[: upto + 1], hovertemplate=hover)

    def cursor_scatter(upto: int) -> go.Scatter:
        return go.Scatter(x=[steps[upto], steps[upto]], y=[-2, 118], mode="lines",
                          line=dict(color="rgba(255,207,38,0.55)", width=1.5, dash="dot"),
                          hoverinfo="skip", showlegend=False, name="cursor")

    # faixas de período (estáticas — reincluídas em cada frame, pois o frame
    # substitui a lista inteira de shapes/annotations).
    period_shapes, period_annots = [], []
    for stage in ["1T", "2T", "Prorrogação 1T", "Prorrogação 2T"]:
        sub = frame[frame["stage"] == stage]
        if sub.empty:
            continue
        x0, x1 = float(sub["step"].min()), float(sub["step"].max())
        x1 = x1 if x1 > x0 else x0 + 0.5
        period_shapes.append(dict(type="rect", xref="x", yref="paper", x0=x0, x1=x1, y0=0, y1=1,
                                  fillcolor=STAGE_FILL[stage], line=dict(width=0), layer="below"))
        period_shapes.append(dict(type="line", xref="x", yref="paper", x0=x0, x1=x0, y0=0, y1=1,
                                  line=dict(color="rgba(241,241,241,0.22)", width=1), layer="below"))
        period_annots.append(dict(x=(x0 + x1) / 2, y=1.01, xref="x", yref="paper",
                                  text=stage.replace("Prorrogação ", "Prorr. "), showarrow=False,
                                  font=dict(color="#C9D1C9", size=10)))

    # gols por seleção como DADO DE TRACE (segmentos verticais). Manter os gols em
    # traces — e não em shapes/annotations — deixa a contagem de layout CONSTANTE
    # entre frames; era a variação dessa contagem que fazia a bandeira/rótulo
    # sumirem e a animação travar antes do fim.
    step_set = set(int(s) for s in steps)
    home_goal_steps = sorted(int(e["step"]) for e in result["events"]
                             if e.get("kind") == "Gol" and e.get("team") == home
                             and int(e.get("step", -1)) in step_set)
    away_goal_steps = sorted(int(e["step"]) for e in result["events"]
                             if e.get("kind") == "Gol" and e.get("team") == away
                             and int(e.get("step", -1)) in step_set)

    # minutagem dos gols: (step, rótulo, cor, altura) — vira TRACE DE TEXTO revelado
    # progressivamente. Por ser dado de trace (e não anotação de layout), a contagem
    # de layout segue constante e nada some/trava.
    goal_info, gi = [], 0
    for e in result["events"]:
        if e.get("kind") != "Gol" or int(e.get("step", -1)) not in step_set:
            continue
        gcolor = home_color if e.get("team") == home else away_color
        goal_info.append((int(e["step"]), e.get("label", ""), gcolor, 103.0 + (gi % 2) * 7.0))
        gi += 1

    def goal_scatter(goal_steps: list, color: str, name: str, cur_step: int) -> go.Scatter:
        xs, ys = [], []
        for gs in goal_steps:
            if gs > cur_step:
                continue
            xs += [gs, gs, None]
            ys += [0, 100, None]
        return go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=color, width=1.5, dash="dot"),
                          hoverinfo="skip", showlegend=False, name=name)

    def goal_labels_scatter(cur_step: int) -> go.Scatter:
        xs, ys, texts, colors = [], [], [], []
        for gstep, glabel, gcolor, gy in goal_info:
            if gstep > cur_step:
                continue
            xs.append(gstep)
            ys.append(gy)
            texts.append(f"⚽ {esc(glabel)}")
            colors.append(gcolor)
        return go.Scatter(x=xs, y=ys, mode="markers+text",
                          marker=dict(size=7, color=colors, line=dict(color="#111611", width=1)),
                          text=texts, textposition="top center", textfont=dict(color=colors, size=10),
                          cliponaxis=False, hoverinfo="skip", showlegend=False, name="gols_min")

    def live_panel(k: int) -> dict:
        # painel central: minuto + placar + probabilidades AO VIVO (a referência principal)
        cur = frame.iloc[k]
        if chart_kind == "avanco":
            probs = (f"<span style='color:{home_color}'>{esc(home)} {float(cur['adv_a']):.1%}</span>"
                     f"    <span style='color:{away_color}'>{esc(away)} {float(cur['adv_b']):.1%}</span>")
        else:
            probs = (f"<span style='color:{home_color}'>{esc(home)} {float(cur['win_a']):.1%}</span>"
                     f"    <span style='color:{DRAW_COLOR}'>Empate {float(cur['draw']):.1%}</span>"
                     f"    <span style='color:{away_color}'>{esc(away)} {float(cur['win_b']):.1%}</span>")
        return dict(x=0.5, y=1.04, xref="paper", yref="paper", align="center",
                    xanchor="center", yanchor="bottom",
                    text=(f"<b>⏱ {esc(cur['label'])}</b>    "
                          f"{esc(home)} <b>{int(cur['score_a'])} x {int(cur['score_b'])}</b> {esc(away)}"
                          f"<br>{probs}"),
                    showarrow=False, bgcolor="rgba(17,22,17,0.94)",
                    bordercolor="rgba(255,207,38,0.5)", borderwidth=1, borderpad=7,
                    font=dict(color="#EDEDED", size=13))

    def annots_at(k: int) -> list:
        return period_annots + [live_panel(k)]  # contagem constante entre frames

    def images_at(k: int) -> list:
        cur = frame.iloc[k]
        imgs = []
        for col, _name, _color, team in series:
            if not team:
                continue
            imgs.append(dict(source=get_bandeira_url(team, bandeiras), xref="x", yref="y",
                             x=float(steps[k]), y=float(cur[col]) * 100, sizex=flag_x, sizey=flag_y,
                             xanchor="center", yanchor="middle", sizing="contain", layer="above"))
        return imgs

    def frame_data(k: int) -> list:
        cur_step = int(steps[k])
        data = [line_scatter(col, color, name, k) for col, name, color, _t in series]
        data.append(cursor_scatter(k))
        data.append(goal_scatter(home_goal_steps, home_color, "gols_casa", cur_step))
        data.append(goal_scatter(away_goal_steps, away_color, "gols_fora", cur_step))
        data.append(goal_labels_scatter(cur_step))
        return data

    # ── estado inicial = 0' + frames (só DADOS de trace; layout com contagem
    #    CONSTANTE: períodos + painel + 2 bandeiras) ──
    fig = go.Figure(data=frame_data(0))
    fig.frames = [
        go.Frame(name=str(k), data=frame_data(k),
                 layout=go.Layout(shapes=period_shapes, annotations=annots_at(k), images=images_at(k)))
        for k in range(n)
    ]

    tick_vals, tick_text = [float(steps[0])], ["0'"]
    for stage, label in STAGE_END_LABEL.items():
        sub = frame[frame["stage"] == stage]
        if not sub.empty:
            tick_vals.append(float(sub.iloc[-1]["step"]))
            tick_text.append(label)
    duration = SPEED_MS.get(speed, 210)

    fig.update_layout(
        height=470, margin=dict(l=35, r=40, t=100, b=140),
        xaxis=dict(tickmode="array", tickvals=tick_vals, ticktext=tick_text,
                   gridcolor="rgba(255,255,255,0.07)",
                   range=[min(steps) - margin_x, max(steps) + margin_x]),
        yaxis=dict(title="Probabilidade", range=[-2, 118], tickmode="array",
                   tickvals=[0, 20, 40, 60, 80, 100],
                   ticktext=["0%", "20%", "40%", "60%", "80%", "100%"],
                   gridcolor="rgba(255,255,255,0.07)"),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#C9D1C9",
        showlegend=False, dragmode=False,  # sem arraste: no celular, o dedo rola a página
        shapes=period_shapes, annotations=annots_at(0), images=images_at(0),
        updatemenus=[dict(type="buttons", direction="left", x=0.0, y=-0.16,
                          xanchor="left", yanchor="top", pad=dict(t=0, r=0), buttons=[
            dict(label="▶ Play", method="animate", args=[None, {
                "frame": {"duration": duration, "redraw": True},
                "transition": {"duration": 0}, "fromcurrent": True, "mode": "immediate"}]),
            dict(label="⏸ Pausar", method="animate", args=[[None], {
                "frame": {"duration": 0, "redraw": False},
                "transition": {"duration": 0}, "mode": "immediate"}])])],
        # slider em linha própria, abaixo dos botões (no celular não se sobrepõem)
        sliders=[dict(active=0, x=0.0, y=-0.42, len=1.0,
                      currentvalue={"visible": False}, steps=[
            dict(label="", method="animate", args=[[str(k)], {
                "frame": {"duration": 0, "redraw": True},
                "transition": {"duration": 0}, "mode": "immediate"}]) for k in range(n)])],
    )
    return fig


def penalty_shootout_chart(result: dict, speed: str) -> go.Figure:
    kicks = result.get("penalty_kicks", [])
    home, away = result["home_team"], result["away_team"]
    max_kicks = max(5, len(kicks))

    def payload(upto: int) -> dict:
        visible = kicks[: upto + 1]
        return {
            "x": [k["kicks_a"] if k["team_code"] == 1 else k["kicks_b"] for k in visible],
            "y": [1 if k["team_code"] == 1 else 0 for k in visible],
            "mode": "markers+text",
            "marker": {"size": 25, "color": ["#68E70F" if k["made"] else "#d64b4b" for k in visible],
                       "line": {"color": "#F1F1F1", "width": 1}},
            "text": ["✓" if k["made"] else "✕" for k in visible],
            "textposition": "middle center", "textfont": {"color": "#111611", "size": 14},
            "hovertext": [f"{home if k['team_code'] == 1 else away}<br>Cobrança {k['kick_index']}"
                          f"<br>{'Gol' if k['made'] else 'Perdeu'}"
                          f"<br>Série: {k['score_a']} x {k['score_b']}" for k in visible],
            "hovertemplate": "%{hovertext}<extra></extra>", "showlegend": False,
        }

    fig = go.Figure(go.Scatter(**payload(-1)))  # começa vazio; ▶ revela cobrança a cobrança
    duration = SPEED_MS.get(speed, 210)
    fig.frames = [go.Frame(name=str(i), data=[go.Scatter(**payload(i))]) for i in range(len(kicks))]
    fig.update_layout(
        height=330, margin=dict(l=100, r=20, t=75, b=55),
        title=dict(text="Disputa de pênaltis", x=0.5, y=0.98, xanchor="center", font=dict(size=15)),
        xaxis=dict(title="Cobrança", range=[0.2, max_kicks + 0.8], tickmode="array",
                   tickvals=list(range(1, max_kicks + 1)), gridcolor="rgba(255,255,255,0.07)"),
        yaxis=dict(range=[-0.6, 1.6], tickmode="array", tickvals=[1, 0],
                   ticktext=[team_with_flag(home), team_with_flag(away)],
                   gridcolor="rgba(255,255,255,0.07)"),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#C9D1C9",
        dragmode=False,
        updatemenus=[dict(type="buttons", direction="left", x=0, y=1.22, buttons=[
            dict(label="▶", method="animate", args=[None, {
                "frame": {"duration": duration, "redraw": True},
                "transition": {"duration": 0}, "fromcurrent": True, "mode": "immediate"}]),
            dict(label="⏸", method="animate", args=[[None], {
                "frame": {"duration": 0, "redraw": False}, "mode": "immediate"}])])],
    )
    return fig


# ───────────────────────── Render (HTML) ─────────────────────────

def inject_minute_css() -> None:
    st.markdown("""
<style>
    h1, h2, h3 { letter-spacing: 0 !important; }
    .block-container { padding-top: 1.35rem !important; max-width: 95% !important; }
    /* Mobile: o gesto vertical rola a página em vez de o gráfico capturar o toque.
       Sobrescreve o touch-action:none que o Plotly coloca na camada de arraste. */
    [data-testid="stPlotlyChart"], .js-plotly-plot, .js-plotly-plot .plotly,
    .js-plotly-plot .draglayer, .js-plotly-plot .nsewdrag,
    .js-plotly-plot .draglayer .drag {
        touch-action: pan-y !important;
    }
    .minute-hero { background: linear-gradient(135deg, rgba(17,22,17,0.98), rgba(3,92,136,0.28));
        border: 1px solid rgba(241,241,241,0.10); border-radius: 8px; padding: 0.95rem 1.1rem; margin-bottom: 0.8rem; }
    .minute-title { color: #E0E4DE; font-family: 'Exo 2', sans-serif; font-size: 2rem; line-height: 1.05;
        font-weight: 900; margin: 0.1rem 0 0.35rem; }
    .minute-subtitle { color: #aeb6ad; font-size: 0.94rem; margin: 0; }
    .minute-scoreboard { background: #111611; border: 1px solid rgba(104,231,15,0.16);
        border-left: 4px solid #68E70F; border-radius: 8px; padding: 1rem; }
    .minute-stage { color: #FFCF26; font-size: 0.78rem; font-weight: 900; text-align: center; text-transform: uppercase; }
    .minute-score-row { align-items: center; display: grid; grid-template-columns: minmax(0,1fr) auto minmax(0,1fr);
        gap: 1rem; margin-top: 0.85rem; }
    .minute-team { align-items: center; color: #F1F1F1; display: flex; flex-direction: column; font-size: 1.05rem;
        font-weight: 900; gap: 0.55rem; line-height: 1.15; min-width: 0; text-align: center; }
    .minute-team-name { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; width: 100%; }
    .minute-flag-large { aspect-ratio: 3/2; border-radius: 6px; box-shadow: 0 8px 24px rgba(0,0,0,0.36);
        object-fit: cover; width: min(100%, 165px); }
    .minute-flag { border-radius: 3px; height: 18px; margin-right: 0.35rem; object-fit: cover;
        vertical-align: middle; width: 27px; }
    .minute-score { color: #68E70F; font-size: 3.4rem; font-weight: 950; line-height: 1; min-width: 140px; text-align: center; }
    .minute-note { color: #c9d1c9; font-size: 0.82rem; margin-top: 0.55rem; text-align: center; }
    .minute-panel { background: #111611; border: 1px solid rgba(241,241,241,0.08); border-radius: 8px; overflow: hidden; }
    .minute-panel-title { align-items: center; background: rgba(255,255,255,0.045); color: #F1F1F1; display: flex;
        font-weight: 900; justify-content: space-between; padding: 0.5rem 0.6rem; }
    .minute-panel-title span { color: #68E70F; font-size: 0.76rem; text-transform: uppercase; }
    .event-row { border-bottom: 1px solid rgba(241,241,241,0.06); color: #e8efe8; display: grid; font-size: 0.8rem;
        gap: 0.45rem; grid-template-columns: 64px minmax(0,1fr) minmax(0,1fr); padding: 0.48rem 0.6rem; }
    .event-row:last-child { border-bottom: none; }
    .event-minute { color: #FFCF26; font-weight: 900; }
    .event-team, .event-kind { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .final-card { background: #111611; border: 1px solid rgba(255,207,38,0.22); border-left: 4px solid #FFCF26;
        border-radius: 8px; padding: 0.85rem; }
    .final-kicker { color: #FFCF26; font-size: 0.72rem; font-weight: 900; text-transform: uppercase; }
    .final-winner { color: #F1F1F1; font-size: 1.35rem; font-weight: 950; line-height: 1.1; margin-top: 0.35rem; }
    .final-row { border-top: 1px solid rgba(241,241,241,0.08); color: #c9d1c9; font-size: 0.82rem;
        margin-top: 0.55rem; padding-top: 0.55rem; }
    @media (max-width: 900px) {
        .minute-score-row { grid-template-columns: 1fr; }
        .minute-score { font-size: 2.5rem; min-width: 0; }
    }
</style>
""", unsafe_allow_html=True)


def render_scoreboard(result: dict | None, point: dict | None, bandeiras: dict[str, str]) -> None:
    if result is None:
        st.markdown("""
<div class="minute-scoreboard">
    <div class="minute-stage">Pronto para simular</div>
    <div class="minute-score-row">
        <div class="minute-team"><div class="minute-flag-large" style="background: rgba(255,255,255,0.06);"></div><span class="minute-team-name">Seleção A</span></div>
        <div class="minute-score">0 x 0</div>
        <div class="minute-team"><div class="minute-flag-large" style="background: rgba(255,255,255,0.06);"></div><span class="minute-team-name">Seleção B</span></div>
    </div>
    <div class="minute-note">Escolha as seleções e clique em Simular partida.</div>
</div>""", unsafe_allow_html=True)
        return

    home, away = result["home_team"], result["away_team"]
    st.markdown(f"""
<div class="minute-scoreboard">
    <div class="minute-stage">{esc(point['stage'])}</div>
    <div class="minute-score-row">
        <div class="minute-team">{flag_img(home, bandeiras, "minute-flag-large")}<span class="minute-team-name">{esc(home)}</span></div>
        <div class="minute-score">{point['score_a']} x {point['score_b']}</div>
        <div class="minute-team">{flag_img(away, bandeiras, "minute-flag-large")}<span class="minute-team-name">{esc(away)}</span></div>
    </div>
    <div class="minute-note">{esc(point['note'])}</div>
</div>""", unsafe_allow_html=True)


def render_events(events: list[dict], home: str, away: str, bandeiras: dict[str, str], limit: int = 8) -> None:
    visible = list(reversed(events[-limit:]))
    rows = []
    for event in visible:
        team = event["team"]
        team_html = f"{flag_img(team, bandeiras)}{esc(team)}" if team in (home, away) else esc(team)
        rows.append(f'<div class="event-row"><div class="event-minute">{esc(event["label"])}</div>'
                    f'<div class="event-team">{team_html}</div>'
                    f'<div class="event-kind">{esc(event["kind"])}</div></div>')
    st.markdown(f'<div class="minute-panel"><div class="minute-panel-title">Eventos '
                f'<span>{len(events)}</span></div>{"".join(rows)}</div>', unsafe_allow_html=True)


def render_final_card(result: dict, bandeiras: dict[str, str]) -> None:
    home, away = result["home_team"], result["away_team"]
    code = result["winner_code"]
    winner = home if code == 1 else away if code == 2 else "Empate"
    winner_flag = flag_img(winner, bandeiras) if winner in (home, away) else ""
    decision = "Pênaltis" if result.get("went_penalties") else \
        "Prorrogação" if result.get("went_extra") else "Tempo normal"
    penalty_row = ""
    if result.get("penalty_kicks"):
        last = result["penalty_kicks"][-1]
        penalty_row = f"<div class='final-row'>Pênaltis: {last['score_a']} x {last['score_b']}</div>"
    st.markdown(
        '<div class="final-card"><div class="final-kicker">Fechamento</div>'
        f'<div class="final-winner">{winner_flag}{esc(winner)}</div>'
        f"<div class='final-row'><b>Placar:</b><br>{esc(home)} {result['score_a']} x {result['score_b']} {esc(away)}</div>"
        f"<div class='final-row'><b>Decisão:</b><br>{esc(decision)}</div>{penalty_row}</div>",
        unsafe_allow_html=True)


# ───────────────────────── Página ─────────────────────────

inject_custom_css()
inject_minute_css()

st.markdown("""
<div class="minute-hero">
    <div class="minute-title">Simulação Minuto a Minuto</div>
    <p class="minute-subtitle">Simule uma partida e clique em <b>▶ Play</b> no gráfico para vê-la
    minuto a minuto: a linha de probabilidade cresce, revelando gols, placar e o tempo até o apito
    final. Mesmo modelo calibrado das páginas Partida e Simulação da Copa.</p>
</div>""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("#### Simulação minuto a minuto")
    modo_mata_mata = st.toggle("Modo mata-mata", value=True, key="minute_knockout_mode")
    speed = st.pills("Velocidade", options=list(SPEED_MS), default="Normal",
                     required=True, key="minute_speed")
    chart_view = st.radio("Visualização do gráfico",
                          options=["Resultado (Vitória/Empate)", "Probabilidade de Avanço"],
                          key="minute_chart_view", disabled=not modo_mata_mata)
    if not modo_mata_mata:
        chart_view = "Resultado (Vitória/Empate)"

params = render_param_sidebar()

try:
    base_df = load_force_dataframe()
    combined_df, _ = build_combined(base_df, params)
except Exception as error:  # noqa: BLE001
    st.error(f"Erro ao carregar dados: {error}")
    st.stop()

team_options = combined_df["Seleção"].tolist()
ensure_selected_teams(team_options)
if team_options:
    if st.session_state.get("minute_home_team") not in team_options:
        st.session_state["minute_home_team"] = team_options[0]
    if st.session_state.get("minute_away_team") not in team_options:
        st.session_state["minute_away_team"] = team_options[1] if len(team_options) > 1 else team_options[0]
    if len(team_options) > 1 and st.session_state["minute_home_team"] == st.session_state["minute_away_team"]:
        st.session_state["minute_away_team"] = next(
            (t for t in team_options if t != st.session_state["minute_home_team"]), team_options[0])

col_home, col_away, col_btn = st.columns([4.2, 4.2, 1.6])
with col_home:
    home_team = st.selectbox("Seleção 1", team_options, key="minute_home_team", format_func=team_with_flag)
with col_away:
    away_team = st.selectbox("Seleção 2", team_options, key="minute_away_team", format_func=team_with_flag)
with col_btn:
    st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
    simulate_button = st.button("Simular partida", type="primary", use_container_width=True,
                                disabled=home_team == away_team)

if home_team == away_team:
    st.info("Escolha duas seleções diferentes para simular.")

home_row = combined_df.loc[combined_df["Seleção"] == home_team].iloc[0]
away_row = combined_df.loc[combined_df["Seleção"] == away_team].iloc[0]
match = compute_match_probabilities(
    force_a=float(home_row["forca_com_offset"]), force_b=float(away_row["forca_com_offset"]),
    media_gols=params.media_gols, usar_dixon_coles=params.usar_dixon_coles,
    rho_dixon_coles=params.rho_dixon_coles,
)
bandeiras_dict = dict(zip(combined_df["Seleção"], combined_df["Link_Bandeira"]))

chart_slot = st.empty()
event_slot = st.empty()


def render_result(result: dict, speed: str, chart_view: str) -> None:
    """O gráfico minuto a minuto é o principal. O resultado final fica recolhido
    num expander — só aparece ao assistir o Play ou ao abrir a ficha."""
    chart_kind = "avanco" if chart_view == "Probabilidade de Avanço" else "resultado"

    with chart_slot.container():
        st.plotly_chart(build_probability_figure(result, bandeiras_dict, speed, chart_kind),
                        width="stretch", config=PLOTLY_CONFIG, key=f"mm_chart_{result['seed']}_{chart_kind}")
        if result.get("went_penalties"):
            st.plotly_chart(penalty_shootout_chart(result, speed), width="stretch",
                            config=PLOTLY_CONFIG, key=f"mm_pen_{result['seed']}")

    final_point = dict(result["points"][-1])
    final_point["stage"] = "Resultado final"
    final_point["note"] = "Resultado final da simulação"
    with event_slot.container():
        with st.expander("🏁 Ver ficha do resultado", expanded=False):
            render_scoreboard(result, final_point, bandeiras_dict)
            render_final_card(result, bandeiras_dict)
            render_events(result["events"], result["home_team"], result["away_team"], bandeiras_dict)


if simulate_button and home_team != away_team:
    seed = int(pd.Timestamp.now().value % 999999937)
    result = simulate_trajectory(
        home_team=home_team, away_team=away_team,
        lambda_a=float(match["lambda_a"]), lambda_b=float(match["lambda_b"]),
        share_a=float(match["share_a"]), knockout=bool(modo_mata_mata),
        usar_dixon_coles=bool(params.usar_dixon_coles), rho=float(params.rho_dixon_coles), seed=seed,
    )
    st.session_state["minute_last_result"] = result
    render_result(result, str(speed), chart_view)
else:
    cached = st.session_state.get("minute_last_result")
    if cached and cached["home_team"] == home_team and cached["away_team"] == away_team \
            and cached["knockout"] == bool(modo_mata_mata):
        render_result(cached, str(speed), chart_view)
    else:
        render_scoreboard(None, None, bandeiras_dict)
