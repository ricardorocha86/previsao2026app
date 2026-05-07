from __future__ import annotations

import os
import re
import sys
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import json
from datetime import datetime as _dt
from scipy.stats import poisson
from openpyxl.styles import Font, Alignment, PatternFill

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experimento_calibracao_mercado import ODDS_PATH, canonical_team_key, load_market_target
from utils.helpers import inject_custom_css
from utils import config as app_config
from utils.simulador_oficial import dixon_coles_correction, parse_world_cup_score
from utils.simulador_oficial import simulate_one_cup_oficial, PoissonMatchSimulator
from utils.simulador_analitico import run_detailed_simulation

DEFAULT_WEIGHT_FIFA = getattr(app_config, "DEFAULT_WEIGHT_FIFA", 0.05)
DEFAULT_WEIGHT_MARKET = getattr(app_config, "DEFAULT_WEIGHT_MARKET", 1.00)
DEFAULT_WEIGHT_ELO = getattr(app_config, "DEFAULT_WEIGHT_ELO", 0.70)
DEFAULT_WEIGHT_MOMENTUM = getattr(app_config, "DEFAULT_WEIGHT_MOMENTUM", 0.30)
DEFAULT_WEIGHT_HISTORY = getattr(app_config, "DEFAULT_WEIGHT_HISTORY", 0.90)
DEFAULT_WEIGHT_HOST = getattr(app_config, "DEFAULT_WEIGHT_HOST", 0.10)
DEFAULT_MEDIA_GOLS = getattr(app_config, "DEFAULT_MEDIA_GOLS", 3.00)
DEFAULT_OFFSET = getattr(app_config, "DEFAULT_OFFSET", 0.13)
DEFAULT_ELASTICIDADE = getattr(app_config, "DEFAULT_ELASTICIDADE", 1.15)
DEFAULT_USAR_DIXON_COLES = getattr(app_config, "DEFAULT_USAR_DIXON_COLES", True)
DEFAULT_RHO_DIXON_COLES = getattr(app_config, "DEFAULT_RHO_DIXON_COLES", -0.13)


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "dataset"


def find_latest_enriched_dataset() -> Path:
    candidates = sorted(DATA_DIR.glob("FIFA_ELO_DadosSeleções_*.xlsx"))
    if candidates:
        return candidates[-1]
    fallback = DATA_DIR / "FIFA_ELO_DadosSeleções_2026-04-15.xlsx"
    return fallback


def minmax_scale(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    minimum = numeric.min()
    maximum = numeric.max()
    if pd.isna(minimum) or pd.isna(maximum) or maximum == minimum:
        return pd.Series(0.5, index=series.index, dtype=float)
    return ((numeric - minimum) / (maximum - minimum)).astype(float)


# parse_world_cup_history_score removido - agora usa utils.simulador_oficial.parse_world_cup_score


@st.cache_data
def load_force_table(dataset_path: str) -> pd.DataFrame:
    df = pd.read_excel(dataset_path)

    required_columns = [
        "Seleção",
        "Grupo",
        "Link_Bandeira",
        "FIFA_Current_Rank",
        "FIFA_Current_Points",
        "ELO_Ranking",
        "ELO_Rating",
        "ELO_Chg_1A",
        "Valor_Mercado_Milhoes_EUR",
        "Participações_Copa_Mundo",
        "Melhor_Resultado_Copa_Mundo",
    ]
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Colunas ausentes na planilha: {', '.join(missing)}")

    result = df.copy()
    result["team_key"] = result["NomeIngles"].map(canonical_team_key)
    
    hosts = ["Estados Unidos", "México", "Canadá"]
    result["is_host"] = result["Seleção"].isin(hosts).astype(int)
    
    result["fifa_force_01"] = minmax_scale(result["FIFA_Current_Points"])
    result["elo_force_01"] = minmax_scale(result["ELO_Rating"])
    result["momentum_force_01"] = minmax_scale(result["ELO_Chg_1A"])
    result["market_force_01"] = minmax_scale(result["Valor_Mercado_Milhoes_EUR"])
    result["world_cup_apps_01"] = minmax_scale(result["Participações_Copa_Mundo"])
    result["world_cup_best_raw"] = result["Melhor_Resultado_Copa_Mundo"].map(parse_world_cup_score)
    result["world_cup_history_01"] = (
        0.5 * result["world_cup_apps_01"] + 0.5 * result["world_cup_best_raw"]
    )

    odds_df = load_market_target(ODDS_PATH)
    result = result.merge(odds_df[["team_key", "market_prob"]], on="team_key", how="left")
    return result


def build_combined_table(
    dataframe: pd.DataFrame,
    weight_fifa: float,
    weight_elo: float,
    weight_momentum: float,
    weight_market: float,
    weight_history: float,
    offset: float,
    elasticidade: float,
    weight_host: float = 0.0,
) -> tuple[pd.DataFrame, float]:
    result = dataframe.copy()
    weight_sum = weight_fifa + weight_elo + weight_momentum + weight_market + weight_history + weight_host

    if weight_sum > 0:
        result["forca_resultante_01"] = (
            weight_fifa * result["fifa_force_01"]
            + weight_elo * result["elo_force_01"]
            + weight_momentum * result["momentum_force_01"]
            + weight_market * result["market_force_01"]
            + weight_history * result["world_cup_history_01"]
            + weight_host * result["is_host"]
        ) / weight_sum
    else:
        result["forca_resultante_01"] = 0.0

    max_force = float(result["forca_resultante_01"].max())
    if max_force > 0:
        result["forca_resultante_01"] = result["forca_resultante_01"] / max_force

    result["forca_elastica"] = result["forca_resultante_01"] ** elasticidade
    result["forca_com_offset"] = offset + result["forca_elastica"]
    
    result = result.sort_values(
        by=["forca_resultante_01", "fifa_force_01", "elo_force_01", "market_force_01"],
        ascending=False,
    ).reset_index(drop=True)
    result["ranking_odds"] = (
        result["market_prob"].rank(method="min", ascending=False).fillna(len(result) + 1).astype(int)
    )
    result.index = result.index + 1
    result.insert(0, "ranking_forca", result.index)
    return result, weight_sum


def poisson_matrix(
    lambda_a: float,
    lambda_b: float,
    max_goals: int = 10,
    usar_dixon_coles: bool = False,
    rho_dixon_coles: float = -0.13,
) -> np.ndarray:
    goal_range = np.arange(max_goals + 1)
    probs_a = poisson.pmf(goal_range, lambda_a)
    probs_b = poisson.pmf(goal_range, lambda_b)

    residual_a = max(0.0, 1.0 - probs_a.sum())
    residual_b = max(0.0, 1.0 - probs_b.sum())
    probs_a[-1] += residual_a
    probs_b[-1] += residual_b

    matrix = np.outer(probs_a, probs_b)
    if usar_dixon_coles:
        for goals_a in range(max_goals + 1):
            for goals_b in range(max_goals + 1):
                matrix[goals_a, goals_b] *= dixon_coles_correction(
                    goals_a,
                    goals_b,
                    lambda_a,
                    lambda_b,
                    rho=rho_dixon_coles,
                )
    matrix /= matrix.sum()
    return matrix


def compute_match_probabilities(
    force_a: float,
    force_b: float,
    media_gols: float,
    max_goals: int = 10,
    usar_dixon_coles: bool = False,
    rho_dixon_coles: float = -0.13,
) -> dict[str, float | np.ndarray]:
    total_force = force_a + force_b
    if total_force <= 0:
        share_a = 0.5
    else:
        share_a = force_a / total_force
    share_b = 1.0 - share_a

    lambda_a = media_gols * share_a
    lambda_b = media_gols * share_b
    matrix = poisson_matrix(
        lambda_a=lambda_a,
        lambda_b=lambda_b,
        max_goals=max_goals,
        usar_dixon_coles=usar_dixon_coles,
        rho_dixon_coles=rho_dixon_coles,
    )

    win_a = 0.0
    draw = 0.0
    win_b = 0.0
    scorelines: list[dict[str, float | int | str]] = []

    for goals_a in range(matrix.shape[0]):
        for goals_b in range(matrix.shape[1]):
            probability = float(matrix[goals_a, goals_b])
            scorelines.append(
                {
                    "placar": f"{goals_a} x {goals_b}",
                    "gols_a": goals_a,
                    "gols_b": goals_b,
                    "probabilidade": probability,
                }
            )
            if goals_a > goals_b:
                win_a += probability
            elif goals_a < goals_b:
                win_b += probability
            else:
                draw += probability

    top_scorelines = (
        pd.DataFrame(scorelines)
        .sort_values(by="probabilidade", ascending=False)
        .head(5)
        .reset_index(drop=True)
    )

    return {
        "share_a": share_a,
        "share_b": share_b,
        "lambda_a": lambda_a,
        "lambda_b": lambda_b,
        "win_a": win_a,
        "draw": draw,
        "win_b": win_b,
        "matrix": matrix,
        "top_scorelines": top_scorelines,
    }


def ensure_selected_teams(team_options: list[str]) -> None:
    """Inicializa os valores padrão no session_state apenas na primeira execução."""
    if not team_options:
        return

    default_home = team_options[0]
    default_away = team_options[1] if len(team_options) > 1 else team_options[0]

    if "explorador_home_team" not in st.session_state or st.session_state["explorador_home_team"] not in team_options:
        st.session_state["explorador_home_team"] = default_home
    if "explorador_away_team" not in st.session_state or st.session_state["explorador_away_team"] not in team_options:
        st.session_state["explorador_away_team"] = default_away


SIM_STAGE_COLUMNS = ["pos_1", "pos_2", "pos_3", "pos_4", "Top32", "Oitavas", "Quartas", "Semifinal", "Final", "Campeao"]


def simulate_match_result(
    rng: np.random.Generator,
    mata_mata: bool,
    match_data: dict[str, float | np.ndarray],
) -> tuple[int, int, int | None]:
    matrix = match_data["matrix"]
    flat_index = int(rng.choice(matrix.size, p=matrix.ravel()))
    goals_a, goals_b = np.unravel_index(flat_index, matrix.shape)

    if goals_a > goals_b:
        return int(goals_a), int(goals_b), 1
    if goals_b > goals_a:
        return int(goals_a), int(goals_b), 2
    if not mata_mata:
        return int(goals_a), int(goals_b), None

    share_a = float(match_data["share_a"])
    winner = 1 if rng.random() < share_a else 2
    return int(goals_a), int(goals_b), winner


def group_sort_key(record: dict) -> tuple[float, ...]:
    return (
        record["points"],
        record["goal_diff"],
        record["goals_for"],
        record["force"],
    )


def simulate_one_cup(
    groups: dict[str, list[str]],
    strengths: dict[str, float],
    rng: np.random.Generator,
    match_cache: dict[tuple[str, str], dict[str, float | np.ndarray]],
) -> dict[str, dict[str, int]]:
    history = {
        team_key: {stage: 0 for stage in SIM_STAGE_COLUMNS}
        for teams in groups.values()
        for team_key in teams
    }
    campaign_records: list[dict] = []

    for group_name, teams in sorted(groups.items()):
        table = {
            team_key: {
                "team_key": team_key,
                "group": group_name,
                "points": 0,
                "goal_diff": 0,
                "goals_for": 0,
                "force": strengths[team_key],
            }
            for team_key in teams
        }

        for i in range(len(teams)):
            for j in range(i + 1, len(teams)):
                team_a = teams[i]
                team_b = teams[j]
                match_data = match_cache[(team_a, team_b)]
                goals_a, goals_b, winner = simulate_match_result(
                    rng=rng,
                    mata_mata=False,
                    match_data=match_data,
                )

                table[team_a]["goal_diff"] += goals_a - goals_b
                table[team_b]["goal_diff"] += goals_b - goals_a
                table[team_a]["goals_for"] += goals_a
                table[team_b]["goals_for"] += goals_b

                if winner == 1:
                    table[team_a]["points"] += 3
                elif winner == 2:
                    table[team_b]["points"] += 3
                else:
                    table[team_a]["points"] += 1
                    table[team_b]["points"] += 1

        ranking = sorted(table.values(), key=group_sort_key, reverse=True)
        for position, record in enumerate(ranking, start=1):
            record["group_position"] = position
            campaign_records.append(record)
            # Registrar a posição no grupo no histórico
            history[record["team_key"]][f"pos_{position}"] = 1

    firsts = [row for row in campaign_records if row["group_position"] == 1]
    seconds = [row for row in campaign_records if row["group_position"] == 2]
    thirds = [row for row in campaign_records if row["group_position"] == 3]
    best_thirds = sorted(thirds, key=group_sort_key, reverse=True)[:8]
    top32 = sorted(firsts + seconds + best_thirds, key=group_sort_key, reverse=True)

    for record in top32:
        history[record["team_key"]]["Top32"] = 1

    current_round = top32
    stage_by_round_size = {
        32: "Oitavas",
        16: "Quartas",
        8: "Semifinal",
        4: "Final",
        2: "Campeao",
    }

    while len(current_round) > 1:
        next_stage = stage_by_round_size[len(current_round)]
        next_round = []
        for idx in range(len(current_round) // 2):
            left = current_round[idx]
            right = current_round[-1 - idx]
            match_data = match_cache[(left["team_key"], right["team_key"])]
            _, _, winner = simulate_match_result(
                rng=rng,
                mata_mata=True,
                match_data=match_data,
            )
            winner_record = left if winner == 1 else right
            history[winner_record["team_key"]][next_stage] = 1
            next_round.append(winner_record)
        current_round = next_round

    return history


def build_match_cache(
    dataframe: pd.DataFrame,
    media_gols: float,
    usar_dixon_coles: bool,
    rho_dixon_coles: float,
) -> dict[tuple[str, str], dict[str, float | np.ndarray]]:
    cache: dict[tuple[str, str], dict[str, float | np.ndarray]] = {}
    rows = dataframe.set_index("team_key")[["forca_com_offset"]]
    team_keys = list(rows.index)
    for team_a in team_keys:
        force_a = float(rows.loc[team_a, "forca_com_offset"])
        for team_b in team_keys:
            if team_a == team_b:
                continue
            force_b = float(rows.loc[team_b, "forca_com_offset"])
            cache[(team_a, team_b)] = compute_match_probabilities(
                force_a=force_a,
                force_b=force_b,
                media_gols=media_gols,
                max_goals=10,
                usar_dixon_coles=usar_dixon_coles,
                rho_dixon_coles=rho_dixon_coles,
            )
    return cache


def build_simulation_table(
    dataframe: pd.DataFrame,
    accumulated: dict[str, dict[str, int]],
    n_sims: int,
) -> pd.DataFrame:
    result = dataframe[
        [
            "Seleção",
            "team_key",
            "forca_com_offset",
        ]
    ].copy()
    
    for stage in SIM_STAGE_COLUMNS:
        result[f"{stage}_pct"] = result["team_key"].map(lambda key: accumulated[key][stage] / n_sims)

    # Ordenar pelos favoritos (Campeão, Final, etc.)
    result = result.sort_values(
        by=["Campeao_pct", "Final_pct", "Semifinal_pct", "forca_com_offset"],
        ascending=False,
    ).reset_index(drop=True)
    
    result.index = result.index + 1
    result.insert(0, "Rank Sim", result.index)
    
    # Remover colunas internas/auxiliares não solicitadas no export final
    # Mas manteremos o necessário para o display abaixo
    return result


_MESES_PT = {
    "janeiro": 1, "fevereiro": 2, "março": 3, "abril": 4,
    "maio": 5, "junho": 6, "julho": 7, "agosto": 8,
    "setembro": 9, "outubro": 10, "novembro": 11, "dezembro": 12,
}


def _parse_date_pt(raw: str) -> str:
    """'Quinta-feira, 11 de junho de 2026' -> '11/06/2026' (sortable in Excel)."""
    m = re.search(r"(\d{1,2})\s+de\s+(\w+)\s+de\s+(\d{4})", raw)
    if m:
        day = int(m.group(1))
        month = _MESES_PT.get(m.group(2).lower(), 0)
        year = int(m.group(3))
        if month:
            return _dt(year, month, day).strftime("%d/%m/%Y")
    return raw


def _extract_local(raw: str) -> str:
    """'Cidade do México, no México — 13h00 no horário local' -> 'Cidade do México, México'."""
    # Remove tudo a partir de ' —' ou ' –' (travessão)
    local = re.split(r"\s*[—–]\s*", raw)[0].strip()
    # Limpar ', no/nos/na/nas PAIS' -> ', PAIS'
    local = re.sub(r",\s*n[oa]s?\s+", ", ", local)
    return local


def _extract_br_time(raw: str) -> str:
    """'(16h00 em Brasília / 18h00 em Praia / ...)' -> '16h00'.
    Also handles '(1h00 de 14 de junho em Brasília ...) -> '01h00'."""
    m = re.search(r"\((\d{1,2}h\d{2})", raw)
    if m:
        t = m.group(1)
        # Normalizar hora de 1 dígito -> 2 dígitos (1h00 -> 01h00)
        parts = t.split("h")
        return f"{int(parts[0]):02d}h{parts[1]}"
    return raw


def generate_group_predictions(
    dataframe: pd.DataFrame, media_gols: float, usar_dixon_coles: bool, rho_dixon_coles: float
) -> pd.DataFrame:
    try:
        with open(DATA_DIR / "calendario_copa_2026.json", "r", encoding="utf-8") as f:
            schedule = json.load(f)
    except FileNotFoundError:
        return pd.DataFrame()
        
    name_map = {
        "República da Coreia": "Coreia do Sul",
        "República Democrática do Congo": "RD do Congo",
        "República Tcheca": "Tcheca"
    }
    
    rows = []
    for match in schedule:
        team_a_str = name_map.get(match["team_a"], match["team_a"])
        team_b_str = name_map.get(match["team_b"], match["team_b"])
        
        # Obter forca se times existirem
        if team_a_str in dataframe["Seleção"].values and team_b_str in dataframe["Seleção"].values:
            forca_a = float(dataframe.loc[dataframe["Seleção"] == team_a_str, "forca_com_offset"].iloc[0])
            forca_b = float(dataframe.loc[dataframe["Seleção"] == team_b_str, "forca_com_offset"].iloc[0])
            
            # A função compute_match_probabilities já usa Poisson, média e rho do slider
            probs = compute_match_probabilities(
                force_a=forca_a, force_b=forca_b, media_gols=media_gols,
                usar_dixon_coles=usar_dixon_coles, rho_dixon_coles=rho_dixon_coles
            )
            
            wa, d, wb = probs["win_a"], probs["draw"], probs["win_b"]
                
            rows.append({
                "Grupo": f"Grupo {match['group']}",
                "Data": _parse_date_pt(match["date"]),
                "Local": _extract_local(match["location_time"]),
                "Horário Brasília": _extract_br_time(match["br_time"]),
                "Seleção A": team_a_str,
                "Vitória A": wa,
                "Empate": d,
                "Vitória B": wb,
                "Seleção B": team_b_str,
                "Força A": forca_a
            })
            
    df_res = pd.DataFrame(rows)
    if not df_res.empty:
        df_res = df_res.sort_values(by=["Grupo", "Força A"], ascending=[True, False]).reset_index(drop=True)
        df_res = df_res.drop(columns=["Força A"])
    return df_res

def simulation_excel_bytes(
    simulation_df: pd.DataFrame,
    info_df: pd.DataFrame,
    matches_df: pd.DataFrame = None,
    extra_sheets: dict[str, pd.DataFrame] | None = None,
) -> bytes:
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        simulation_df.to_excel(writer, sheet_name="Simulações", index=False)
        info_df.to_excel(writer, sheet_name="Parâmetros", index=False)
        
        if matches_df is not None and not matches_df.empty:
            # Formatar percentuais para exibicao sem converter pra string pra não estragar
            # Mas pandas to_excel precisa ou da string ou que apliquemos style.
            export_matches = matches_df.copy()
            for col in ["Vitória A", "Empate", "Vitória B"]:
                export_matches[col] = export_matches[col].apply(lambda x: f"{x:.1%}")
                
            export_matches.to_excel(writer, sheet_name="Previsão Jogos", index=False)
            workbook = writer.book
            worksheet = writer.sheets["Previsão Jogos"]
            
            header_fill = PatternFill(start_color="1A1A2E", end_color="1A1A2E", fill_type="solid")
            header_font = Font(color="FFFFFF", bold=True)
            center_align = Alignment(horizontal="center", vertical="center")
            
            # Novo esquema de cores
            base_fill = PatternFill(start_color="EBF4FA", end_color="EBF4FA", fill_type="solid") # Azul bem clarinho para toda a planilha
            win_fill = PatternFill(start_color="CCE5FF", end_color="CCE5FF", fill_type="solid") # Azul um pouco mais forte para vitórias
            draw_fill = PatternFill(start_color="E8E8E8", end_color="E8E8E8", fill_type="solid") # Cinza claro para empate
            
            for col in range(1, 10):
                cell = worksheet.cell(row=1, column=col)
                cell.fill = header_fill
                cell.font = header_font
                cell.alignment = center_align

            worksheet.column_dimensions['A'].width = 10   # Grupo
            worksheet.column_dimensions['B'].width = 14   # Data (dd/mm/yyyy)
            worksheet.column_dimensions['C'].width = 35   # Local (cidade, país)
            worksheet.column_dimensions['D'].width = 16   # Horário Brasília (HHhMM)
            worksheet.column_dimensions['E'].width = 20   # Seleção A
            worksheet.column_dimensions['F'].width = 12   # Vitória A
            worksheet.column_dimensions['G'].width = 12   # Empate
            worksheet.column_dimensions['H'].width = 12   # Vitória B
            worksheet.column_dimensions['I'].width = 20   # Seleção B
            
            for row in range(2, len(export_matches) + 2):
                for col in range(1, 10):
                    # Preenchimento base (azul claro)
                    worksheet.cell(row=row, column=col).fill = base_fill
                    
                    if col in [1, 2, 3, 4, 6, 7, 8]:
                        worksheet.cell(row=row, column=col).alignment = center_align
                
                worksheet.cell(row=row, column=5).alignment = Alignment(horizontal="right", vertical="center")
                worksheet.cell(row=row, column=5).font = Font(bold=True)
                worksheet.cell(row=row, column=9).alignment = Alignment(horizontal="left", vertical="center")
                worksheet.cell(row=row, column=9).font = Font(bold=True)
                
                worksheet.cell(row=row, column=6).fill = win_fill
                worksheet.cell(row=row, column=7).fill = draw_fill
                worksheet.cell(row=row, column=8).fill = win_fill

        if extra_sheets:
            for sheet_name, sheet_df in extra_sheets.items():
                safe_name = sheet_name[:31]
                sheet_df.to_excel(writer, sheet_name=safe_name, index=False)

    buffer.seek(0)
    return buffer.getvalue()


def run_simulation_progressive(
    dataframe: pd.DataFrame,
    media_gols: float,
    n_sims: int,
    usar_dixon_coles: bool,
    rho_dixon_coles: float,
    tipo_chaveamento: str = "Sorteio Oficial",
    chunk_size: int = 1000,
) -> pd.DataFrame:
    groups = dataframe.groupby("Grupo")["team_key"].apply(list).to_dict()
    strengths = dict(zip(dataframe["team_key"], dataframe["forca_com_offset"]))
    match_cache = build_match_cache(
        dataframe=dataframe,
        media_gols=media_gols,
        usar_dixon_coles=usar_dixon_coles,
        rho_dixon_coles=rho_dixon_coles,
    )
    rng = np.random.default_rng()
    match_simulator = PoissonMatchSimulator(match_cache=match_cache, strengths=strengths)
    
    # Mapeamento para acelerar a acumulação usando numpy (muito mais rápido que dicts aninhados)
    team_keys = dataframe["team_key"].tolist()
    team_idx_map = {key: i for i, key in enumerate(team_keys)}
    stage_idx_map = {stage: i for i, stage in enumerate(SIM_STAGE_COLUMNS)}
    # Matriz (Times x Estágios) iniciada com zeros
    accum_np = np.zeros((len(team_keys), len(SIM_STAGE_COLUMNS)), dtype=np.int32)
    
    completed = 0
    status_placeholder = st.empty()
    
    with st.spinner("Processando simulação...", show_time=True):
        while completed < n_sims:
            current_chunk = min(chunk_size, n_sims - completed)
            for _ in range(current_chunk):
                # Se chaveamento aleatório, sorteia novos grupos a cada simulação
                if "Aleatório" in tipo_chaveamento:
                    from utils.simulador_oficial import randomizar_grupos
                    groups_uso = randomizar_grupos(groups, strengths)
                else:
                    groups_uso = groups
                    
                history = simulate_one_cup_oficial(
                    groups=groups_uso,
                    strengths=strengths,
                    rng=rng,
                    match_simulator=match_simulator,
                )
                for team_key, stages in history.items():
                    t_idx = team_idx_map[team_key]
                    for stage, value in stages.items():
                        if value:
                            s_idx = stage_idx_map[stage]
                            accum_np[t_idx, s_idx] += 1

            completed += current_chunk
            status_placeholder.markdown(f"**Progresso:** {completed:,} / {n_sims:,} copas")

    # Converter a matriz numpy de volta para o formato de dicionário aninhado que build_simulation_table espera
    accumulated = {
        team_key: {
            stage: int(accum_np[team_idx_map[team_key], stage_idx_map[stage]])
            for stage in SIM_STAGE_COLUMNS
        }
        for team_key in team_keys
    }
    
    status_placeholder.success(f"Simulação concluída: {n_sims:,} copas!")
    return build_simulation_table(dataframe=dataframe, accumulated=accumulated, n_sims=n_sims)


def run_complete_simulation_progressive(
    dataframe: pd.DataFrame,
    media_gols: float,
    n_sims: int,
    usar_dixon_coles: bool,
    rho_dixon_coles: float,
    tipo_chaveamento: str = "Sorteio Oficial",
    chunk_size: int = 10000,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    strengths = dict(zip(dataframe["team_key"], dataframe["forca_com_offset"]))
    match_cache = build_match_cache(
        dataframe=dataframe,
        media_gols=media_gols,
        usar_dixon_coles=usar_dixon_coles,
        rho_dixon_coles=rho_dixon_coles,
    )
    match_simulator = PoissonMatchSimulator(match_cache=match_cache, strengths=strengths)
    status_placeholder = st.empty()

    def update_progress(completed: int, total: int) -> None:
        status_placeholder.markdown(f"**Progresso detalhado:** {completed:,} / {total:,} copas")

    with st.spinner("Processando simulação completa...", show_time=True):
        detailed_result = run_detailed_simulation(
            dataframe=dataframe,
            n_sims=n_sims,
            match_simulator=match_simulator,
            strengths=strengths,
            tipo_chaveamento=tipo_chaveamento,
            chunk_size=chunk_size,
            progress_callback=update_progress,
        )

    status_placeholder.success(f"Simulação completa concluída: {n_sims:,} copas!")
    sim_table = build_simulation_table(
        dataframe=dataframe,
        accumulated=detailed_result["accumulated"],
        n_sims=n_sims,
    )
    return sim_table, detailed_result["tables"]


inject_custom_css()

st.markdown("## Simulação Copa do Mundo 2026")
st.markdown(
    """
<p style="font-size: 1rem; margin-bottom: 1.5rem;">
Combine os índices normalizados de <b>FIFA</b>,
<b>ELO</b>, o <b>momento recente</b>, <b>valor de mercado</b>,
o <b>histórico em Copas</b>, ajuste <b>elasticidade</b> e <b>offset</b> e veja como isso altera a probabilidade de um jogo.
</p>
""",
    unsafe_allow_html=True,
)

dataset_path = find_latest_enriched_dataset()

try:
    base_df = load_force_table(str(dataset_path))
except Exception as error:
    st.error(f"Erro ao carregar a base enriquecida: {error}")
    st.stop()

with st.sidebar:
    st.markdown("#### Composição do Indicador de Força")
    col1, col2 = st.columns(2)
    with col1:
        weight_fifa = st.slider("FIFA", min_value=0.0, max_value=1.0, value=DEFAULT_WEIGHT_FIFA, step=0.01)
    with col2:
        weight_market = st.slider("Mercado", min_value=0.0, max_value=1.0, value=DEFAULT_WEIGHT_MARKET, step=0.01)
        
    col3, col4 = st.columns(2)
    with col3:
        weight_elo = st.slider("ELO", min_value=0.0, max_value=1.0, value=DEFAULT_WEIGHT_ELO, step=0.01)
    with col4:
        weight_momentum = st.slider("Momento", min_value=0.0, max_value=1.0, value=DEFAULT_WEIGHT_MOMENTUM, step=0.01)

    col5, col6 = st.columns(2)
    with col5:
        weight_history = st.slider("Histórico Copas", min_value=0.0, max_value=1.0, value=DEFAULT_WEIGHT_HISTORY, step=0.01)
    with col6:
        weight_host = st.slider("Anfitrião (Sede)", min_value=0.0, max_value=1.0, value=DEFAULT_WEIGHT_HOST, step=0.01)

    st.markdown("---")
    st.markdown("#### Parâmetros do Modelo")
    
    media_gols = st.slider("Média de gols da partida", min_value=0.5, max_value=5.0, value=DEFAULT_MEDIA_GOLS, step=0.05)

    col7, col8 = st.columns(2)
    with col7:
        offset = st.slider("Offset", min_value=0.0, max_value=1.0, value=DEFAULT_OFFSET, step=0.01)
    with col8:
        elasticidade = st.slider("Elasticidade", min_value=0.1, max_value=5.0, value=DEFAULT_ELASTICIDADE, step=0.01)

    col9, col10 = st.columns([2, 3])
    with col9:
        st.markdown("<div style='margin-top: 2rem;'></div>", unsafe_allow_html=True)
        usar_dixon_coles = st.toggle("Dixon-Coles", value=DEFAULT_USAR_DIXON_COLES)
    with col10:
        rho_dixon_coles = st.slider("Parâmetro rho", min_value=-0.30, max_value=0.00, value=DEFAULT_RHO_DIXON_COLES, step=0.01, disabled=not usar_dixon_coles)

    st.markdown("---")
    st.markdown("#### Parâmetros da Simulação")
    n_sims = st.slider("Nº de Copas", min_value=10000, max_value=1000000, value=10000, step=10000)
    tipo_chaveamento = st.radio("", ["Sorteio Oficial", "Sorteio Aleatório"], horizontal=True)
    tipo_simulacao = st.radio("Tipo de simulação", ["Básica", "Completa"], horizontal=True)
    run_simulation = st.button("Rodar simulação", width='stretch')
    if "explorador_sim_excel_bytes" not in st.session_state:
        st.session_state["explorador_sim_excel_bytes"] = None
    st.download_button(
        label="Baixar Excel",
        data=st.session_state["explorador_sim_excel_bytes"] if st.session_state["explorador_sim_excel_bytes"] else "",
        file_name="simulacao_explorador_forca.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        width='stretch',
        disabled=st.session_state["explorador_sim_excel_bytes"] is None,
    )

combined_df, weight_sum = build_combined_table(
    dataframe=base_df,
    weight_fifa=weight_fifa,
    weight_elo=weight_elo,
    weight_momentum=weight_momentum,
    weight_market=weight_market,
    weight_history=weight_history,
    offset=offset,
    elasticidade=elasticidade,
    weight_host=weight_host,
)

if weight_sum <= 0:
    st.warning("A soma dos pesos está zerada. Ajuste ao menos um peso para construir a força resultante.")

effective_fifa = weight_fifa / weight_sum if weight_sum > 0 else 0.0
effective_elo = weight_elo / weight_sum if weight_sum > 0 else 0.0
effective_momentum = weight_momentum / weight_sum if weight_sum > 0 else 0.0
effective_market = weight_market / weight_sum if weight_sum > 0 else 0.0
effective_history = weight_history / weight_sum if weight_sum > 0 else 0.0
effective_host = weight_host / weight_sum if weight_sum > 0 else 0.0

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

st.dataframe(
    display_table,
    width='stretch',
    height=520,
    column_config={
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

st.markdown("---")
st.markdown("### Partida")
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
        max_gols_display = 6
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

st.markdown("---")
st.markdown("### Simulação")

if "explorador_sim_display" not in st.session_state:
    st.session_state["explorador_sim_display"] = None
if "explorador_detailed_tables" not in st.session_state:
    st.session_state["explorador_detailed_tables"] = None

if run_simulation:
    detailed_tables = None
    if tipo_simulacao == "Completa":
        sim_table, detailed_tables = run_complete_simulation_progressive(
            dataframe=combined_df,
            media_gols=media_gols,
            n_sims=int(n_sims),
            usar_dixon_coles=usar_dixon_coles,
            rho_dixon_coles=rho_dixon_coles,
            tipo_chaveamento=tipo_chaveamento,
            chunk_size=10000,
        )
    else:
        sim_table = run_simulation_progressive(
            dataframe=combined_df,
            media_gols=media_gols,
            n_sims=int(n_sims),
            usar_dixon_coles=usar_dixon_coles,
            rho_dixon_coles=rho_dixon_coles,
            tipo_chaveamento=tipo_chaveamento,
            chunk_size=10000,
        )

    sim_display = sim_table[
        [
            "Seleção",
            "pos_1_pct",
            "pos_2_pct",
            "pos_3_pct",
            "pos_4_pct",
            "Top32_pct",
            "Oitavas_pct",
            "Quartas_pct",
            "Semifinal_pct",
            "Final_pct",
            "Campeao_pct",
        ]
    ].rename(
        columns={
            "pos_1_pct": "1º Grupo",
            "pos_2_pct": "2º Grupo",
            "pos_3_pct": "3º Grupo",
            "pos_4_pct": "4º Grupo",
            "Top32_pct": "Top 32",
            "Oitavas_pct": "Oitavas",
            "Quartas_pct": "Quartas",
            "Semifinal_pct": "Semi",
            "Final_pct": "Final",
            "Campeao_pct": "Campeão",
        }
    )

    info_df = pd.DataFrame(
        [
            {"Parametro": "Etapa", "Valor": "Pré-Torneio"},
            {"Parametro": "Peso FIFA", "Valor": weight_fifa},
            {"Parametro": "Peso ELO", "Valor": weight_elo},
            {"Parametro": "Peso Momento", "Valor": weight_momentum},
            {"Parametro": "Peso Mercado", "Valor": weight_market},
            {"Parametro": "Peso Histórico Copas", "Valor": weight_history},
            {"Parametro": "Peso Anfitrião", "Valor": weight_host},
            {"Parametro": "Offset", "Valor": offset},
            {"Parametro": "Elasticidade", "Valor": elasticidade},
            {"Parametro": "Média de gols", "Valor": media_gols},
            {"Parametro": "Usar Dixon-Coles", "Valor": usar_dixon_coles},
            {"Parametro": "Rho Dixon-Coles", "Valor": rho_dixon_coles},
            {"Parametro": "Número de Copas", "Valor": int(n_sims)},
            {"Parametro": "Tipo de Simulação", "Valor": tipo_simulacao},
        ]
    )

    matches_df = generate_group_predictions(
        dataframe=combined_df, media_gols=media_gols,
        usar_dixon_coles=usar_dixon_coles, rho_dixon_coles=rho_dixon_coles
    )

    extra_sheets = None
    if detailed_tables:
        extra_sheets = {
            "Finais": detailed_tables["finais"],
            "Brasil 1o Top32": detailed_tables["brasil_1o_grupo_top32"],
            "Brasil 2o Top32": detailed_tables["brasil_2o_grupo_top32"],
            "Brasil 3o Top32": detailed_tables["brasil_3o_grupo_top32"],
            "Eliminadores Brasil": detailed_tables["eliminadores_brasil"],
            "Carrascos Brasil": detailed_tables["eliminadores_brasil_agrupado"],
            "Titulo Cond Brasil": detailed_tables["titulo_condicional_brasil"],
            "Impacto Pos Grupo": detailed_tables["impacto_posicao_grupo"],
            "Bottom16 Surpresa": detailed_tables["bottom16_surpresa"],
            "Bottom16 Lista": detailed_tables["bottom16_lista"],
            "Semifinais": detailed_tables["semifinais"],
        }

    st.session_state["explorador_sim_excel_bytes"] = simulation_excel_bytes(
        sim_display,
        info_df,
        matches_df,
        extra_sheets=extra_sheets,
    )
    
    if int(n_sims) >= 100000:
        import datetime
        os.makedirs(BASE_DIR / "resultados", exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%d-%m-%Y")
        filepath = BASE_DIR / "resultados" / f"simulacao_previsao_esportiva_{tipo_simulacao}_Pre-Torneio_{timestamp}.xlsx"
        with open(filepath, "wb") as f:
            f.write(st.session_state["explorador_sim_excel_bytes"])
            
    st.session_state["explorador_sim_display"] = sim_display
    st.session_state["explorador_detailed_tables"] = detailed_tables
    st.rerun()

if st.session_state.get("explorador_sim_display") is not None:
    st.dataframe(
        st.session_state["explorador_sim_display"],
        width='stretch',
        height=520,
        column_config={
            "1º Grupo": st.column_config.NumberColumn(format="%.3f"),
            "2º Grupo": st.column_config.NumberColumn(format="%.3f"),
            "3º Grupo": st.column_config.NumberColumn(format="%.3f"),
            "4º Grupo": st.column_config.NumberColumn(format="%.3f"),
            "Top 32": st.column_config.NumberColumn(format="%.3f"),
            "Oitavas": st.column_config.NumberColumn(format="%.3f"),
            "Quartas": st.column_config.NumberColumn(format="%.3f"),
            "Semi": st.column_config.NumberColumn(format="%.3f"),
            "Final": st.column_config.NumberColumn(format="%.3f"),
            "Campeão": st.column_config.NumberColumn(format="%.3f"),
        },
    )

detailed_tables = st.session_state.get("explorador_detailed_tables")
if detailed_tables:
    def show_table(df: pd.DataFrame, height: int = 360) -> None:
        probability_columns = [
            col
            for col in df.columns
            if col.startswith("Prob") or col.startswith("Titulo ") or col == "Prob titulo"
        ]
        st.dataframe(
            df,
            width='stretch',
            height=height,
            column_config={
                col: st.column_config.NumberColumn(format="%.3f")
                for col in probability_columns
            },
        )

    st.markdown("### Análises da simulação completa")
    st.markdown("#### Finais")
    st.markdown("##### Finais mais prováveis")
    show_table(detailed_tables["finais"], height=360)

    st.markdown("#### Brasil")
    st.markdown("##### Primeiro mata-mata do Brasil por posição no grupo")
    col_brasil_1, col_brasil_2, col_brasil_3 = st.columns(3)
    with col_brasil_1:
        st.markdown("###### Brasil 1º do grupo")
        show_table(detailed_tables["brasil_1o_grupo_top32"], height=320)
    with col_brasil_2:
        st.markdown("###### Brasil 2º do grupo")
        show_table(detailed_tables["brasil_2o_grupo_top32"], height=320)
    with col_brasil_3:
        st.markdown("###### Brasil 3º do grupo")
        show_table(detailed_tables["brasil_3o_grupo_top32"], height=320)

    st.markdown("##### Eliminações do Brasil")
    col_elim_1, col_elim_2 = st.columns([1, 2])
    with col_elim_1:
        st.markdown("###### Top carrascos")
        show_table(detailed_tables["eliminadores_brasil_agrupado"], height=320)
    with col_elim_2:
        st.markdown("###### Carrascos por fase")
        show_table(detailed_tables["eliminadores_brasil"], height=320)

    st.markdown("#### Condicionais")
    col_cond_1, col_cond_2 = st.columns([1, 2])
    with col_cond_1:
        st.markdown("##### Título do Brasil por condição")
        show_table(detailed_tables["titulo_condicional_brasil"], height=360)
    with col_cond_2:
        st.markdown("##### Impacto da posição no grupo")
        show_table(detailed_tables["impacto_posicao_grupo"], height=360)

    st.markdown("#### Bottom 16")
    col_bottom_1, col_bottom_2 = st.columns([2, 1])
    with col_bottom_1:
        st.markdown("##### Pelo menos uma bottom 16 por fase")
        show_table(detailed_tables["bottom16_surpresa"], height=260)
    with col_bottom_2:
        st.markdown("##### Lista pelo indicador atual")
        show_table(detailed_tables["bottom16_lista"], height=260)

    st.markdown("#### Semifinais")
    st.markdown("##### Quartetos de semifinalistas mais prováveis")
    show_table(detailed_tables["semifinais"], height=520)
