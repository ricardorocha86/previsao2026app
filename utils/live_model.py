from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import poisson

from utils.config import (
    DEFAULT_ELASTICIDADE,
    DEFAULT_MEDIA_GOLS,
    DEFAULT_OFFSET,
    DEFAULT_RHO_DIXON_COLES,
    DEFAULT_USAR_DIXON_COLES,
    DEFAULT_WEIGHT_ELO,
    DEFAULT_WEIGHT_FIFA,
    DEFAULT_WEIGHT_HISTORY,
    DEFAULT_WEIGHT_HOST,
    DEFAULT_WEIGHT_MARKET,
    DEFAULT_WEIGHT_MOMENTUM,
)
from utils.simulador_oficial import dixon_coles_correction, parse_world_cup_score


HOSTS_2026 = {"Estados Unidos", "México", "Canadá"}


@dataclass(frozen=True)
class DefaultModelParams:
    weight_fifa: float = DEFAULT_WEIGHT_FIFA
    weight_market: float = DEFAULT_WEIGHT_MARKET
    weight_elo: float = DEFAULT_WEIGHT_ELO
    weight_momentum: float = DEFAULT_WEIGHT_MOMENTUM
    weight_history: float = DEFAULT_WEIGHT_HISTORY
    weight_host: float = DEFAULT_WEIGHT_HOST
    media_gols: float = DEFAULT_MEDIA_GOLS
    offset: float = DEFAULT_OFFSET
    elasticidade: float = DEFAULT_ELASTICIDADE
    usar_dixon_coles: bool = DEFAULT_USAR_DIXON_COLES
    rho_dixon_coles: float = DEFAULT_RHO_DIXON_COLES


def minmax_scale(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    minimum = numeric.min()
    maximum = numeric.max()
    if pd.isna(minimum) or pd.isna(maximum) or maximum == minimum:
        return pd.Series(0.5, index=series.index, dtype=float)
    return ((numeric - minimum) / (maximum - minimum)).fillna(0.0).astype(float)


def build_default_force_table(dataframe: pd.DataFrame, params: DefaultModelParams | None = None) -> pd.DataFrame:
    params = params or DefaultModelParams()
    result = dataframe.copy()
    result["team_key"] = result["Seleção"].astype(str)
    result["is_host"] = result["Seleção"].isin(HOSTS_2026).astype(int)

    result["fifa_force_01"] = minmax_scale(result.get("FIFA_Current_Points", pd.Series(0, index=result.index)))
    result["elo_force_01"] = minmax_scale(result.get("ELO_Rating", pd.Series(0, index=result.index)))
    result["momentum_force_01"] = minmax_scale(result.get("ELO_Chg_1A", pd.Series(0, index=result.index)))
    result["market_force_01"] = minmax_scale(result.get("Valor_Mercado_Milhoes_EUR", pd.Series(0, index=result.index)))
    result["world_cup_apps_01"] = minmax_scale(result.get("Participações_Copa_Mundo", pd.Series(0, index=result.index)))
    result["world_cup_best_raw"] = result.get(
        "Melhor_Resultado_Copa_Mundo",
        pd.Series("", index=result.index),
    ).map(parse_world_cup_score)
    result["world_cup_history_01"] = (
        0.5 * result["world_cup_apps_01"] + 0.5 * result["world_cup_best_raw"]
    )

    weight_sum = (
        params.weight_fifa
        + params.weight_market
        + params.weight_elo
        + params.weight_momentum
        + params.weight_history
        + params.weight_host
    )
    if weight_sum <= 0:
        result["forca_resultante_01"] = 0.0
    else:
        result["forca_resultante_01"] = (
            params.weight_fifa * result["fifa_force_01"]
            + params.weight_market * result["market_force_01"]
            + params.weight_elo * result["elo_force_01"]
            + params.weight_momentum * result["momentum_force_01"]
            + params.weight_history * result["world_cup_history_01"]
            + params.weight_host * result["is_host"]
        ) / weight_sum

    max_force = float(result["forca_resultante_01"].max())
    if max_force > 0:
        result["forca_resultante_01"] = result["forca_resultante_01"] / max_force

    result["forca_elastica"] = result["forca_resultante_01"] ** params.elasticidade
    result["forca_com_offset"] = params.offset + result["forca_elastica"]
    result = result.sort_values("forca_com_offset", ascending=False).reset_index(drop=True)
    result["rank_forca"] = result.index + 1
    return result


def poisson_matrix(
    lambda_a: float,
    lambda_b: float,
    max_goals: int = 10,
    usar_dixon_coles: bool = DEFAULT_USAR_DIXON_COLES,
    rho_dixon_coles: float = DEFAULT_RHO_DIXON_COLES,
) -> np.ndarray:
    goal_range = np.arange(max_goals + 1)
    probs_a = poisson.pmf(goal_range, lambda_a)
    probs_b = poisson.pmf(goal_range, lambda_b)
    probs_a[-1] += max(0.0, 1.0 - probs_a.sum())
    probs_b[-1] += max(0.0, 1.0 - probs_b.sum())

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
    return matrix / matrix.sum()


def match_probabilities(force_a: float, force_b: float, params: DefaultModelParams) -> dict:
    total_force = force_a + force_b
    share_a = 0.5 if total_force <= 0 else force_a / total_force
    lambda_a = params.media_gols * share_a
    lambda_b = params.media_gols * (1.0 - share_a)
    matrix = poisson_matrix(
        lambda_a=lambda_a,
        lambda_b=lambda_b,
        usar_dixon_coles=params.usar_dixon_coles,
        rho_dixon_coles=params.rho_dixon_coles,
    )
    return {
        "share_a": share_a,
        "lambda_a": lambda_a,
        "lambda_b": lambda_b,
        "matrix": matrix,
        "win_a": float(np.tril(matrix, -1).sum()),
        "draw": float(np.trace(matrix)),
        "win_b": float(np.triu(matrix, 1).sum()),
    }


def simulate_match(
    team_a: str,
    team_b: str,
    strengths: dict[str, float],
    rng: np.random.Generator,
    params: DefaultModelParams,
    knockout: bool = False,
) -> dict:
    probabilities = match_probabilities(strengths[team_a], strengths[team_b], params)
    matrix = probabilities["matrix"]
    flat_index = int(rng.choice(matrix.size, p=matrix.ravel()))
    goals_a, goals_b = np.unravel_index(flat_index, matrix.shape)

    winner = None
    penalty_winner = None
    if goals_a > goals_b:
        winner = team_a
    elif goals_b > goals_a:
        winner = team_b
    elif knockout:
        penalty_winner = team_a if rng.random() < probabilities["share_a"] else team_b
        winner = penalty_winner

    return {
        "team_a": team_a,
        "team_b": team_b,
        "goals_a": int(goals_a),
        "goals_b": int(goals_b),
        "winner": winner,
        "penalty_winner": penalty_winner,
        "lambda_a": float(probabilities["lambda_a"]),
        "lambda_b": float(probabilities["lambda_b"]),
        "win_a": float(probabilities["win_a"]),
        "draw": float(probabilities["draw"]),
        "win_b": float(probabilities["win_b"]),
    }


def new_group_table(groups: dict[str, list[str]], strengths: dict[str, float]) -> dict[str, dict[str, dict]]:
    return {
        group: {
            team: {
                "team": team,
                "group": group,
                "played": 0,
                "points": 0,
                "wins": 0,
                "draws": 0,
                "losses": 0,
                "goals_for": 0,
                "goals_against": 0,
                "goal_diff": 0,
                "fair_play": 0.0,
                "force": strengths.get(team, 0.0),
            }
            for team in teams
        }
        for group, teams in groups.items()
    }


def apply_group_result(table: dict[str, dict], match: dict, rng: np.random.Generator) -> None:
    team_a = match["team_a"]
    team_b = match["team_b"]
    goals_a = match["goals_a"]
    goals_b = match["goals_b"]
    table[team_a]["played"] += 1
    table[team_b]["played"] += 1
    table[team_a]["goals_for"] += goals_a
    table[team_a]["goals_against"] += goals_b
    table[team_b]["goals_for"] += goals_b
    table[team_b]["goals_against"] += goals_a
    table[team_a]["goal_diff"] = table[team_a]["goals_for"] - table[team_a]["goals_against"]
    table[team_b]["goal_diff"] = table[team_b]["goals_for"] - table[team_b]["goals_against"]
    table[team_a]["fair_play"] = float(rng.random())
    table[team_b]["fair_play"] = float(rng.random())

    if goals_a > goals_b:
        table[team_a]["points"] += 3
        table[team_a]["wins"] += 1
        table[team_b]["losses"] += 1
    elif goals_b > goals_a:
        table[team_b]["points"] += 3
        table[team_b]["wins"] += 1
        table[team_a]["losses"] += 1
    else:
        table[team_a]["points"] += 1
        table[team_b]["points"] += 1
        table[team_a]["draws"] += 1
        table[team_b]["draws"] += 1


def rank_group(records: dict[str, dict]) -> list[dict]:
    return sorted(
        records.values(),
        key=lambda item: (
            item["points"],
            item["goal_diff"],
            item["goals_for"],
            item["fair_play"],
            item["force"],
        ),
        reverse=True,
    )


def group_stage_records(group_tables: dict[str, dict[str, dict]]) -> list[dict]:
    records = []
    for group in sorted(group_tables):
        for position, record in enumerate(rank_group(group_tables[group]), start=1):
            item = record.copy()
            item["group_position"] = position
            records.append(item)
    return records


def build_round_of_32(records: list[dict], strengths: dict[str, float]) -> list[dict]:
    firsts = {row["group"]: row for row in records if row["group_position"] == 1}
    seconds = {row["group"]: row for row in records if row["group_position"] == 2}
    thirds = sorted(
        [row for row in records if row["group_position"] == 3],
        key=lambda row: (
            row["points"],
            row["goal_diff"],
            row["goals_for"],
            row["fair_play"],
            strengths[row["team"]],
        ),
        reverse=True,
    )[:8]

    pools = {
        "E": ["A", "B", "C", "D", "F"],
        "I": ["C", "D", "F", "G", "H"],
        "A": ["C", "E", "F", "H", "I"],
        "L": ["E", "H", "I", "J", "K"],
        "D": ["B", "E", "F", "I", "J"],
        "G": ["A", "E", "H", "I", "J"],
        "B": ["E", "F", "G", "I", "J"],
        "K": ["D", "E", "I", "J", "L"],
    }

    third_by_slot = {}
    available = thirds.copy()
    for slot_group in ["E", "I", "A", "L", "D", "G", "B", "K"]:
        for index, candidate in enumerate(available):
            if candidate["group"] in pools[slot_group]:
                third_by_slot[slot_group] = available.pop(index)
                break
        else:
            if available:
                third_by_slot[slot_group] = available.pop(0)

    bracket = [
        seconds.get("A"),
        seconds.get("B"),
        firsts.get("E"),
        third_by_slot.get("E"),
        firsts.get("F"),
        seconds.get("C"),
        firsts.get("C"),
        seconds.get("F"),
        firsts.get("I"),
        third_by_slot.get("I"),
        seconds.get("E"),
        seconds.get("I"),
        firsts.get("A"),
        third_by_slot.get("A"),
        firsts.get("L"),
        third_by_slot.get("L"),
        firsts.get("D"),
        third_by_slot.get("D"),
        firsts.get("G"),
        third_by_slot.get("G"),
        seconds.get("K"),
        seconds.get("L"),
        firsts.get("H"),
        seconds.get("J"),
        firsts.get("B"),
        third_by_slot.get("B"),
        firsts.get("J"),
        seconds.get("H"),
        firsts.get("K"),
        third_by_slot.get("K"),
        seconds.get("D"),
        seconds.get("G"),
    ]
    return [row for row in bracket if row is not None]
