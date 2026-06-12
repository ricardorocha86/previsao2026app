from __future__ import annotations

from collections import Counter, defaultdict
from typing import Callable

import numpy as np
import pandas as pd

from utils.simulador_oficial import (
    BaseMatchSimulator,
    build_official_round_of_32,
    ordenar_grupo_oficial,
    randomizar_grupos,
)


SIM_STAGE_COLUMNS = [
    "pos_1",
    "pos_2",
    "pos_3",
    "pos_4",
    "Top32",
    "Oitavas",
    "Quartas",
    "Semifinal",
    "Final",
    "Campeao",
]

ADVANCE_STAGE_BY_ROUND_SIZE = {
    32: "Oitavas",
    16: "Quartas",
    8: "Semifinal",
    4: "Final",
    2: "Campeao",
}

MATCH_STAGE_BY_ROUND_SIZE = {
    32: "Top 32",
    16: "Oitavas",
    8: "Quartas",
    4: "Semifinal",
    2: "Final",
}

DISPLAY_STAGE_BY_HISTORY = {
    "Top32": "Top 32",
    "Oitavas": "Oitavas",
    "Quartas": "Quartas",
    "Semifinal": "Semifinal",
    "Final": "Final",
    "Campeao": "Campeao",
}

BRAZIL_KEY = "brazil"


def _ordered_pair(team_a: str, team_b: str) -> tuple[str, str]:
    return tuple(sorted((team_a, team_b)))


def _format_pair(pair: tuple[str, str], names: dict[str, str]) -> str:
    return " x ".join(names.get(team, team) for team in pair)


def _team_name_column(dataframe: pd.DataFrame) -> str:
    for column in dataframe.columns:
        if str(column).startswith("Sele"):
            return column
    return "team_key"


def _same_official_top32_bracket(
    firsts: dict[str, dict],
    seconds: dict[str, dict],
    best_thirds: list[dict],
) -> list[dict]:
    return build_official_round_of_32(firsts, seconds, best_thirds)


def _find_opponent(current_round: list[dict], team_key: str) -> str | None:
    for idx in range(0, len(current_round), 2):
        left = current_round[idx]["team_key"]
        right = current_round[idx + 1]["team_key"]
        if left == team_key:
            return right
        if right == team_key:
            return left
    return None


def simulate_one_cup_analytics(
    groups: dict[str, list[str]],
    strengths: dict[str, float],
    rng: np.random.Generator,
    match_simulator: BaseMatchSimulator,
    tracked_team: str = BRAZIL_KEY,
) -> dict:
    history = {
        team_key: {stage: 0 for stage in SIM_STAGE_COLUMNS}
        for teams in groups.values()
        for team_key in teams
    }
    records = []
    group_positions = {}

    for group_name, teams in sorted(groups.items()):
        table = {
            team_key: {
                "team_key": team_key,
                "group": group_name,
                "points": 0,
                "goal_diff": 0,
                "goals_for": 0,
                "fair_play": 0,
                "confrontos": {},
            }
            for team_key in teams
        }
        for idx_a, team_a in enumerate(teams):
            for team_b in teams[idx_a + 1 :]:
                goals_a, goals_b, winner = match_simulator.simulate_match(
                    team_a,
                    team_b,
                    rng,
                    False,
                )
                table[team_a]["confrontos"][team_b] = (goals_a, goals_b)
                table[team_b]["confrontos"][team_a] = (goals_b, goals_a)
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

        ranking = ordenar_grupo_oficial(list(table.values()), strengths, rng)
        for position, record in enumerate(ranking, start=1):
            record["group_position"] = position
            records.append(record)
            team_key = record["team_key"]
            group_positions[team_key] = position
            history[team_key][f"pos_{position}"] = 1

    firsts = {record["group"]: record for record in records if record["group_position"] == 1}
    seconds = {record["group"]: record for record in records if record["group_position"] == 2}
    thirds = sorted(
        [record for record in records if record["group_position"] == 3],
        key=lambda item: (
            item["points"],
            item["goal_diff"],
            item["goals_for"],
            item["fair_play"],
            strengths[item["team_key"]],
        ),
        reverse=True,
    )[:8]

    current_round = _same_official_top32_bracket(firsts, seconds, thirds)
    first_knockout_opponent = _find_opponent(current_round, tracked_team)

    for record in current_round:
        history[record["team_key"]]["Top32"] = 1

    finalists: tuple[str, str] | None = None
    tracked_eliminator: tuple[str, str] | None = None
    tracked_opponents_by_stage = {}

    while len(current_round) > 1:
        round_size = len(current_round)
        if round_size == 2:
            finalists = _ordered_pair(
                current_round[0]["team_key"],
                current_round[1]["team_key"],
            )

        match_stage = MATCH_STAGE_BY_ROUND_SIZE[round_size]
        next_stage = ADVANCE_STAGE_BY_ROUND_SIZE[round_size]
        next_round = []
        for idx in range(0, round_size, 2):
            left = current_round[idx]
            right = current_round[idx + 1]
            left_key = left["team_key"]
            right_key = right["team_key"]
            if left_key == tracked_team:
                tracked_opponents_by_stage[match_stage] = right_key
            elif right_key == tracked_team:
                tracked_opponents_by_stage[match_stage] = left_key
            _, _, winner = match_simulator.simulate_match(
                left_key,
                right_key,
                rng,
                True,
            )
            winner_record = left if winner == 1 else right
            loser_record = right if winner == 1 else left
            winner_key = winner_record["team_key"]
            loser_key = loser_record["team_key"]
            history[winner_key][next_stage] = 1
            if loser_key == tracked_team:
                tracked_eliminator = (winner_key, match_stage)
            next_round.append(winner_record)
        current_round = next_round

    champion = current_round[0]["team_key"]
    return {
        "history": history,
        "group_positions": group_positions,
        "first_knockout_opponent": first_knockout_opponent,
        "tracked_eliminator": tracked_eliminator,
        "tracked_opponents_by_stage": tracked_opponents_by_stage,
        "finalists": finalists,
        "champion": champion,
    }


def run_detailed_simulation(
    dataframe: pd.DataFrame,
    n_sims: int,
    match_simulator: BaseMatchSimulator,
    strengths: dict[str, float],
    tipo_chaveamento: str = "Sorteio Oficial",
    chunk_size: int = 10_000,
    progress_callback: Callable[[int, int], None] | None = None,
    tracked_team: str = BRAZIL_KEY,
) -> dict:
    groups = dataframe.groupby("Grupo")["team_key"].apply(list).to_dict()
    team_keys = dataframe["team_key"].tolist()
    team_idx_map = {key: idx for idx, key in enumerate(team_keys)}
    stage_idx_map = {stage: idx for idx, stage in enumerate(SIM_STAGE_COLUMNS)}
    name_column = _team_name_column(dataframe)
    team_names = dict(zip(dataframe["team_key"], dataframe[name_column]))

    # Times ordenados do mais fraco para o mais forte pelo indicador de força.
    # bottom_16 = as 16 mais fracas (zebra pesada); mini_zebra = as 32 mais fracas (mini-zebra).
    _weak_first = dataframe.sort_values("forca_com_offset", ascending=True)["team_key"].tolist()
    bottom_16_order = _weak_first[:16]
    mini_zebra_order = _weak_first[:32]
    bottom_16 = set(bottom_16_order)
    mini_zebra = set(mini_zebra_order)

    accum_np = np.zeros((len(team_keys), len(SIM_STAGE_COLUMNS)), dtype=np.int32)
    finals_counter: Counter[tuple[str, str]] = Counter()
    tracked_first_ko_by_group_position = {position: Counter() for position in [1, 2, 3]}
    tracked_first_ko_base_by_group_position = Counter()
    tracked_opponents_by_stage = defaultdict(Counter)
    tracked_stage_base = Counter()
    tracked_eliminators: Counter[tuple[str, str]] = Counter()
    tracked_conditions = defaultdict(lambda: {"den": 0, "champ": 0})
    title_by_group_position = defaultdict(lambda: {"den": 0, "champ": 0})
    bottom_16_stage_counter = Counter()
    mini_zebra_stage_counter = Counter()

    rng = np.random.default_rng()
    completed = 0

    while completed < n_sims:
        current_chunk = min(chunk_size, n_sims - completed)
        for _ in range(current_chunk):
            if "Aleatório" in tipo_chaveamento or "Aleatorio" in tipo_chaveamento:
                groups_in_use = randomizar_grupos(groups, strengths)
            else:
                groups_in_use = groups

            result = simulate_one_cup_analytics(
                groups=groups_in_use,
                strengths=strengths,
                rng=rng,
                match_simulator=match_simulator,
                tracked_team=tracked_team,
            )
            history = result["history"]
            champion = result["champion"]

            for team_key, stages in history.items():
                team_idx = team_idx_map[team_key]
                for stage, value in stages.items():
                    if value:
                        accum_np[team_idx, stage_idx_map[stage]] += 1

            finalists = result["finalists"]
            if finalists is not None:
                finals_counter[finalists] += 1

            for stage, opponent in result["tracked_opponents_by_stage"].items():
                tracked_stage_base[stage] += 1
                tracked_opponents_by_stage[stage][opponent] += 1

            tracked_position = result["group_positions"].get(tracked_team)
            if tracked_position is not None:
                condition_key = f"Brasil pos_{tracked_position} no grupo"
                tracked_conditions[condition_key]["den"] += 1
                if champion == tracked_team:
                    tracked_conditions[condition_key]["champ"] += 1

            opponent = result["first_knockout_opponent"]
            if tracked_position in tracked_first_ko_by_group_position and opponent:
                tracked_first_ko_base_by_group_position[tracked_position] += 1
                tracked_first_ko_by_group_position[tracked_position][opponent] += 1

            for stage in ["Top32", "Oitavas", "Quartas", "Semifinal", "Final"]:
                if history[tracked_team][stage]:
                    condition_key = f"Brasil chegou a {DISPLAY_STAGE_BY_HISTORY[stage]}"
                    tracked_conditions[condition_key]["den"] += 1
                    if champion == tracked_team:
                        tracked_conditions[condition_key]["champ"] += 1

            tracked_eliminator = result["tracked_eliminator"]
            if tracked_eliminator is not None:
                tracked_eliminators[tracked_eliminator] += 1

            for team_key, position in result["group_positions"].items():
                if position == 4:
                    continue
                if position == 3 and not history[team_key]["Top32"]:
                    continue
                group_key = (team_key, position)
                title_by_group_position[group_key]["den"] += 1
                if champion == team_key:
                    title_by_group_position[group_key]["champ"] += 1

            for stage in ["Top32", "Oitavas", "Quartas", "Semifinal", "Final", "Campeao"]:
                if any(history[team_key][stage] for team_key in bottom_16):
                    bottom_16_stage_counter[stage] += 1
                if any(history[team_key][stage] for team_key in mini_zebra):
                    mini_zebra_stage_counter[stage] += 1

        completed += current_chunk
        if progress_callback is not None:
            progress_callback(completed, n_sims)

    accumulated = {
        team_key: {
            stage: int(accum_np[team_idx_map[team_key], stage_idx_map[stage]])
            for stage in SIM_STAGE_COLUMNS
        }
        for team_key in team_keys
    }

    return {
        "accumulated": accumulated,
        "tables": build_detailed_tables(
            n_sims=n_sims,
            team_names=team_names,
            finals_counter=finals_counter,
            tracked_first_ko_by_group_position=tracked_first_ko_by_group_position,
            tracked_first_ko_base_by_group_position=tracked_first_ko_base_by_group_position,
            tracked_opponents_by_stage=tracked_opponents_by_stage,
            tracked_stage_base=tracked_stage_base,
            tracked_eliminators=tracked_eliminators,
            tracked_conditions=tracked_conditions,
            title_by_group_position=title_by_group_position,
            bottom_16_stage_counter=bottom_16_stage_counter,
            mini_zebra_stage_counter=mini_zebra_stage_counter,
            bottom_16_order=bottom_16_order,
            mini_zebra_order=mini_zebra_order,
        ),
    }


def build_detailed_tables(
    n_sims: int,
    team_names: dict[str, str],
    finals_counter: Counter[tuple[str, str]],
    tracked_first_ko_by_group_position: dict[int, Counter[str]],
    tracked_first_ko_base_by_group_position: Counter,
    tracked_opponents_by_stage: dict[str, Counter[str]],
    tracked_stage_base: Counter,
    tracked_eliminators: Counter[tuple[str, str]],
    tracked_conditions: dict,
    title_by_group_position: dict,
    bottom_16_stage_counter: Counter,
    mini_zebra_stage_counter: Counter,
    bottom_16_order: list[str],
    mini_zebra_order: list[str],
) -> dict[str, pd.DataFrame]:
    final_rows = []
    for pair, count in finals_counter.most_common():
        team_a, team_b = pair
        row = {
            "Final": _format_pair(pair, team_names),
            "Finalista A": team_names.get(team_a, team_a),
            "Finalista B": team_names.get(team_b, team_b),
            "Probabilidade": count / n_sims,
        }
        final_rows.append(row)

    def first_ko_rows_for_position(position: int) -> list[dict]:
        base = tracked_first_ko_base_by_group_position[position]
        return [
            {
                "Adversario": team_names.get(team, team),
                f"Prob dado Brasil {position}o grupo": count / base if base else 0.0,
                f"Base Brasil {position}o grupo": base,
            }
            for team, count in tracked_first_ko_by_group_position[position].most_common()
        ]

    def opponent_rows_for_stage(stage: str) -> list[dict]:
        base = tracked_stage_base[stage]
        return [
            {
                "Selecao": team_names.get(team, team),
                "Probabilidade": count / base if base else 0.0,
            }
            for team, count in tracked_opponents_by_stage[stage].most_common()
        ]

    total_tracked_eliminations = sum(tracked_eliminators.values())
    eliminator_rows = [
        {
            "Eliminador": team_names.get(team, team),
            "Fase": stage,
            "Prob dado Brasil eliminado": count / total_tracked_eliminations
            if total_tracked_eliminations
            else 0.0,
            "Prob no total de copas": count / n_sims,
        }
        for (team, stage), count in tracked_eliminators.most_common()
    ]

    eliminator_by_team = Counter()
    for (team, _stage), count in tracked_eliminators.items():
        eliminator_by_team[team] += count
    eliminator_grouped_rows = [
        {
            "Eliminador": team_names.get(team, team),
            "Prob dado Brasil eliminado": count / total_tracked_eliminations
            if total_tracked_eliminations
            else 0.0,
            "Prob no total de copas": count / n_sims,
        }
        for team, count in eliminator_by_team.most_common()
    ]

    condition_order = {
        "Brasil pos_1 no grupo": 1,
        "Brasil pos_2 no grupo": 2,
        "Brasil pos_3 no grupo": 3,
        "Brasil pos_4 no grupo": 4,
        "Brasil chegou a Top 32": 5,
        "Brasil chegou a Oitavas": 6,
        "Brasil chegou a Quartas": 7,
        "Brasil chegou a Semifinal": 8,
        "Brasil chegou a Final": 9,
    }
    condition_labels = {
        "Brasil pos_1 no grupo": "Brasil 1o no grupo",
        "Brasil pos_2 no grupo": "Brasil 2o no grupo",
        "Brasil pos_3 no grupo": "Brasil 3o no grupo",
        "Brasil pos_4 no grupo": "Brasil 4o no grupo",
    }
    conditional_rows = []
    for condition, values in tracked_conditions.items():
        den = values["den"]
        conditional_rows.append(
            {
                "Condicao": condition_labels.get(condition, condition),
                "Base": den,
                "Titulos Brasil": values["champ"],
                "Prob titulo": values["champ"] / den if den else 0.0,
                "_ordem": condition_order.get(condition, 999),
            }
        )
    conditional_rows = sorted(conditional_rows, key=lambda row: row["_ordem"])
    for row in conditional_rows:
        row.pop("_ordem", None)

    position_by_team = {}
    for (team, position), values in title_by_group_position.items():
        den = values["den"]
        if team not in position_by_team:
            position_by_team[team] = {"Selecao": team_names.get(team, team)}
        position_by_team[team][f"Prob titulo se {position}o grupo"] = (
            values["champ"] / den if den else 0.0
        )
    position_rows = []
    for team, row in position_by_team.items():
        position_rows.append(
            {
                "Selecao": row["Selecao"],
                "Prob titulo se 1o grupo": row.get("Prob titulo se 1o grupo", 0.0),
                "Prob titulo se 2o grupo": row.get("Prob titulo se 2o grupo", 0.0),
                "Prob titulo se avancou em 3o": row.get("Prob titulo se 3o grupo", 0.0),
            }
        )
    position_rows = sorted(
        position_rows,
        key=lambda row: (
            row["Prob titulo se 1o grupo"],
            row["Prob titulo se 2o grupo"],
            row["Prob titulo se avancou em 3o"],
        ),
        reverse=True,
    )

    _zebra_stage_label = {
        "Top32": "Top 32", "Oitavas": "Oitavas", "Quartas": "Quartas",
        "Semifinal": "Semifinal", "Final": "Final", "Campeao": "Campeã",
    }

    def _surpresa_rows(counter: Counter) -> list[dict]:
        return [
            {
                "Fase": _zebra_stage_label[stage],
                "Probabilidade": counter[stage] / n_sims,
            }
            for stage in ["Top32", "Oitavas", "Quartas", "Semifinal", "Final", "Campeao"]
        ]

    def _lista_rows(order: list[str]) -> list[dict]:
        # Mantém a ordem do indicador de força (mais fraca primeiro).
        return [{"Seleção": team_names.get(team, team)} for team in order]

    return {
        "finais": pd.DataFrame(final_rows),
        "brasil_1o_grupo_top32": pd.DataFrame(first_ko_rows_for_position(1)),
        "brasil_2o_grupo_top32": pd.DataFrame(first_ko_rows_for_position(2)),
        "brasil_3o_grupo_top32": pd.DataFrame(first_ko_rows_for_position(3)),
        "brasil_adversarios_16avos": pd.DataFrame(opponent_rows_for_stage("Top 32")),
        "brasil_adversarios_oitavas": pd.DataFrame(opponent_rows_for_stage("Oitavas")),
        "brasil_adversarios_quartas": pd.DataFrame(opponent_rows_for_stage("Quartas")),
        "brasil_adversarios_semifinal": pd.DataFrame(opponent_rows_for_stage("Semifinal")),
        "brasil_adversarios_final": pd.DataFrame(opponent_rows_for_stage("Final")),
        "eliminadores_brasil": pd.DataFrame(eliminator_rows),
        "eliminadores_brasil_agrupado": pd.DataFrame(eliminator_grouped_rows),
        "titulo_condicional_brasil": pd.DataFrame(conditional_rows),
        "impacto_posicao_grupo": pd.DataFrame(position_rows),
        "bottom16_surpresa": pd.DataFrame(_surpresa_rows(bottom_16_stage_counter)),
        "bottom16_lista": pd.DataFrame(_lista_rows(bottom_16_order)),
        "minizebra_surpresa": pd.DataFrame(_surpresa_rows(mini_zebra_stage_counter)),
        "minizebra_lista": pd.DataFrame(_lista_rows(mini_zebra_order)),
    }
