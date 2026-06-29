from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime as _dt
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from experimento_calibracao_mercado import canonical_team_key


BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS_PATH = BASE_DIR / "assets" / "resultados_jogos.json"

LockedMatchResult = tuple[int, int, int]
LockedGroupResult = LockedMatchResult
LockedGroupResults = dict[tuple[str, str, str], LockedMatchResult]
LockedKnockoutResults = dict[tuple[str, str], LockedMatchResult]


@dataclass(frozen=True)
class OfficialResults:
    locked_group_results: LockedGroupResults
    locked_group_results_by_name: LockedGroupResults
    locked_knockout_results: LockedKnockoutResults
    locked_knockout_results_by_name: LockedKnockoutResults
    total_rows: int
    matched_rows: int
    knockout_rows: int
    unmatched_rows: tuple[str, ...]
    path: Path

    @property
    def locked_match_count(self) -> int:
        return self.matched_rows

    @property
    def locked_knockout_count(self) -> int:
        return self.knockout_rows

    @property
    def locked_total_match_count(self) -> int:
        return self.matched_rows + self.knockout_rows


def _first_value(row: dict[str, Any], candidates: list[str]) -> Any:
    for key in candidates:
        if key in row:
            return row[key]
    return None


def _group_letter(value: object) -> str:
    return str(value or "").replace("Grupo ", "").strip()


def _team_name_column(dataframe: pd.DataFrame) -> str:
    for column in dataframe.columns:
        if str(column).startswith("Sele"):
            return column
    return "team_key"


def _winner_from_score(goals_a: int, goals_b: int) -> int:
    if goals_a > goals_b:
        return 1
    if goals_b > goals_a:
        return 2
    return 0


def _winner_from_result(goals_a: int, goals_b: int, winner_raw: object, team_a_raw: object, team_b_raw: object) -> int:
    winner_key = canonical_team_key(str(winner_raw or ""))
    if winner_key and winner_key == canonical_team_key(str(team_a_raw or "")):
        return 1
    if winner_key and winner_key == canonical_team_key(str(team_b_raw or "")):
        return 2
    return _winner_from_score(goals_a, goals_b)


def _reverse_result(result: LockedMatchResult) -> LockedMatchResult:
    goals_a, goals_b, winner = result
    if winner == 1:
        reversed_winner = 2
    elif winner == 2:
        reversed_winner = 1
    else:
        reversed_winner = 0
    return goals_b, goals_a, reversed_winner


def load_official_group_results(
    dataframe: pd.DataFrame,
    results_path: str | Path | None = None,
) -> OfficialResults:
    path = Path(results_path) if results_path else DEFAULT_RESULTS_PATH
    if not path.exists():
        return OfficialResults({}, {}, {}, {}, 0, 0, 0, (f"Arquivo nao encontrado: {path}",), path)

    with open(path, "r", encoding="utf-8") as file:
        raw_results = json.load(file)

    if not isinstance(raw_results, list):
        raise ValueError(f"{path} precisa conter uma lista de resultados.")

    name_column = _team_name_column(dataframe)
    team_key_by_name = {
        canonical_team_key(str(row[name_column])): str(row["team_key"])
        for _, row in dataframe.iterrows()
    }
    team_name_by_key = {
        str(row["team_key"]): str(row[name_column])
        for _, row in dataframe.iterrows()
    }
    group_by_team_key = {
        str(row["team_key"]): _group_letter(row["Grupo"])
        for _, row in dataframe.iterrows()
    }

    locked: LockedGroupResults = {}
    locked_by_name: LockedGroupResults = {}
    locked_knockout: LockedKnockoutResults = {}
    locked_knockout_by_name: LockedKnockoutResults = {}
    unmatched: list[str] = []
    matched_rows = 0
    knockout_rows = 0

    team_a_candidates = ["Sele\u00e7\u00e3o A", "Sele\u00c3\u00a7\u00c3\u00a3o A"]
    team_b_candidates = ["Sele\u00e7\u00e3o B", "Sele\u00c3\u00a7\u00c3\u00a3o B"]

    for index, row in enumerate(raw_results, start=1):
        if not isinstance(row, dict):
            unmatched.append(f"Linha {index}: item invalido")
            continue
        if str(row.get("Status", "")).lower() != "encerrado":
            continue

        team_a_raw = _first_value(row, team_a_candidates)
        team_b_raw = _first_value(row, team_b_candidates)
        goals_a_raw = _first_value(row, ["Placar A"])
        goals_b_raw = _first_value(row, ["Placar B"])
        winner_raw = _first_value(row, ["Vencedor", "Winner"])

        team_a_key = team_key_by_name.get(canonical_team_key(str(team_a_raw)))
        team_b_key = team_key_by_name.get(canonical_team_key(str(team_b_raw)))
        if not team_a_key or not team_b_key:
            unmatched.append(f"Linha {index}: selecao nao encontrada ({team_a_raw} x {team_b_raw})")
            continue

        try:
            goals_a = int(goals_a_raw)
            goals_b = int(goals_b_raw)
        except (TypeError, ValueError):
            unmatched.append(f"Linha {index}: placar invalido ({team_a_raw} x {team_b_raw})")
            continue

        team_a_name = team_name_by_key[team_a_key]
        team_b_name = team_name_by_key[team_b_key]
        group_a = group_by_team_key.get(team_a_key)
        group_b = group_by_team_key.get(team_b_key)

        if group_a and group_a == group_b:
            result = (goals_a, goals_b, _winner_from_score(goals_a, goals_b))
            reversed_result = _reverse_result(result)
            locked[(group_a, team_a_key, team_b_key)] = result
            locked[(group_a, team_b_key, team_a_key)] = reversed_result
            locked_by_name[(group_a, team_a_name, team_b_name)] = result
            locked_by_name[(group_a, team_b_name, team_a_name)] = reversed_result
            matched_rows += 1
            continue

        result = (goals_a, goals_b, _winner_from_result(goals_a, goals_b, winner_raw, team_a_raw, team_b_raw))
        reversed_result = _reverse_result(result)
        locked_knockout[(team_a_key, team_b_key)] = result
        locked_knockout[(team_b_key, team_a_key)] = reversed_result
        locked_knockout_by_name[(team_a_name, team_b_name)] = result
        locked_knockout_by_name[(team_b_name, team_a_name)] = reversed_result
        knockout_rows += 1

    return OfficialResults(
        locked,
        locked_by_name,
        locked_knockout,
        locked_knockout_by_name,
        len(raw_results),
        matched_rows,
        knockout_rows,
        tuple(unmatched),
        path,
    )


def render_official_results_sidebar(
    official_results: OfficialResults,
    key: str = "use_official_match_results",
) -> bool:
    can_use = official_results.locked_total_match_count > 0
    with st.sidebar:
        st.markdown("#### Resultados")
        use_results = st.toggle(
            "Usar Resultados de Jogos Encerrados",
            value=can_use,
            disabled=not can_use,
            key=key,
        )
        if can_use:
            updated_at = _dt.fromtimestamp(official_results.path.stat().st_mtime)
            extra = (
                f" - {official_results.locked_knockout_count} jogo(s) de mata-mata travado(s)"
                if official_results.locked_knockout_count
                else ""
            )
            st.caption(
                f"Resultados atualizados em {updated_at.strftime('%d/%m/%Y as %H:%M')} - "
                f"{official_results.locked_match_count} jogo(s) da fase de grupos travado(s)"
                f"{extra}."
            )
        else:
            st.caption("Nenhum resultado de jogo encerrado encontrado.")
    return bool(use_results and can_use)
