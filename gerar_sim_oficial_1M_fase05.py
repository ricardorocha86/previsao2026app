# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path

import pandas as pd

from utils.forca_core import (
    BASE_DIR,
    DATA_DIR,
    build_optimized_force_table,
    compute_match_probabilities,
    find_latest_enriched_dataset,
    load_force_table,
    load_optimized_force_vector,
)
from utils.resultados_oficiais import load_official_group_results
from utils.simulador_analitico import SIM_STAGE_COLUMNS, run_detailed_simulation
from utils.simulador_oficial import PoissonMatchSimulator


ETAPA = "Início das Oitavas"
DATA_STR = "04.07.2026"
N_SIMS = 1_000_000
MEDIA_GOLS = 3.0
USAR_DIXON_COLES = True
RHO_DIXON_COLES = -0.13
TIPO_CHAVEAMENTO = "Sorteio Oficial"
OUT_XLSX = BASE_DIR / "resultados" / f"Simulação Oficial {ETAPA} {DATA_STR}.xlsx"

CATEGORY_STAGE_SLOTS = {
    "Top32": 32,
    "Oitavas": 16,
    "Quartas": 8,
    "Semifinal": 4,
    "Final": 2,
    "Campeao": 1,
}

CATEGORY_STAGE_LABELS = {
    "Top32": "Top 32",
    "Oitavas": "Oitavas",
    "Quartas": "Quartas",
    "Semifinal": "Semi",
    "Final": "Final",
    "Campeao": "Campeão",
}


def team_name_column(dataframe: pd.DataFrame) -> str:
    for column in dataframe.columns:
        if str(column).startswith("Sele"):
            return column
    return "team_key"


def build_match_cache(dataframe: pd.DataFrame) -> dict[tuple[str, str], dict]:
    cache = {}
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
                media_gols=MEDIA_GOLS,
                max_goals=10,
                usar_dixon_coles=USAR_DIXON_COLES,
                rho_dixon_coles=RHO_DIXON_COLES,
            )
    return cache


def build_simulation_table(dataframe: pd.DataFrame, accumulated: dict, n_sims: int) -> pd.DataFrame:
    name_col = team_name_column(dataframe)
    result = dataframe[[name_col, "team_key", "forca_com_offset"]].copy()
    result = result.rename(columns={name_col: "Seleção"})
    for stage in SIM_STAGE_COLUMNS:
        result[f"{stage}_pct"] = result["team_key"].map(lambda key: accumulated[key][stage] / n_sims)
    result = result.sort_values(
        by=["Campeao_pct", "Final_pct", "Semifinal_pct", "forca_com_offset"],
        ascending=False,
    ).reset_index(drop=True)
    result.index = result.index + 1
    result.insert(0, "Rank Sim", result.index)
    return result


def build_display_table(sim_table: pd.DataFrame) -> pd.DataFrame:
    return sim_table[
        [
            "Rank Sim",
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
            "Rank Sim": "Rank",
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


def build_elimination_table(sim_table: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({"Seleção": sim_table["Seleção"].values})
    out["Fase de Grupos"] = 1.0 - sim_table["Top32_pct"].values
    out["16-avos"] = sim_table["Top32_pct"].values - sim_table["Oitavas_pct"].values
    out["Oitavas"] = sim_table["Oitavas_pct"].values - sim_table["Quartas_pct"].values
    out["Quartas"] = sim_table["Quartas_pct"].values - sim_table["Semifinal_pct"].values
    out["Semifinal"] = sim_table["Semifinal_pct"].values - sim_table["Final_pct"].values
    out["Vice (Final)"] = sim_table["Final_pct"].values - sim_table["Campeao_pct"].values
    out["Campeã"] = sim_table["Campeao_pct"].values
    for column in out.columns:
        if column != "Seleção":
            out[column] = out[column].clip(lower=0.0)
    out = out.sort_values(by=["Campeã", "Vice (Final)", "Semifinal", "Quartas"], ascending=False).reset_index(drop=True)
    out.index = out.index + 1
    out.insert(0, "Rank", out.index)
    return out


def build_group_stage_table(sim_table: pd.DataFrame, meta_df: pd.DataFrame) -> pd.DataFrame:
    merged = sim_table.merge(meta_df[["team_key", "Grupo"]], on="team_key", how="left")
    out = pd.DataFrame({"Grupo": merged["Grupo"].values, "Seleção": merged["Seleção"].values})
    out["1º"] = merged["pos_1_pct"].values
    out["2º"] = merged["pos_2_pct"].values
    out["3º"] = merged["pos_3_pct"].values
    out["4º"] = merged["pos_4_pct"].values
    out["Avança como 3º"] = merged["Top32_pct"].values - merged["pos_1_pct"].values - merged["pos_2_pct"].values
    out["Classifica (Top 32)"] = merged["Top32_pct"].values
    out["Cai na fase de grupos"] = 1.0 - merged["Top32_pct"].values
    for column in ["1º", "2º", "3º", "4º", "Avança como 3º", "Classifica (Top 32)", "Cai na fase de grupos"]:
        out[column] = out[column].clip(lower=0.0)
    return out.sort_values(by=["Grupo", "Classifica (Top 32)"], ascending=[True, False]).reset_index(drop=True)


def count_world_titles(value: object) -> int:
    text = str(value or "").strip().lower()
    if not text.startswith("campe"):
        return 0
    if "(" not in text or ")" not in text:
        return 1
    inside = text.split("(", 1)[1].split(")", 1)[0]
    years = [part.strip() for part in inside.replace("/", ",").replace(";", ",").split(",") if part.strip()]
    return len(years) if years else 1


def aggregate_category(merged: pd.DataFrame, group_col: str, sort_alpha: bool = False, order_key=None) -> pd.DataFrame:
    rows = []
    for category, sub in merged.groupby(group_col):
        row = {
            "Categoria": category,
            "Nº Seleções": int(len(sub)),
            "Força Média": float(sub["forca_com_offset"].mean()),
        }
        for stage, slots in CATEGORY_STAGE_SLOTS.items():
            row[CATEGORY_STAGE_LABELS[stage]] = float(sub[f"{stage}_pct"].sum()) / slots
        rows.append(row)
    result = pd.DataFrame(rows)
    result["Média Campeão"] = result["Campeão"] / result["Nº Seleções"]
    if order_key is not None:
        result["_ord"] = result["Categoria"].map(order_key)
        result = result.sort_values(by="_ord").drop(columns="_ord")
    elif sort_alpha:
        result = result.sort_values(by="Categoria")
    else:
        result = result.sort_values(by="Campeão", ascending=False)
    return result.reset_index(drop=True)


def build_category_tables(sim_table: pd.DataFrame, meta_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    meta_cols = ["team_key", "Grupo", "Confederação", "Participações_Copa_Mundo", "Melhor_Resultado_Copa_Mundo"]
    merged = sim_table.merge(meta_df[meta_cols], on="team_key", how="left")
    merged["cat_grupo"] = "Grupo " + merged["Grupo"].astype(str)
    merged["cat_confed"] = merged["Confederação"].fillna("Sem confederação")
    apps = pd.to_numeric(merged["Participações_Copa_Mundo"], errors="coerce").fillna(-1)
    titles = merged["Melhor_Resultado_Copa_Mundo"].apply(count_world_titles)

    def title_category(n_apps: float, n_titles: int) -> str:
        if n_apps == 0:
            return "Estreantes (1ª Copa)"
        if n_titles == 0:
            return "Nunca campeãs"
        if n_titles in (1, 2):
            return "1 ou 2 títulos"
        return "3+ títulos"

    def title_order(category: str) -> tuple[int, int]:
        if category.startswith("Estre"):
            return (0, 0)
        if category.startswith("Nunca"):
            return (1, 0)
        if category.startswith("1"):
            return (2, 1)
        return (2, 3)

    merged["cat_titulos"] = [title_category(a, t) for a, t in zip(apps, titles)]
    return {
        "Por grupo": aggregate_category(merged, "cat_grupo", sort_alpha=True),
        "Por confederação": aggregate_category(merged, "cat_confed"),
        "Por títulos em Copas": aggregate_category(merged, "cat_titulos", order_key=title_order),
    }


def normalize_match(raw_match: dict) -> dict:
    group = raw_match.get("group") or raw_match.get("Grupo") or ""
    return {
        "group": str(group).replace("Grupo ", "").strip(),
        "date": raw_match.get("date") or raw_match.get("Data") or "",
        "team_a": raw_match.get("team_a") or raw_match.get("Seleção A") or "",
        "team_b": raw_match.get("team_b") or raw_match.get("Seleção B") or "",
    }


def generate_group_predictions(dataframe: pd.DataFrame) -> pd.DataFrame:
    schedule_path = BASE_DIR / "assets" / "previsoes_jogos.json"
    if not schedule_path.exists():
        return pd.DataFrame()
    schedule = json.loads(schedule_path.read_text(encoding="utf-8"))
    name_col = team_name_column(dataframe)
    aliases = {
        "República da Coreia": "Coreia do Sul",
        "República Democrática do Congo": "RD do Congo",
        "República Tcheca": "Tcheca",
    }
    rows = []
    for raw in schedule:
        match = normalize_match(raw)
        team_a = aliases.get(match["team_a"], match["team_a"])
        team_b = aliases.get(match["team_b"], match["team_b"])
        if team_a not in set(dataframe[name_col]) or team_b not in set(dataframe[name_col]):
            continue
        force_a = float(dataframe.loc[dataframe[name_col] == team_a, "forca_com_offset"].iloc[0])
        force_b = float(dataframe.loc[dataframe[name_col] == team_b, "forca_com_offset"].iloc[0])
        probs = compute_match_probabilities(force_a, force_b, MEDIA_GOLS, usar_dixon_coles=USAR_DIXON_COLES, rho_dixon_coles=RHO_DIXON_COLES)
        rows.append({
            "Grupo": f"Grupo {match['group']}",
            "Data": match["date"],
            "Local": raw.get("Local", raw.get("location_time", "")),
            "Horário Brasília": raw.get("Horário Brasília", raw.get("br_time", "")),
            "Seleção A": team_a,
            "Vitória A": probs["win_a"],
            "Empate": probs["draw"],
            "Vitória B": probs["win_b"],
            "Seleção B": team_b,
            "Força A": force_a,
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(by=["Grupo", "Força A"], ascending=[True, False]).drop(columns=["Força A"]).reset_index(drop=True)
    return df


def write_excel(path: Path, sim_display: pd.DataFrame, info_df: pd.DataFrame, matches_df: pd.DataFrame, extra_sheets: dict[str, pd.DataFrame]) -> None:
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        sim_display.to_excel(writer, sheet_name="Simulações", index=False)
        info_df.to_excel(writer, sheet_name="Parâmetros", index=False)
        if matches_df is not None and not matches_df.empty:
            matches_df.to_excel(writer, sheet_name="Previsão Jogos", index=False)
        for sheet_name, sheet_df in extra_sheets.items():
            sheet_df.to_excel(writer, sheet_name=sheet_name[:31], index=False)


def main() -> None:
    print(f"Carregando base: {find_latest_enriched_dataset()}")
    base_df = load_force_table(str(find_latest_enriched_dataset()))
    vector_data = load_optimized_force_vector()
    combined_df, _ = build_optimized_force_table(base_df, vector_data)
    strengths = dict(zip(combined_df["team_key"], combined_df["forca_com_offset"]))
    match_cache = build_match_cache(combined_df)
    match_simulator = PoissonMatchSimulator(match_cache=match_cache, strengths=strengths)
    official_results = load_official_group_results(combined_df)
    print(f"Resultados travados: {official_results.locked_match_count} grupos + {official_results.locked_knockout_count} mata-mata")

    def progress(done: int, total: int) -> None:
        if done == total or done % 50_000 == 0:
            print(f"Simuladas {done:,}/{total:,} copas".replace(",", "."))

    detailed_result = run_detailed_simulation(
        dataframe=combined_df,
        n_sims=N_SIMS,
        match_simulator=match_simulator,
        strengths=strengths,
        tipo_chaveamento=TIPO_CHAVEAMENTO,
        chunk_size=10_000,
        progress_callback=progress,
        locked_group_results=official_results.locked_group_results,
        locked_knockout_results=official_results.locked_knockout_results,
    )

    sim_table = build_simulation_table(combined_df, detailed_result["accumulated"], N_SIMS)
    sim_display = build_display_table(sim_table)
    metrics = vector_data.get("metricas_finais", {})
    info_df = pd.DataFrame([
        {"Parametro": "Etapa", "Valor": ETAPA},
        {"Parametro": "Modo do vetor de força", "Valor": "Otimizado - aprovado manualmente"},
        {"Parametro": "Arquivo do vetor", "Valor": "vetor_forca_otimo.json"},
        {"Parametro": "KL final do vetor", "Valor": metrics.get("kl_div", "")},
        {"Parametro": "MAE final do vetor", "Valor": metrics.get("mae", "")},
        {"Parametro": "Erro max final do vetor", "Valor": metrics.get("max_err", "")},
        {"Parametro": "Média de gols", "Valor": MEDIA_GOLS},
        {"Parametro": "Usar Dixon-Coles", "Valor": USAR_DIXON_COLES},
        {"Parametro": "Rho Dixon-Coles", "Valor": RHO_DIXON_COLES},
        {"Parametro": "Número de Copas", "Valor": N_SIMS},
        {"Parametro": "Tipo de Simulação", "Valor": "Completa"},
        {"Parametro": "Resultados oficiais travados", "Valor": True},
        {"Parametro": "Jogos oficiais travados", "Valor": official_results.locked_total_match_count},
    ])

    tables = detailed_result["tables"]
    extra_sheets = {
        "Finais": tables["finais"],
        "Brasil 1o Top32": tables["brasil_1o_grupo_top32"],
        "Brasil 2o Top32": tables["brasil_2o_grupo_top32"],
        "Brasil 3o Top32": tables["brasil_3o_grupo_top32"],
        "Brasil Adv 16avos": tables["brasil_adversarios_16avos"],
        "Brasil Adv Oitavas": tables["brasil_adversarios_oitavas"],
        "Brasil Adv Quartas": tables["brasil_adversarios_quartas"],
        "Brasil Adv Semi": tables["brasil_adversarios_semifinal"],
        "Brasil Adv Final": tables["brasil_adversarios_final"],
        "Eliminadores Brasil": tables["eliminadores_brasil"],
        "Carrascos Brasil": tables["eliminadores_brasil_agrupado"],
        "Titulo Cond Brasil": tables["titulo_condicional_brasil"],
        "Impacto Pos Grupo": tables["impacto_posicao_grupo"],
        "Bottom16 Surpresa": tables["bottom16_surpresa"],
        "Bottom16 Lista": tables["bottom16_lista"],
        "MiniZebra Surpresa": tables["minizebra_surpresa"],
        "MiniZebra Lista": tables["minizebra_lista"],
        "Brasil Encontros": tables["brasil_encontros"],
        "Terceiros Cond": tables["terceiro_condicional"],
        "Terceiros Por Pontos": tables["terceiro_por_pontos"],
        "Terceiros Por Saldo": tables["terceiro_por_saldo"],
        "Fase de Grupos Detalhe": build_group_stage_table(sim_table, combined_df),
        "Fase de Eliminacao": build_elimination_table(sim_table),
        "Cat Por Grupo": build_category_tables(sim_table, combined_df)["Por grupo"],
        "Cat Por Confederacao": build_category_tables(sim_table, combined_df)["Por confederação"],
        "Cat Por Titulos": build_category_tables(sim_table, combined_df)["Por títulos em Copas"],
    }
    OUT_XLSX.parent.mkdir(parents=True, exist_ok=True)
    write_excel(OUT_XLSX, sim_display, info_df, generate_group_predictions(combined_df), extra_sheets)
    print(f"Salvo: {OUT_XLSX}")


if __name__ == "__main__":
    main()
