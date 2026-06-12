import argparse
import json
import shutil
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from buscador_vetor_forca import (
    FORCA_MAXIMA,
    FORCA_MINIMA,
    MEDIA_GOLS,
    OFFSET_SIMULACAO,
    OUTPUT_CSV,
    OUTPUT_VETOR,
    RHO_DIXON_COLES,
    USAR_DIXON_COLES,
    build_match_cache,
    compute_match_data,
    compute_metrics,
    load_data,
    simulate_and_count_champions,
)


SENTINELA_MATCH = ("brazil", "haiti")
SENTINELA_TARGET_RAW = np.array([0.91, 0.07, 0.05], dtype=float)  # win A, draw, win B
SENTINELA_TARGET = SENTINELA_TARGET_RAW / SENTINELA_TARGET_RAW.sum()


def aplicar_curva_s(
    base_forces: np.ndarray,
    tail_drop: float,
    tail_power: float,
    mid_bump: float,
    mid_center: float,
    mid_width: float,
) -> np.ndarray:
    """Abaixa a cauda, levanta o miolo e preserva aproximadamente o topo."""
    current_min = float(np.min(base_forces))
    current_max = float(np.max(base_forces))
    if current_max <= current_min:
        return np.full_like(base_forces, FORCA_MINIMA, dtype=float)

    x = (base_forces - current_min) / (current_max - current_min)
    tail = tail_drop * ((1.0 - x) ** tail_power)
    bump = mid_bump * np.exp(-((x - mid_center) / mid_width) ** 2)
    bump *= (1.0 - x**4)  # quase nao mexe na elite
    transformed = base_forces - tail + bump
    transformed = np.clip(transformed, FORCA_MINIMA, FORCA_MAXIMA)

    # Preserva a ordenacao do vetor atual, sem impor ranking de mercado.
    order = np.argsort(base_forces)
    sorted_values = transformed[order]
    sorted_values = np.maximum.accumulate(sorted_values)
    transformed[order] = sorted_values
    return np.clip(transformed, FORCA_MINIMA, FORCA_MAXIMA)


def match_outcome_probabilities(force_a: float, force_b: float) -> np.ndarray:
    data = compute_match_data(
        OFFSET_SIMULACAO + force_a,
        OFFSET_SIMULACAO + force_b,
        MEDIA_GOLS,
        usar_dc=USAR_DIXON_COLES,
        rho=RHO_DIXON_COLES,
    )
    matrix = data["matrix"]
    win_a = float(np.tril(matrix, -1).sum())
    draw = float(np.trace(matrix))
    win_b = float(np.triu(matrix, 1).sum())
    return np.array([win_a, draw, win_b], dtype=float)


def block_mse(p_market: np.ndarray, p_sim: np.ndarray) -> float:
    order = np.argsort(-p_market)
    blocks = [
        order[:2],
        order[:6],
        order[6:16],
        order[16:32],
        order[32:],
    ]
    errors = [(float(p_sim[idx].sum()) - float(p_market[idx].sum())) ** 2 for idx in blocks]
    return float(np.mean(errors))


def avaliar_vetor(team_keys, groups, p_market, forces, n_sims, seed, match_weight):
    strengths = {tk: float(OFFSET_SIMULACAO + forces[i]) for i, tk in enumerate(team_keys)}
    cache = build_match_cache(team_keys, strengths, MEDIA_GOLS, USAR_DIXON_COLES, RHO_DIXON_COLES)
    counts = simulate_and_count_champions(
        groups,
        strengths,
        cache,
        n_sims,
        rng=np.random.default_rng(seed),
    )
    p_sim = np.array([counts[tk] / n_sims for tk in team_keys])
    metrics = compute_metrics(p_market, p_sim)
    b_mse = block_mse(p_market, p_sim)

    idx_a = team_keys.index(SENTINELA_MATCH[0])
    idx_b = team_keys.index(SENTINELA_MATCH[1])
    match_probs = match_outcome_probabilities(float(forces[idx_a]), float(forces[idx_b]))
    match_mse = float(np.mean((match_probs - SENTINELA_TARGET) ** 2))

    objective = metrics["mse"] + b_mse + match_weight * match_mse
    return p_sim, metrics, b_mse, match_probs, match_mse, objective


def main():
    parser = argparse.ArgumentParser(description="Testa curvas S sobre o vetor otimizado atual.")
    parser.add_argument("--grid-sims", type=int, default=6000)
    parser.add_argument("--final-sims", type=int, default=40000)
    parser.add_argument("--match-weight", type=float, default=0.0002)
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    data = load_data()
    groups = data["groups"]
    team_keys = list(data["team_keys"])
    p_market = np.array([data["market_probs"][tk] for tk in team_keys], dtype=float)

    with open(OUTPUT_VETOR, encoding="utf-8") as f:
        vector_data = json.load(f)
    base_vector = {str(k): float(v) for k, v in vector_data["vetor_forca"].items()}
    base_forces = np.array([base_vector[tk] for tk in team_keys], dtype=float)

    candidates = [("baseline", base_forces, {})]
    for tail_drop in [0.03, 0.06, 0.09]:
        for tail_power in [1.5, 2.5]:
            for mid_bump in [0.03, 0.06, 0.09]:
                for mid_center in [0.35, 0.50]:
                    params = {
                        "tail_drop": tail_drop,
                        "tail_power": tail_power,
                        "mid_bump": mid_bump,
                        "mid_center": mid_center,
                        "mid_width": 0.22,
                    }
                    forces = aplicar_curva_s(base_forces, **params)
                    candidates.append(("s_curve", forces, params))

    rows = []
    start = time.time()
    for idx, (kind, forces, params) in enumerate(candidates, start=1):
        p_sim, metrics, b_mse, match_probs, match_mse, objective = avaliar_vetor(
            team_keys,
            groups,
            p_market,
            forces,
            args.grid_sims,
            91000 + idx,
            args.match_weight,
        )
        row = {
            "idx": idx,
            "kind": kind,
            **params,
            "objective": objective,
            "champ_mse": metrics["mse"],
            "block_mse": b_mse,
            "match_mse": match_mse,
            "mae": metrics["mae"],
            "kl_div": metrics["kl_div"],
            "brazil_haiti_brazil": match_probs[0],
            "brazil_haiti_draw": match_probs[1],
            "brazil_haiti_haiti": match_probs[2],
            "min_force": float(np.min(forces)),
            "max_force": float(np.max(forces)),
        }
        rows.append(row)
        print(
            f"{idx:02d}/{len(candidates)} obj={objective:.8f} "
            f"champ={metrics['mse']:.8f} match={match_mse:.5f} "
            f"BRxHA={match_probs[0]:.1%}/{match_probs[1]:.1%}/{match_probs[2]:.1%} "
            f"params={params}",
            flush=True,
        )

    ranking = pd.DataFrame(rows).sort_values("objective").reset_index(drop=True)
    best_idx = int(ranking.loc[0, "idx"])
    kind, best_forces, best_params = candidates[best_idx - 1]
    print("\nMelhores candidatos:")
    print(ranking.head(8).to_string(index=False))

    p_sim, metrics, b_mse, match_probs, match_mse, objective = avaliar_vetor(
        team_keys,
        groups,
        p_market,
        best_forces,
        args.final_sims,
        92000,
        args.match_weight,
    )
    print("\nValidacao final:")
    print(f"objective={objective:.8f} champ_mse={metrics['mse']:.8f} block_mse={b_mse:.8f}")
    print(f"metrics={metrics}")
    print(
        f"Brasil x Haiti: Brazil={match_probs[0]:.1%}, Draw={match_probs[1]:.1%}, "
        f"Haiti={match_probs[2]:.1%}"
    )
    print(f"params={best_params}")

    output = pd.DataFrame(
        {
            "team_key": team_keys,
            "forca": best_forces,
            "prob_mercado": p_market,
            "prob_simulada": p_sim,
            "erro_absoluto": np.abs(p_sim - p_market),
        }
    ).sort_values("prob_mercado", ascending=False)
    print(output.head(20).to_string(index=False))

    if args.save:
        backup = OUTPUT_VETOR.with_name(
            f"{OUTPUT_VETOR.stem}_backup_pre_curva_s_{datetime.now():%Y%m%d_%H%M%S}{OUTPUT_VETOR.suffix}"
        )
        shutil.copy2(OUTPUT_VETOR, backup)
        vector_data["descricao"] = "Vetor de forca otimizado com curva S sobre o vetor base"
        vector_data["parametros_simulacao"]["ajuste"] = "curva_s_sobre_vetor_base"
        vector_data["parametros_simulacao"]["curva_s"] = best_params
        vector_data["parametros_simulacao"]["match_sentinela"] = {
            "jogo": list(SENTINELA_MATCH),
            "alvo_normalizado": SENTINELA_TARGET.tolist(),
            "peso_objetivo": args.match_weight,
        }
        vector_data["metricas_finais"] = metrics
        vector_data["vetor_forca"] = {tk: float(best_forces[i]) for i, tk in enumerate(team_keys)}
        with open(OUTPUT_VETOR, "w", encoding="utf-8") as f:
            json.dump(vector_data, f, ensure_ascii=False, indent=2)
        output.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
        print(f"\nBackup salvo em: {backup}")
        print(f"Vetor atualizado em: {OUTPUT_VETOR}")
        print(f"CSV atualizado em: {OUTPUT_CSV}")

    print(f"\nTempo total: {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
