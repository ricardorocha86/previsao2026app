"""
============================================================
ESTÁGIO 2: Ajustar Parâmetros do Modelo ao Vetor de Força
============================================================
Dado o vetor de força ótimo f* (encontrado pelo Estágio 1),
encontra os parâmetros do modelo (pesos, elasticidade, offset)
que melhor reproduzem aquele vetor.

Vantagem: ZERO simulações de copa. Cada avaliação é uma 
operação vetorial instantânea (~microsegundos).

Usa scipy.optimize.differential_evolution para busca global.
============================================================
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import differential_evolution, minimize

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from experimento_calibracao_mercado import canonical_team_key

# ==========================================
# CONFIGURAÇÃO
# ==========================================
DATA_DIR = BASE_DIR / "dataset"
RESULTADO_DIR = BASE_DIR / "resultados"
VETOR_FORCA_PATH = RESULTADO_DIR / "vetor_forca_otimo.json"
OUTPUT_PARAMS = RESULTADO_DIR / "parametros_otimos.json"
OUTPUT_COMPARATIVO = RESULTADO_DIR / "comparativo_forca_modelo.csv"

# ==========================================
# FUNÇÕES DE APOIO
# ==========================================
def find_latest_enriched_dataset() -> Path:
    candidates = sorted(DATA_DIR.glob("FIFA_ELO_DadosSeleções_*.xlsx"))
    if candidates:
        return candidates[-1]
    return DATA_DIR / "FIFA_ELO_DadosSeleções_2026-04-15.xlsx"


def minmax_scale(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    minimum = numeric.min()
    maximum = numeric.max()
    if pd.isna(minimum) or pd.isna(maximum) or maximum == minimum:
        return pd.Series(0.5, index=series.index, dtype=float)
    return ((numeric - minimum) / (maximum - minimum)).astype(float)


def parse_world_cup_history_score(text: object) -> float:
    raw = str(text or "").lower()
    if "campe" in raw: return 1.00
    if "2" in raw or "vice" in raw: return 0.88
    if "3" in raw: return 0.78
    if "4" in raw: return 0.70
    if "quart" in raw: return 0.55
    if "oitav" in raw: return 0.38
    if "grupos" in raw: return 0.20
    if "estre" in raw: return 0.10
    return 0.10


def load_features() -> pd.DataFrame:
    """Carrega o dataset e prepara todas as features normalizadas (0-1)."""
    dataset_path = find_latest_enriched_dataset()
    df = pd.read_excel(dataset_path)
    result = df.copy()
    result["team_key"] = result["NomeIngles"].map(canonical_team_key)
    
    hosts = ["Estados Unidos", "México", "Canadá"]
    result["is_host"] = result["Seleção"].isin(hosts).astype(int)
    
    result["fifa_force_01"] = minmax_scale(result["FIFA_Current_Points"])
    result["elo_force_01"] = minmax_scale(result["ELO_Rating"])
    result["momentum_force_01"] = minmax_scale(result["ELO_Chg_2A"])
    result["market_force_01"] = minmax_scale(result["Valor_Mercado_Milhoes_EUR"])
    result["world_cup_apps_01"] = minmax_scale(result["Participações_Copa_Mundo"])
    result["world_cup_best_raw"] = result["Melhor_Resultado_Copa_Mundo"].map(parse_world_cup_history_score)
    result["world_cup_history_01"] = (0.5 * result["world_cup_apps_01"] + 0.5 * result["world_cup_best_raw"])
    
    return result


def load_target_vector() -> tuple[dict, float, dict]:
    """
    Carrega o vetor de força ótimo do Estágio 1.
    Retorna (vetor, offset_usado_na_simulacao, metricas_do_teto).
    """
    with open(VETOR_FORCA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    offset = float(data.get("parametros_simulacao", {}).get("offset", 0.10))
    teto = data.get("metricas_finais", {})
    return data["vetor_forca"], offset, teto


def compute_model_force(features_df, params):
    """
    Calcula a força resultante do modelo dados os parâmetros.
    Replica exatamente a lógica do build_combined_table do otimizador.
    
    params: (w_fifa, w_elo, w_momentum, w_market, w_history, w_host, elasticidade, offset)
    """
    w_fifa, w_elo, w_momentum, w_market, w_history, w_host, elasticidade, offset = params
    
    weight_sum = w_fifa + w_elo + w_momentum + w_market + w_history + w_host
    if weight_sum <= 0:
        return np.zeros(len(features_df))
    
    forca = (
        w_fifa * features_df["fifa_force_01"].values +
        w_elo * features_df["elo_force_01"].values +
        w_momentum * features_df["momentum_force_01"].values +
        w_market * features_df["market_force_01"].values +
        w_history * features_df["world_cup_history_01"].values +
        w_host * features_df["is_host"].values
    ) / weight_sum
    
    # Normalizar pelo máximo
    max_forca = forca.max()
    if max_forca > 0:
        forca = forca / max_forca
    
    # Aplicar elasticidade
    forca_elastica = forca ** elasticidade
    
    # Aplicar offset (mas não usar o offset para comparação com o vetor alvo,
    # pois o vetor alvo já é a força "pura" antes do offset de simulação)
    # O offset de simulação é adicionado separadamente na hora de simular
    
    return forca_elastica


def objective_mse(params, features_df, target_forces):
    """Função objetivo: MSE entre força do modelo e força alvo."""
    model_forces = compute_model_force(features_df, params)
    return float(np.mean((model_forces - target_forces) ** 2))


def objective_weighted(params, features_df, target_forces, target_probs):
    """
    Função objetivo ponderada: dá mais peso a seleções com alta probabilidade,
    pois errar a Espanha (15%) importa mais que errar o Haiti (0.06%).
    """
    model_forces = compute_model_force(features_df, params)
    
    # Pesos proporcionais à prob de mercado (normalizado)
    weights = target_probs / target_probs.sum()
    
    # MSE ponderado
    weighted_mse = float(np.sum(weights * (model_forces - target_forces) ** 2))
    
    # Penalização extra: correlação de Spearman (queremos preservar o ranking)
    from scipy.stats import spearmanr
    rho, _ = spearmanr(model_forces, target_forces)
    rank_penalty = 0.01 * (1.0 - rho)  # Penalidade se ranking divergir
    
    return weighted_mse + rank_penalty


# ==========================================
# BUSCA GLOBAL DOS PARÂMETROS
# ==========================================
def rmspe_acima_1pct(p_market, p_sim):
    """RMSPE restrito às seleções com prob de mercado >= 1% (métrica principal)."""
    mask = p_market >= 0.01
    if not mask.any():
        return float("nan")
    return float(np.sqrt(np.mean(((p_sim[mask] - p_market[mask]) / p_market[mask]) ** 2)))


def validar_ponta_a_ponta(best_params, features_df, p_market, n_sims, seed=123):
    """
    Fecha o ciclo: simula a Copa com os pesos ajustados e compara as
    probabilidades de campeão com o mercado (a métrica que importa).
    """
    from buscador_vetor_forca import (
        MEDIA_GOLS, USAR_DIXON_COLES, RHO_DIXON_COLES,
        build_match_cache, simulate_and_count_champions, compute_metrics,
    )
    offset = best_params[7]
    model_forces = compute_model_force(features_df, best_params)
    team_keys = list(features_df["team_key"])
    groups = features_df.groupby("Grupo")["team_key"].apply(list).to_dict()
    strengths = {tk: float(offset + model_forces[i]) for i, tk in enumerate(team_keys)}

    cache = build_match_cache(team_keys, strengths, MEDIA_GOLS, USAR_DIXON_COLES, RHO_DIXON_COLES)
    counts = simulate_and_count_champions(
        groups, strengths, cache, n_sims, rng=np.random.default_rng(seed))
    p_sim = np.array([counts[tk] / n_sims for tk in team_keys])

    metricas = compute_metrics(p_market, p_sim)
    metricas["rmspe_1pct"] = rmspe_acima_1pct(p_market, p_sim)
    return metricas, p_sim


def ajustar_parametros(val_sims=128_000):
    print("=" * 65)
    print("🔧 ESTÁGIO 2: Ajustando Parâmetros do Modelo ao Vetor de Força")
    print("=" * 65)

    # Carregar dados
    features_df = load_features()
    target_vector, offset_fixo, metricas_teto = load_target_vector()
    print(f"   Offset fixo (herdado do Estágio 1): {offset_fixo:.2f}")
    
    # Alinhar a ordem
    team_keys = list(features_df["team_key"])
    target_forces = np.array([target_vector.get(tk, 0.0) for tk in team_keys])
    
    # Carregar probabilidades de mercado para ponderação
    from experimento_calibracao_mercado import load_market_target, ODDS_PATH
    odds_df = load_market_target(ODDS_PATH)
    prob_map = dict(zip(odds_df["team_key"], odds_df["market_prob"]))
    target_probs = np.array([prob_map.get(tk, 0.001) for tk in team_keys])
    
    print(f"   Seleções: {len(team_keys)}")
    print(f"   Força máxima alvo: {target_forces.max():.4f}")
    print(f"   Força mínima alvo: {target_forces.min():.4f}")
    
    # Bounds para os parâmetros de busca (7 dimensões — o offset NÃO entra:
    # ele não afeta a comparação com o vetor e fica fixo no valor do Estágio 1)
    bounds = [
        (0.0, 1.0),   # w_fifa
        (0.0, 1.0),   # w_elo
        (0.0, 1.0),   # w_momentum
        (0.0, 1.0),   # w_market
        (0.0, 1.0),   # w_history
        (0.0, 0.5),   # w_host (mais limitado, é binário)
        (0.5, 3.0),   # elasticidade
    ]

    param_names = [
        "peso_fifa", "peso_elo", "peso_momentum", "peso_mercado",
        "peso_historico", "peso_anfitriao", "elasticidade", "offset"
    ]

    def objective_busca(p7, features_df, target_forces, target_probs):
        """Recompõe o vetor de 8 parâmetros com o offset fixo."""
        return objective_weighted(np.append(p7, offset_fixo), features_df, target_forces, target_probs)
    
    print("\n🔍 Rodando Differential Evolution (busca global)...")
    start_time = time.time()
    
    # Callback para progresso
    iteration_count = [0]
    def callback(xk, convergence):
        iteration_count[0] += 1
        if iteration_count[0] % 50 == 0:
            loss = objective_busca(xk, features_df, target_forces, target_probs)
            print(f"   Geração {iteration_count[0]:4d} | Loss: {loss:.8f}")

    result = differential_evolution(
        objective_busca,
        bounds=bounds,
        args=(features_df, target_forces, target_probs),
        maxiter=600,
        popsize=30,
        tol=1e-12,
        seed=42,
        callback=callback,
        workers=1,
        polish=True,  # Refina com L-BFGS-B no final
    )

    elapsed = time.time() - start_time
    print(f"\n   Concluído em {elapsed:.1f}s")
    print(f"   Função objetivo final: {result.fun:.10f}")
    print(f"   Convergiu: {result.success}")

    # Extrair parâmetros (recompõe os 8 com o offset fixo do Estágio 1)
    best_params = np.append(result.x, offset_fixo)
    w_sum = sum(best_params[:6])
    
    # Calcular força do modelo com os parâmetros ótimos
    model_forces = compute_model_force(features_df, best_params)
    
    # Métricas de ajuste
    mse = float(np.mean((model_forces - target_forces) ** 2))
    mae = float(np.mean(np.abs(model_forces - target_forces)))
    r_squared = 1.0 - mse / np.var(target_forces)
    
    from scipy.stats import spearmanr, pearsonr
    rho_spearman, _ = spearmanr(model_forces, target_forces)
    rho_pearson, _ = pearsonr(model_forces, target_forces)
    
    # ==========================================
    # RESULTADOS
    # ==========================================
    print("\n" + "=" * 65)
    print("👑 PARÂMETROS ÓTIMOS ENCONTRADOS")
    print("-" * 65)
    for i, name in enumerate(param_names[:6]):
        pct = (best_params[i] / w_sum * 100) if w_sum > 0 else 0
        print(f"  {name:<20} = {best_params[i]:.4f}  ({pct:>6.2f}%)")
    print(f"  {'elasticidade':<20} = {best_params[6]:.4f}")
    print(f"  {'offset':<20} = {best_params[7]:.4f}  (fixo, herdado do Estágio 1)")
    
    print("\n📊 QUALIDADE DO AJUSTE")
    print(f"  MSE:         {mse:.8f}")
    print(f"  MAE:         {mae:.6f}")
    print(f"  R²:          {r_squared:.6f}")
    print(f"  Pearson ρ:   {rho_pearson:.6f}")
    print(f"  Spearman ρ:  {rho_spearman:.6f}")
    
    # ==========================================
    # VALIDAÇÃO PONTA A PONTA (pesos -> simulação -> mercado)
    # ==========================================
    print("\n" + "=" * 65)
    print(f"🔁 VALIDAÇÃO PONTA A PONTA ({val_sims:,} copas com os pesos ajustados)")
    metricas_e2e, p_sim_e2e = validar_ponta_a_ponta(
        best_params, features_df, target_probs, n_sims=val_sims)
    print(f"  RMSPE (prob mercado >= 1%): {metricas_e2e['rmspe_1pct']:.4f}   <- métrica principal")
    print(f"  KL-Div:                     {metricas_e2e['kl_div']:.6f}")
    print(f"  MAE:                        {metricas_e2e['mae']:.6f}")
    print(f"  Spearman ρ:                 {metricas_e2e['spearman_rho']:.4f}")
    if metricas_teto:
        print(f"  — Teto (vetor livre do Estágio 1): KL={metricas_teto.get('kl_div', float('nan')):.6f} "
              f"| MAE={metricas_teto.get('mae', float('nan')):.6f}")
        print("  O gap entre o ajuste e o teto = o que as 6 variáveis NÃO explicam do mercado.")

    # Salvar resultados
    RESULTADO_DIR.mkdir(parents=True, exist_ok=True)

    output = {
        "descricao": "Parâmetros do modelo ajustados ao vetor de força ótimo (Estágio 2)",
        "parametros": {name: float(best_params[i]) for i, name in enumerate(param_names)},
        "parametros_normalizados_pct": {
            name: float(best_params[i] / w_sum * 100) if w_sum > 0 else 0
            for i, name in enumerate(param_names[:6])
        },
        "metricas_ajuste": {
            "mse": mse,
            "mae": mae,
            "r_squared": r_squared,
            "pearson_rho": float(rho_pearson),
            "spearman_rho": float(rho_spearman),
        },
        "metricas_ponta_a_ponta": {k: float(v) for k, v in metricas_e2e.items()},
        "validacao_n_sims": val_sims,
        "funcao_objetivo": float(result.fun),
    }
    
    with open(OUTPUT_PARAMS, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n💾 Parâmetros salvos em: {OUTPUT_PARAMS}")
    
    # CSV comparativo
    df_comp = pd.DataFrame({
        "team_key": team_keys,
        "forca_alvo": target_forces,
        "forca_modelo": model_forces,
        "erro": np.abs(model_forces - target_forces),
        "prob_mercado": target_probs,
    }).sort_values("prob_mercado", ascending=False)
    
    df_comp.to_csv(OUTPUT_COMPARATIVO, index=False, encoding="utf-8-sig")
    print(f"📊 Comparativo salvo em: {OUTPUT_COMPARATIVO}")
    
    # Tabela resumo
    print("\n" + "=" * 65)
    print("📋 TOP 15 - Força Alvo vs Força do Modelo")
    print("-" * 65)
    print(f"{'Seleção':<25} {'Alvo':>7} {'Modelo':>8} {'Erro':>7}")
    print("-" * 65)
    for _, row in df_comp.head(15).iterrows():
        print(
            f"{row['team_key']:<25} "
            f"{row['forca_alvo']:.4f} "
            f"{row['forca_modelo']:>8.4f} "
            f"{row['erro']:>7.4f}"
        )
    
    print("=" * 65)
    return best_params, param_names


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Estágio 2: pesos que explicam o vetor de força do mercado')
    parser.add_argument('--val-sims', type=int, default=128_000,
                        help='copas na validação ponta a ponta')
    args = parser.parse_args()
    ajustar_parametros(val_sims=args.val_sims)
