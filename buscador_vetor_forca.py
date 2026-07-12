"""
============================================================
ESTÁGIO 1: Buscador do Vetor de Força Ótimo
============================================================
Encontra o vetor de força (48 valores) que, quando usado
na simulação da Copa, reproduz as probabilidades de campeão
do mercado (odds normalizadas).

Algoritmo: Ajuste Iterativo Multiplicativo (Coordinate Descent)
  1. Inicializa forças proporcionais às probabilidades de mercado
  2. Simula N copas → obtém P_sim(Campeão)
  3. Ajusta cada força: f_i *= (P_mercado / P_sim)^alpha
  4. Renormaliza e repete até convergir

Vantagem sobre o Optuna: convergência direcionada, sem busca cega.
============================================================
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import poisson

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from utils.simulador_oficial import simulate_one_cup_oficial, PoissonMatchSimulator
from utils.simulador_oficial import dixon_coles_correction
from experimento_calibracao_mercado import ODDS_PATH, canonical_team_key, load_market_target

# ==========================================
# CONFIGURAÇÃO
# ==========================================
MEDIA_GOLS = 3.0
USAR_DIXON_COLES = True
RHO_DIXON_COLES = -0.13

N_SIMS_POR_ITERACAO = 40_000     # Copas por iteração (mais = menos ruído)
MAX_ITERACOES = 40               # Máximo de iterações
SEED = 42                        # Semente base (reprodutibilidade; varia por iteração)
ALPHA_INICIAL = 0.40             # Learning rate inicial (agressivo no começo)
ALPHA_FINAL = 0.15               # Learning rate final (conservador na convergência)
TOLERANCIA_KL = 0.005            # Critério de parada: KL-Divergence
TOLERANCIA_MAE = 0.003           # Critério de parada alternativo: MAE
FORCA_MINIMA = 0.15              # Piso real do vetor de forca
FORCA_MAXIMA = 1.00              # Teto real do vetor de forca
OFFSET_SIMULACAO = 0.0           # Vetor otimizado e usado puro, sem offset.

OUTPUT_DIR = BASE_DIR / "resultados"
OUTPUT_VETOR = OUTPUT_DIR / "vetor_forca_otimo.json"
OUTPUT_CSV = OUTPUT_DIR / "vetor_forca_otimo.csv"
OUTPUT_LOG = OUTPUT_DIR / "log_busca_vetor.csv"

# ==========================================
# FUNÇÕES DE APOIO
# ==========================================
DATA_DIR = BASE_DIR / "dataset"

def find_latest_enriched_dataset() -> Path:
    candidates = sorted(DATA_DIR.glob("FIFA_ELO_DadosSeleções_*.xlsx"))
    if candidates:
        return candidates[-1]
    fallback = DATA_DIR / "FIFA_ELO_DadosSeleções_2026-04-15.xlsx"
    return fallback


def poisson_matrix(lambda_a, lambda_b, max_goals=10, usar_dixon_coles=False, rho=-0.13):
    goal_range = np.arange(max_goals + 1)
    probs_a = poisson.pmf(goal_range, lambda_a)
    probs_b = poisson.pmf(goal_range, lambda_b)
    probs_a[-1] += max(0.0, 1.0 - probs_a.sum())
    probs_b[-1] += max(0.0, 1.0 - probs_b.sum())
    matrix = np.outer(probs_a, probs_b)
    if usar_dixon_coles:
        for ga in range(max_goals + 1):
            for gb in range(max_goals + 1):
                matrix[ga, gb] *= dixon_coles_correction(ga, gb, lambda_a, lambda_b, rho=rho)
    matrix /= matrix.sum()
    return matrix


def compute_match_data(force_a, force_b, media_gols, max_goals=10, usar_dc=False, rho=-0.13):
    total = force_a + force_b
    share_a = force_a / total if total > 0 else 0.5
    lambda_a = media_gols * share_a
    lambda_b = media_gols * (1.0 - share_a)
    matrix = poisson_matrix(lambda_a, lambda_b, max_goals, usar_dc, rho)
    return {"matrix": matrix, "share_a": share_a, "lambda_a": lambda_a, "lambda_b": lambda_b}


def build_match_cache(team_keys, strengths, media_gols, usar_dc, rho):
    """Constrói o cache de matrizes de Poisson para todos os pares de seleções."""
    cache = {}
    for i, team_a in enumerate(team_keys):
        force_a = strengths[team_a]
        for j, team_b in enumerate(team_keys):
            if i == j:
                continue
            force_b = strengths[team_b]
            cache[(team_a, team_b)] = compute_match_data(
                force_a, force_b, media_gols, 10, usar_dc, rho
            )
    return cache


def simulate_and_count_champions(groups, strengths, match_cache, n_sims, rng=None,
                                 locked_group_results=None,
                                 locked_knockout_results=None):
    """Simula n_sims copas e retorna a contagem de títulos por seleção."""
    if rng is None:
        rng = np.random.default_rng()
    match_simulator = PoissonMatchSimulator(match_cache=match_cache, strengths=strengths)

    champion_counts = {team_key: 0 for team_key in strengths}

    for _ in range(n_sims):
        history = simulate_one_cup_oficial(
            groups,
            strengths,
            rng,
            match_simulator,
            locked_group_results=locked_group_results,
            locked_knockout_results=locked_knockout_results,
        )
        for team_key, stages in history.items():
            if stages.get("Campeao", 0) == 1:
                champion_counts[team_key] += 1
    
    return champion_counts


def compute_metrics(p_market, p_sim, focus_mask=None):
    """Calcula métricas; opcionalmente concentra erro nas seleções positivas."""
    if focus_mask is not None:
        focus_mask = np.asarray(focus_mask, dtype=bool)
        p_market = np.asarray(p_market)[focus_mask]
        p_sim = np.asarray(p_sim)[focus_mask]
    # KL Divergence
    p_m = np.clip(p_market, 1e-10, 1.0)
    p_s = np.clip(p_sim, 1e-10, 1.0)
    p_m_norm = p_m / p_m.sum()
    p_s_norm = p_s / p_s.sum()
    kl_div = float(np.sum(p_m_norm * np.log(p_m_norm / p_s_norm)))
    
    # MAE
    mae = float(np.mean(np.abs(p_sim - p_market)))
    
    # EQM / RMSE
    mse = float(np.mean((p_sim - p_market) ** 2))
    rmse = float(np.sqrt(mse))
    
    # RMSPE
    base = np.maximum(p_market, 0.001)
    rmspe = float(np.sqrt(np.mean(((p_sim - p_market) / base) ** 2)))
    
    # Max absolute error
    max_err = float(np.max(np.abs(p_sim - p_market)))
    
    # Correlação de ranking (Spearman)
    from scipy.stats import spearmanr
    rho_spearman, _ = spearmanr(p_market, p_sim)
    
    return {
        "kl_div": kl_div,
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "rmspe": rmspe,
        "max_err": max_err,
        "spearman_rho": float(rho_spearman),
    }


def normalizar_forcas(forces, forca_minima=FORCA_MINIMA, forca_maxima=FORCA_MAXIMA):
    """Aplica piso e reescala pelo maior valor preservando diferencas no topo."""
    forces = np.asarray(forces, dtype=float)
    forces = np.maximum(forces, forca_minima)
    maximum = float(np.max(forces))
    if maximum > 0:
        forces = forces / maximum * forca_maxima
    return np.clip(forces, forca_minima, forca_maxima)


# ==========================================
# CARREGAMENTO DE DADOS
# ==========================================
def load_data(reference_path=None):
    """Carrega dataset e odds de mercado, retorna dict com tudo necessário."""
    dataset_path = find_latest_enriched_dataset()
    df = pd.read_excel(dataset_path)
    df["team_key"] = df["NomeIngles"].map(canonical_team_key)
    
    odds_path = Path(reference_path) if reference_path else Path(ODDS_PATH)
    odds_df = load_market_target(odds_path)
    df = df.merge(odds_df[["team_key", "market_prob"]], on="team_key", how="left")
    
    groups = df.groupby("Grupo")["team_key"].apply(list).to_dict()
    team_keys = list(df["team_key"])
    market_probs = dict(zip(df["team_key"], df["market_prob"]))
    
    return {
        "df": df,
        "groups": groups,
        "team_keys": team_keys,
        "market_probs": market_probs,
        "reference_path": str(odds_path),
    }


# ==========================================
# HERANÇA DE FORÇA PARA SELEÇÕES ELIMINADAS
# ==========================================
FASES_DIR = OUTPUT_DIR / "vetores_forca"


def carregar_heranca_historica(team_keys, fases_dir=FASES_DIR):
    """Percorre as fases arquivadas (resultados/vetores_forca/NN_fase_data/) em
    ordem cronológica e retorna, por seleção, a força otimizada na ÚLTIMA fase
    em que ela ainda tinha chance real de título (market_prob > 0 na referência
    daquela fase). Serve para que uma seleção eliminada mantenha, no vetor
    ativo, o valor de força da última rodada em que foi otimizada de verdade —
    em vez de cair ao piso — permitindo simular partidas hipotéticas coerentes
    com ela na página de Partida.
    """
    heranca = {}
    if not fases_dir.exists():
        return heranca
    for fase_dir in sorted(fases_dir.iterdir()):
        vetor_path = fase_dir / "vetor_forca_otimo.json"
        if not vetor_path.exists():
            continue
        try:
            with open(vetor_path, "r", encoding="utf-8") as f:
                fase_data = json.load(f)
            vetor_forca = fase_data.get("vetor_forca", {})
            ref = fase_data.get("parametros_simulacao", {}).get("referencia_probabilidades")
            if not ref or not vetor_forca:
                continue
            ref_path = Path(ref)
            if not ref_path.is_absolute():
                ref_path = BASE_DIR / ref_path
            if not ref_path.exists():
                continue
            odds_fase = load_market_target(ref_path)
            market_probs_fase = dict(zip(odds_fase["team_key"], odds_fase["market_prob"]))
        except Exception as exc:
            print(f"   ⚠️ Herança: pulando fase {fase_dir.name} ({exc})")
            continue
        for tk in team_keys:
            if market_probs_fase.get(tk, 0) > 0 and tk in vetor_forca:
                heranca[tk] = float(vetor_forca[tk])
    return heranca


# ==========================================
# ALGORITMO PRINCIPAL: COORDINATE DESCENT
# ==========================================
def buscar_vetor_forca(reference_path=None, vetor_inicial=None, travar_resultados=False,
                       usar_crn=False, alpha_ini=None, alpha_fim=None,
                       tol_kl=None, tol_mae=None, foco_positivas=False, sem_heranca=False):
    alpha_ini = ALPHA_INICIAL if alpha_ini is None else alpha_ini
    alpha_fim = ALPHA_FINAL if alpha_fim is None else alpha_fim
    tol_kl = TOLERANCIA_KL if tol_kl is None else tol_kl
    tol_mae = TOLERANCIA_MAE if tol_mae is None else tol_mae

    print("=" * 65)
    print("🎯 ESTÁGIO 1: Buscando Vetor de Força Ótimo")
    print(f"   Simulações por iteração: {N_SIMS_POR_ITERACAO:,}")
    print(f"   Máximo de iterações:     {MAX_ITERACOES}")
    print(f"   Alpha: {alpha_ini:.2f} → {alpha_fim:.2f}")
    print(f"   Tolerância KL/MAE:       {tol_kl} / {tol_mae}")
    print(f"   CRN (semente fixa):      {usar_crn}")
    print("   Offset:                  0.00 (vetor puro, sem soma)")
    print("=" * 65)

    data = load_data(reference_path)
    groups = data["groups"]
    team_keys = data["team_keys"]
    market_probs = data["market_probs"]
    print(f"   Referencia:              {data['reference_path']}")

    # Resultados oficiais já encerrados: trava os jogos de grupo na simulação,
    # exatamente como o app (página Simulação) faz ao reproduzir o mercado.
    locked_group_results = None
    locked_knockout_results = None
    n_travados = 0
    n_mata_mata_travados = 0
    if travar_resultados:
        from utils.resultados_oficiais import load_official_group_results
        oficiais = load_official_group_results(data["df"])
        locked_group_results = oficiais.locked_group_results
        locked_knockout_results = oficiais.locked_knockout_results
        n_travados = oficiais.locked_match_count
        n_mata_mata_travados = oficiais.locked_knockout_count
        print(f"   Resultados travados:     {n_travados} jogo(s) de grupo + {n_mata_mata_travados} mata-mata")
        if oficiais.unmatched_rows:
            print(f"   AVISO — linhas nao casadas: {oficiais.unmatched_rows}")

    # Vetor de probabilidades de mercado (ordenado por team_key)
    p_market = np.array([market_probs[tk] for tk in team_keys])
    positive_mask = p_market > 0 if foco_positivas else None
    if foco_positivas:
        print(f"   Métricas focadas:        {int(positive_mask.sum())} seleções positivas")

    # -------------------------------------------------------
    # INICIALIZAÇÃO: warm-start de um vetor já otimizado (poucas iterações para
    # refinar), ou forças proporcionais ao mercado (raiz p/ não exagerar no início,
    # já que a relação força→prob é não-linear e amplifica diferenças).
    # -------------------------------------------------------
    if vetor_inicial:
        with open(vetor_inicial, "r", encoding="utf-8") as f:
            vec0 = json.load(f)["vetor_forca"]
        forces = np.array([float(vec0[tk]) for tk in team_keys])
        print(f"   Warm-start de:           {Path(vetor_inicial).name}")
    else:
        forces = np.sqrt(p_market / p_market.max())
    forces = normalizar_forcas(forces)

    # Log de iterações
    iteration_log = []
    best_mse = float("inf")
    best_forces = forces.copy()
    best_iter = 0
    stagnation_count = 0
    
    start_time = time.time()
    
    for iteration in range(1, MAX_ITERACOES + 1):
        # Learning rate com decay linear
        alpha = alpha_ini - (alpha_ini - alpha_fim) * (iteration - 1) / max(1, MAX_ITERACOES - 1)
        
        # Construir strengths dict com a forca pura
        strengths = {tk: float(forces[i]) for i, tk in enumerate(team_keys)}
        
        # Construir cache de partidas
        match_cache = build_match_cache(team_keys, strengths, MEDIA_GOLS, USAR_DIXON_COLES, RHO_DIXON_COLES)
        
        # Simular. Sem CRN: semente varia por iteração (ruído se compensa entre
        # iterações). Com CRN: semente fixa — as MESMAS copas aleatórias em toda
        # iteração, então a única mudança é a força (variância da diferença ~0,
        # convergência fina). A validação final usa semente independente.
        rng_iter = np.random.default_rng(SEED if usar_crn else SEED * 100_000 + iteration)
        champion_counts = simulate_and_count_champions(
            groups, strengths, match_cache, N_SIMS_POR_ITERACAO, rng=rng_iter,
            locked_group_results=locked_group_results,
            locked_knockout_results=locked_knockout_results)
        
        # Probabilidades simuladas
        p_sim = np.array([champion_counts[tk] / N_SIMS_POR_ITERACAO for tk in team_keys])
        
        # Métricas
        metrics = compute_metrics(p_market, p_sim, focus_mask=positive_mask)
        
        # Tempo
        elapsed = time.time() - start_time
        avg_iter_time = elapsed / iteration
        eta = avg_iter_time * (MAX_ITERACOES - iteration)
        
        # Log
        iteration_log.append({
            "iteracao": iteration,
            "alpha": alpha,
            **metrics,
            "tempo_acumulado_s": elapsed,
        })
        
        # Melhor resultado?
        if metrics["mse"] < best_mse:
            best_mse = metrics["mse"]
            best_forces = forces.copy()
            best_iter = iteration
            stagnation_count = 0
            marker = "⭐"
        else:
            stagnation_count += 1
            marker = "  "
        
        # Print
        eta_min = int(eta // 60)
        eta_sec = int(eta % 60)
        print(
            f"{marker} Iter {iteration:03d}/{MAX_ITERACOES} | "
            f"EQM: {metrics['mse']:.8f} | KL: {metrics['kl_div']:.5f} | MAE: {metrics['mae']:.5f} | "
            f"RMSPE: {metrics['rmspe']:.4f} | ρ: {metrics['spearman_rho']:.4f} | "
            f"α: {alpha:.3f} | {avg_iter_time:.1f}s/iter | ETA: {eta_min:02d}:{eta_sec:02d}"
        )
        
        # Critério de parada
        if metrics["kl_div"] < tol_kl and metrics["mae"] < tol_mae:
            print(f"\n✅ Convergiu na iteração {iteration}! KL={metrics['kl_div']:.6f}, MAE={metrics['mae']:.6f}")
            break
        
        # Se estagnou demais, parar
        if stagnation_count >= 12:
            print(f"\n⚠️ Sem melhoria por {stagnation_count} iterações. Usando melhor resultado (iter {best_iter}).")
            forces = best_forces.copy()
            break
        
        # -------------------------------------------------------
        # ATUALIZAÇÃO MULTIPLICATIVA
        # f_i *= (P_mercado_i / P_sim_i)^alpha
        # -------------------------------------------------------
        p_sim_safe = np.clip(p_sim, 1e-8, 1.0)
        ratio = p_market / p_sim_safe
        
        # Clipar o ratio para evitar saltos absurdos (especialmente para azarões)
        ratio_clipped = np.clip(ratio, 0.3, 3.0)
        
        # Aplicar ajuste
        forces = forces * (ratio_clipped ** alpha)

        # Renormalizar: pior=0.15 e melhor=1.00
        forces = normalizar_forcas(forces)
    
    else:
        print(f"\n⚠️ Atingiu o limite de {MAX_ITERACOES} iterações. Usando melhor resultado (iter {best_iter}).")
        forces = best_forces.copy()
    
    # ==========================================
    # VALIDAÇÃO FINAL: simular com o vetor ótimo
    # ==========================================
    print("\n" + "=" * 65)
    print("🔍 VALIDAÇÃO FINAL (simulação com o vetor ótimo)")
    
    strengths_final = {tk: float(forces[i]) for i, tk in enumerate(team_keys)}
    match_cache_final = build_match_cache(team_keys, strengths_final, MEDIA_GOLS, USAR_DIXON_COLES, RHO_DIXON_COLES)
    
    # Simular mais copas para validação
    n_validacao = N_SIMS_POR_ITERACAO * 2
    print(f"   Simulando {n_validacao:,} copas de validação...")
    champion_counts_final = simulate_and_count_champions(
        groups, strengths_final, match_cache_final, n_validacao,
        rng=np.random.default_rng(SEED - 1),
        locked_group_results=locked_group_results,
        locked_knockout_results=locked_knockout_results)
    p_sim_final = np.array([champion_counts_final[tk] / n_validacao for tk in team_keys])
    metrics_final = compute_metrics(p_market, p_sim_final, focus_mask=positive_mask)
    
    print(f"   KL-Div:      {metrics_final['kl_div']:.6f}")
    print(f"   MAE:         {metrics_final['mae']:.6f}")
    print(f"   EQM:         {metrics_final['mse']:.8f}")
    print(f"   RMSE:        {metrics_final['rmse']:.6f}")
    print(f"   RMSPE:       {metrics_final['rmspe']:.4f}")
    print(f"   Spearman ρ:  {metrics_final['spearman_rho']:.4f}")

    # ==========================================
    # HERANÇA: seleções eliminadas mantêm a força da última rodada em que
    # ainda tinham chance real, em vez de cair ao piso.
    # ==========================================
    vetor_forca_bruto = {tk: float(forces[i]) for i, tk in enumerate(team_keys)}
    heranca = carregar_heranca_historica(team_keys) if not sem_heranca else {}
    vetor_forca_final = dict(vetor_forca_bruto)
    n_herdados = 0
    for i, tk in enumerate(team_keys):
        if p_market[i] <= 0 and tk in heranca:
            vetor_forca_final[tk] = heranca[tk]
            n_herdados += 1
    if not sem_heranca:
        print(f"   Herança de eliminados:   {n_herdados} seleção(ões) usando força da última rodada em que jogaram")

    # ==========================================
    # SALVAR RESULTADOS
    # ==========================================
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. JSON com o vetor completo
    vetor_resultado = {
        "descricao": "Vetor de força ótimo encontrado pelo Coordinate Descent",
        "parametros_simulacao": {
            "media_gols": MEDIA_GOLS,
            "dixon_coles": USAR_DIXON_COLES,
            "rho": RHO_DIXON_COLES,
            "offset": OFFSET_SIMULACAO,
            "forca_minima": FORCA_MINIMA,
            "forca_maxima": FORCA_MAXIMA,
            "criterio_otimizacao": "mse",
            "ajuste": "multiplicativo_livre_clip015_max1",
            "n_sims_por_iteracao": N_SIMS_POR_ITERACAO,
            "iteracao_convergencia": best_iter,
            "referencia_probabilidades": data["reference_path"],
            "warm_start_de": vetor_inicial,
            "crn": usar_crn,
            "foco_positivas": foco_positivas,
            "travou_rodada_grupos": travar_resultados,
            "jogos_grupo_travados": n_travados,
            "travou_mata_mata": travar_resultados,
            "jogos_mata_mata_travados": n_mata_mata_travados,
            "heranca_forca_eliminados": not sem_heranca,
            "n_selecoes_herdadas": n_herdados,
        },
        "metricas_finais": metrics_final,
        # vetor_forca: saida bruta do Coordinate Descent (mantida como estava —
        # e o alvo que ajustar_parametros_ao_vetor.py/otimizador_optuna.py usam
        # para calibrar as 6 variaveis, entao NAO pode conter heranca).
        "vetor_forca": vetor_forca_bruto,
        # vetor_forca_com_heranca: o que o app (forca_core) usa de fato. Selecoes
        # ainda vivas = forca otimizada nesta rodada; selecoes ja eliminadas =
        # forca herdada da ultima rodada em que tinham chance real (ver
        # carregar_heranca_historica), para a pagina de Partida continuar
        # simulando confrontos hipoteticos com elas de forma coerente com a
        # forca historica, em vez de todo eliminado cair no mesmo piso.
        "vetor_forca_com_heranca": vetor_forca_final,
    }
    
    with open(OUTPUT_VETOR, "w", encoding="utf-8") as f:
        json.dump(vetor_resultado, f, ensure_ascii=False, indent=2)
    print(f"\n💾 Vetor de força salvo em: {OUTPUT_VETOR}")
    
    # 2. CSV comparativo
    df_resultado = pd.DataFrame({
        "team_key": team_keys,
        "forca": forces,
        "forca_final_com_heranca": [vetor_forca_final[tk] for tk in team_keys],
        "herdada_de_rodada_anterior": [
            bool(p_market[i] <= 0 and team_keys[i] in heranca) for i in range(len(team_keys))
        ],
        "prob_mercado": p_market,
        "prob_simulada": p_sim_final,
        "erro_absoluto": np.abs(p_sim_final - p_market),
    }).sort_values("prob_mercado", ascending=False)
    
    df_resultado.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"📊 Comparativo salvo em: {OUTPUT_CSV}")
    
    # 3. Log de iterações
    pd.DataFrame(iteration_log).to_csv(OUTPUT_LOG, index=False, encoding="utf-8-sig")
    print(f"📈 Log de iterações salvo em: {OUTPUT_LOG}")
    
    # 4. Tabela resumo no console
    print("\n" + "=" * 65)
    print("📋 TOP 15 - Comparativo Mercado vs Simulação")
    print("-" * 65)
    print(f"{'Seleção':<22} {'Força':>7} {'Mercado':>8} {'Simulado':>9} {'Erro':>7}")
    print("-" * 65)
    for _, row in df_resultado.head(15).iterrows():
        print(
            f"{row['team_key']:<22} "
            f"{row['forca']:>7.4f} "
            f"{row['prob_mercado']*100:>7.2f}% "
            f"{row['prob_simulada']*100:>8.2f}% "
            f"{row['erro_absoluto']*100:>6.2f}%"
        )
    
    total_time = time.time() - start_time
    print(f"\n⏱️ Tempo total: {int(total_time//60)}min {int(total_time%60)}s")
    print("=" * 65)
    
    return forces, team_keys


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Estágio 1: vetor de força que reproduz o mercado')
    parser.add_argument('--sims', type=int, default=N_SIMS_POR_ITERACAO,
                        help='copas simuladas por iteração')
    parser.add_argument('--max-iter', type=int, default=MAX_ITERACOES,
                        help='máximo de iterações')
    parser.add_argument('--referencia', type=str, default=None,
                        help='arquivo XLSX/CSV com probabilidades de referencia')
    parser.add_argument('--vetor-inicial', type=str, default=None,
                        help='JSON de vetor de força para warm-start (parte daqui)')
    parser.add_argument('--travar-resultados', action='store_true',
                        help='trava os jogos de grupo já encerrados (assets/resultados_jogos.json)')
    parser.add_argument('--crn', action='store_true',
                        help='common random numbers: semente fixa por iteração (convergência fina)')
    parser.add_argument('--alpha-inicial', type=float, default=None, help='learning rate inicial')
    parser.add_argument('--alpha-final', type=float, default=None, help='learning rate final')
    parser.add_argument('--tol-kl', type=float, default=None, help='tolerância de parada (KL)')
    parser.add_argument('--tol-mae', type=float, default=None, help='tolerância de parada (MAE)')
    parser.add_argument('--foco-positivas', action='store_true',
                        help='calcula métricas de erro apenas nas seleções com mercado positivo')
    parser.add_argument('--sem-heranca', action='store_true',
                        help='desliga a herança de força para eliminados (usa o piso puro, comportamento antigo)')
    args = parser.parse_args()
    N_SIMS_POR_ITERACAO = args.sims
    MAX_ITERACOES = args.max_iter
    buscar_vetor_forca(
        reference_path=args.referencia,
        vetor_inicial=args.vetor_inicial,
        travar_resultados=args.travar_resultados,
        usar_crn=args.crn,
        alpha_ini=args.alpha_inicial,
        alpha_fim=args.alpha_final,
        tol_kl=args.tol_kl,
        tol_mae=args.tol_mae,
        foco_positivas=args.foco_positivas,
        sem_heranca=args.sem_heranca,
    )
