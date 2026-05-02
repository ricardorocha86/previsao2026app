import os
import sys
import time
import numpy as np
import pandas as pd
from scipy.stats import poisson
from pathlib import Path

# Adiciona o diretório raiz ao path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

try:
    import optuna
except ImportError:
    print("ERRO: Optuna não encontrado. Por favor, instale usando: pip install optuna")
    sys.exit(1)

from utils.simulador_oficial import simulate_one_cup_oficial, PoissonMatchSimulator
from utils.simulador_oficial import dixon_coles_correction, parse_world_cup_score
from experimento_calibracao_mercado import ODDS_PATH, canonical_team_key, load_market_target

# ==========================================
# PARÂMETROS PRINCIPAIS DA OTIMIZAÇÃO
# ==========================================
N_SIMULACOES_POR_TRIAL = 32000 # 48 mil copas garantem estabilidade total na cauda de azarões
N_TRIALS = 300                # Rodando isso à noite, garantimos encontrar o ótimo global
MEDIA_GOLS = 3
USAR_DIXON_COLES = True
RHO_DIXON_COLES = -0.13

SIM_STAGE_COLUMNS = ["pos_1", "pos_2", "pos_3", "pos_4", "Top32", "Oitavas", "Quartas", "Semifinal", "Final", "Campeao"]

# ==========================================
# FUNÇÕES DE APOIO (Copiadas para não depender do Streamlit)
# ==========================================
DATA_DIR = BASE_DIR / "dados"

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

def load_force_table(dataset_path: str) -> pd.DataFrame:
    df = pd.read_excel(dataset_path)
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
    result["world_cup_history_01"] = (0.5 * result["world_cup_apps_01"] + 0.5 * result["world_cup_best_raw"])

    odds_df = load_market_target(ODDS_PATH)
    result = result.merge(odds_df[["team_key", "market_prob"]], on="team_key", how="left")
    return result

def build_combined_table(dataframe, weight_fifa, weight_elo, weight_momentum, weight_market, weight_history, offset, elasticidade, weight_host=0.0):
    result = dataframe.copy()
    weight_sum = weight_fifa + weight_elo + weight_momentum + weight_market + weight_history + weight_host

    if weight_sum > 0:
        result["forca_resultante_01"] = (
            weight_fifa * result["fifa_force_01"] +
            weight_elo * result["elo_force_01"] +
            weight_momentum * result["momentum_force_01"] +
            weight_market * result["market_force_01"] +
            weight_history * result["world_cup_history_01"] +
            weight_host * result["is_host"]
        ) / weight_sum
    else:
        result["forca_resultante_01"] = 0.0

    max_force = float(result["forca_resultante_01"].max())
    if max_force > 0:
        result["forca_resultante_01"] = result["forca_resultante_01"] / max_force

    result["forca_elastica"] = result["forca_resultante_01"] ** elasticidade
    result["forca_com_offset"] = offset + result["forca_elastica"]
    
    result = result.sort_values(by=["forca_resultante_01"], ascending=False).reset_index(drop=True)
    return result

def poisson_matrix(lambda_a, lambda_b, max_goals=10, usar_dixon_coles=False, rho_dixon_coles=-0.13):
    goal_range = np.arange(max_goals + 1)
    probs_a = poisson.pmf(goal_range, lambda_a)
    probs_b = poisson.pmf(goal_range, lambda_b)

    probs_a[-1] += max(0.0, 1.0 - probs_a.sum())
    probs_b[-1] += max(0.0, 1.0 - probs_b.sum())

    matrix = np.outer(probs_a, probs_b)
    if usar_dixon_coles:
        for goals_a in range(max_goals + 1):
            for goals_b in range(max_goals + 1):
                matrix[goals_a, goals_b] *= dixon_coles_correction(goals_a, goals_b, lambda_a, lambda_b, rho=rho_dixon_coles)
    matrix /= matrix.sum()
    return matrix

def compute_match_probabilities(force_a, force_b, media_gols, max_goals=10, usar_dixon_coles=False, rho_dixon_coles=-0.13):
    total_force = force_a + force_b
    share_a = force_a / total_force if total_force > 0 else 0.5
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
    return {"matrix": matrix, "share_a": share_a, "lambda_a": lambda_a, "lambda_b": lambda_b}

def build_match_cache(dataframe, media_gols, usar_dixon_coles, rho_dixon_coles):
    cache = {}
    rows = dataframe.set_index("team_key")[["forca_com_offset"]]
    team_keys = list(rows.index)
    for team_a in team_keys:
        force_a = float(rows.loc[team_a, "forca_com_offset"])
        for team_b in team_keys:
            if team_a == team_b: continue
            force_b = float(rows.loc[team_b, "forca_com_offset"])
            cache[(team_a, team_b)] = compute_match_probabilities(force_a, force_b, media_gols, 10, usar_dixon_coles, rho_dixon_coles)
    return cache

# Carregamento global de dados para não ler do disco em cada Trial
GLOBAL_DATASET_PATH = find_latest_enriched_dataset()
GLOBAL_BASE_DF = load_force_table(str(GLOBAL_DATASET_PATH))

# ==========================================
# FUNÇÃO OBJETIVO DO OPTUNA
# ==========================================
def objective(trial):
    # 1. Sugerir pesos continuamente com passos definidos (max 2 casas decimais)
    w_fifa = trial.suggest_float("peso_fifa", 0.0, 1.0, step=0.01)
    w_elo = trial.suggest_float("peso_elo", 0.0, 1.0, step=0.01)
    w_momentum = trial.suggest_float("peso_momentum", 0.0, 1.0, step=0.01)
    w_market = trial.suggest_float("peso_mercado", 0.0, 1.0, step=0.01)
    w_history = trial.suggest_float("peso_historico", 0.0, 1.0, step=0.01)
    w_host = trial.suggest_float("peso_anfitriao", 0.0, 1.0, step=0.01)
    
    soma = w_fifa + w_elo + w_momentum + w_market + w_history + w_host
    if soma == 0: return float('inf')
    
    # 2. Sugerir hiperparâmetros
    elasticidade = trial.suggest_float("elasticidade", 1.0, 2.0, step=0.05)
    offset = trial.suggest_float("offset", 0.0, 0.20, step=0.01)
    
    # 3. Gerar Dataframe Final
    combined_df = build_combined_table(
        GLOBAL_BASE_DF, w_fifa, w_elo, w_momentum, w_market, w_history, offset, elasticidade, w_host
    )
    
    groups = combined_df.groupby("Grupo")["team_key"].apply(list).to_dict()
    strengths = dict(zip(combined_df["team_key"], combined_df["forca_com_offset"]))
    
    match_cache = build_match_cache(combined_df, MEDIA_GOLS, USAR_DIXON_COLES, RHO_DIXON_COLES)
    rng = np.random.default_rng()
    match_simulator = PoissonMatchSimulator(match_cache=match_cache, strengths=strengths)
    
    # Pre-aloca um dicionário rápido de históricos (otimização já implementada)
    accumulated_campeao = {team_key: 0 for team_key in combined_df["team_key"]}
    
    # 4. Rodar as Simulações
    for _ in range(N_SIMULACOES_POR_TRIAL):
        history = simulate_one_cup_oficial(groups, strengths, rng, match_simulator)
        # O histórico possui 1 nos estágios que o time alcançou
        for team_key, stages in history.items():
            if stages.get("Campeao", 0) == 1:
                accumulated_campeao[team_key] += 1
                
    # 5. Avaliar o Erro (Múltiplas Métricas para o Tabelão)
    y_true_list = []
    y_pred_list = []
    
    for idx, row in combined_df.iterrows():
        team_key = row["team_key"]
        prob_mercado = row["market_prob"]
        
        if pd.isna(prob_mercado):
            continue
            
        prob_simulada = accumulated_campeao[team_key] / N_SIMULACOES_POR_TRIAL
        y_true_list.append(prob_mercado)
        y_pred_list.append(prob_simulada)
        
    y_true = np.array(y_true_list)
    y_pred = np.array(y_pred_list)
    
    # Base divisor (piso) para as métricas relativas (evitar divisão por quase-zero)
    base_divisor = np.maximum(y_true, 0.001)
    
    # Cálculos das Métricas
    rmspe = np.sqrt(np.mean(((y_pred - y_true) / base_divisor) ** 2))
    mse = np.mean((y_pred - y_true) ** 2)
    mae = np.mean(np.abs(y_pred - y_true))
    mape = np.mean(np.abs((y_pred - y_true) / base_divisor))
    
    # KL Divergence
    y_true_norm = y_true / np.sum(y_true)
    y_pred_safe = np.clip(y_pred, 1e-10, 1.0)
    y_pred_norm = y_pred_safe / np.sum(y_pred_safe)
    kl_div = np.sum(y_true_norm * np.log(y_true_norm / y_pred_norm))
    
    # Grava as métricas na memória do Trial para aparecerem na planilha depois
    trial.set_user_attr("RMSPE", rmspe)
    trial.set_user_attr("MSE", mse)
    trial.set_user_attr("MAE", mae)
    trial.set_user_attr("MAPE", mape)
    trial.set_user_attr("KL_Div", kl_div)
    
    return rmspe

# ==========================================
# EXECUÇÃO PRINCIPAL
# ==========================================
if __name__ == "__main__":
    print("="*60)
    print("🏆 INICIANDO BUSCA BAYESIANA (OPTUNA) - COPA 2026")
    print(f"Simulações por Trial: {N_SIMULACOES_POR_TRIAL:,}")
    print(f"Total de Trials:      {N_TRIALS}")
    print("="*60)
    
    # Desliga os logs super detalhados e feios do Optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    study = optuna.create_study(direction="minimize", study_name="calibracao_copa_2026")
    
    # ----------------------------------------------------
    # CHUTE INICIAL (Dando um "norte" para o algoritmo)
    # Valores extraídos da imagem do usuário (Pesos: 0, 1, 0.5, 1, 1 - normalizados para somar 1)
    # ----------------------------------------------------
    parametros_iniciais = {
        "peso_fifa": 0.00,
        "peso_elo": 0.29,       # 1.0 / 3.5
        "peso_momentum": 0.14,  # 0.5 / 3.5
        "peso_mercado": 0.29,   # 1.0 / 3.5
        "peso_historico": 0.28, # 1.0 / 3.5
        "peso_anfitriao": 0.00,
        "elasticidade": 1.20,
        "offset": 0.10
    }
    study.enqueue_trial(parametros_iniciais)
    print("📍 Tentativa 000 carregada com parâmetros iniciais (da interface).")
    
    START_TIME = time.time()
    
    # Callback para printar a cada tentativa de forma limpa, mostrando os parâmetros
    def print_callback(study, trial):
        p = trial.params
        w_sum = sum([p.get(k, 0) for k in ["peso_fifa", "peso_elo", "peso_momentum", "peso_mercado", "peso_historico", "peso_anfitriao"]])
        if w_sum == 0: w_sum = 1.0
        
        # Pesos finalizados e normalizados
        f = p.get('peso_fifa',0)/w_sum
        e = p.get('peso_elo',0)/w_sum
        mo = p.get('peso_momentum',0)/w_sum
        me = p.get('peso_mercado',0)/w_sum
        h = p.get('peso_historico',0)/w_sum
        anf = p.get('peso_anfitriao',0)/w_sum
        
        elast = p.get('elasticidade', 1.0)
        off = p.get('offset', 0.0)
        
        # Métricas de Tempo
        elapsed = time.time() - START_TIME
        avg_time = elapsed / (trial.number + 1)
        remaining = N_TRIALS - (trial.number + 1)
        eta_s = avg_time * remaining
        
        h_eta, m_eta = int(eta_s // 3600), int((eta_s % 3600) // 60)
        s_eta = int(eta_s % 60)
        
        best_val = study.best_value
        
        print(
            f"Trial {trial.number:03d}/{N_TRIALS-1} | "
            f"RMSPE: {trial.value:.4f} (Melhor: {best_val:.4f}) | "
            f"Avg: {avg_time:.1f}s | ETA: {h_eta:02d}:{m_eta:02d}:{s_eta:02d}"
        )
        print(
            f"  └─ Parâmetros: {{'peso_fifa': {f:.3f}, 'peso_elo': {e:.3f}, 'peso_momentum': {mo:.3f}, "
            f"'peso_mercado': {me:.3f}, 'peso_historico': {h:.3f}, 'peso_anfitriao': {anf:.3f}, "
            f"'elasticidade': {elast:.3f}, 'offset': {off:.3f}}}"
        )
    
    try:
        # Removido o show_progress_bar para o nosso print ficar mais limpo
        study.optimize(objective, n_trials=N_TRIALS, callbacks=[print_callback])
    except KeyboardInterrupt:
        print("\nOtimização interrompida pelo usuário.")
        
    print("\n" + "="*60)
    print("✨ OTIMIZAÇÃO CONCLUÍDA ✨")
    print(f"🎯 Melhor Erro (RMSPE): {study.best_value:.4f}")
    print("\n👑 MELHORES PARÂMETROS ENCONTRADOS:")
    
    best_params = study.best_params
    w_sum = (best_params["peso_fifa"] + best_params["peso_elo"] + 
             best_params["peso_momentum"] + best_params["peso_mercado"] + 
             best_params["peso_historico"] + best_params["peso_anfitriao"])
             
    print(f"  • Peso FIFA:      {best_params['peso_fifa']/w_sum * 100:>6.2f}%")
    print(f"  • Peso ELO:       {best_params['peso_elo']/w_sum * 100:>6.2f}%")
    print(f"  • Peso Momento:   {best_params['peso_momentum']/w_sum * 100:>6.2f}%")
    print(f"  • Peso Mercado:   {best_params['peso_mercado']/w_sum * 100:>6.2f}%")
    print(f"  • Peso Histórico: {best_params['peso_historico']/w_sum * 100:>6.2f}%")
    print(f"  • Peso Anfitrião: {best_params['peso_anfitriao']/w_sum * 100:>6.2f}%")
    print(f"  • Elasticidade:   {best_params['elasticidade']:>6.3f}")
    print(f"  • Offset:         {best_params['offset']:>6.3f}")
    print("="*60)
    
    df_trials = study.trials_dataframe()
    
    # Limpeza e formatação das colunas para o Tabelão Excel
    rename_cols = {c: c.replace("user_attrs_", "") for c in df_trials.columns if "user_attrs_" in c}
    rename_cols.update({c: c.replace("params_", "") for c in df_trials.columns if "params_" in c})
    df_trials = df_trials.rename(columns=rename_cols)
    
    # Reordenar colunas (colocar métricas logo após o RMSPE/value)
    cols = list(df_trials.columns)
    metric_cols = ["RMSPE", "MSE", "MAE", "MAPE", "KL_Div"]
    for m in metric_cols:
        if m in cols: cols.remove(m)
        
    if "value" in cols:
        val_idx = cols.index("value")
        cols = cols[:val_idx+1] + metric_cols + cols[val_idx+1:]
        
    df_trials = df_trials[cols]
    
    output_file = "tabelao_experimento_copa.xlsx"
    df_trials.to_excel(output_file, index=False)
    print(f"📊 Tabelão Completo de Resultados salvo em: {output_file}")
