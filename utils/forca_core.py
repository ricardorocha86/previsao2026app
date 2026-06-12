"""Núcleo compartilhado entre as páginas de Indicador de Força, Partida e Simulação.

Reúne a carga da base enriquecida, a construção da tabela de força, o cálculo de
probabilidades de partida (Poisson + Dixon-Coles) e a renderização dos parâmetros
do modelo na barra lateral — que ficam compartilhados entre as três páginas.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import poisson

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experimento_calibracao_mercado import ODDS_PATH, canonical_team_key, load_market_target
from utils import config as app_config
from utils.simulador_oficial import dixon_coles_correction, parse_world_cup_score


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


# Emoji de bandeira por seleção (chave = coluna "Seleção" da planilha).
# Observação: no Chrome/Edge do Windows os emojis de bandeira de país costumam
# aparecer como o código de duas letras (ex.: "FR") em vez da bandeira; no Firefox
# e na maioria dos navegadores mobile/macOS renderizam como bandeira.
TEAM_FLAG_EMOJI: dict[str, str] = {
    "México": "🇲🇽",
    "Canadá": "🇨🇦",
    "Estados Unidos": "🇺🇸",
    "África do Sul": "🇿🇦",
    "Coreia do Sul": "🇰🇷",
    "Catar": "🇶🇦",
    "Suíça": "🇨🇭",
    "Brasil": "🇧🇷",
    "Escócia": "🏴󠁧󠁢󠁳󠁣󠁴󠁿",
    "Haiti": "🇭🇹",
    "Marrocos": "🇲🇦",
    "Austrália": "🇦🇺",
    "Paraguai": "🇵🇾",
    "Alemanha": "🇩🇪",
    "Costa do Marfim": "🇨🇮",
    "Curaçau": "🇨🇼",
    "Curaçao": "🇨🇼",
    "Equador": "🇪🇨",
    "Holanda": "🇳🇱",
    "Japão": "🇯🇵",
    "Tunísia": "🇹🇳",
    "Bélgica": "🇧🇪",
    "Egito": "🇪🇬",
    "Irã": "🇮🇷",
    "Nova Zelândia": "🇳🇿",
    "Arábia Saudita": "🇸🇦",
    "Cabo Verde": "🇨🇻",
    "Espanha": "🇪🇸",
    "Uruguai": "🇺🇾",
    "França": "🇫🇷",
    "Noruega": "🇳🇴",
    "Senegal": "🇸🇳",
    "Argélia": "🇩🇿",
    "Argentina": "🇦🇷",
    "Áustria": "🇦🇹",
    "Jordânia": "🇯🇴",
    "Colômbia": "🇨🇴",
    "Portugal": "🇵🇹",
    "Uzbequistão": "🇺🇿",
    "Croácia": "🇭🇷",
    "Gana": "🇬🇭",
    "Inglaterra": "🏴󠁧󠁢󠁥󠁮󠁧󠁿",
    "Panamá": "🇵🇦",
    "Iraque": "🇮🇶",
    "RD do Congo": "🇨🇩",
    "Tcheca": "🇨🇿",
    "Bósnia e Herzegovina": "🇧🇦",
    "Turquia": "🇹🇷",
    "Suécia": "🇸🇪",
}


def team_with_flag(team_name: str) -> str:
    """Devolve o nome da seleção precedido pelo emoji de bandeira, quando houver."""
    flag = TEAM_FLAG_EMOJI.get(team_name)
    return f"{flag} {team_name}" if flag else team_name


@dataclass
class ModelParams:
    """Parâmetros de força + modelo compartilhados pela barra lateral."""

    weight_fifa: float
    weight_elo: float
    weight_momentum: float
    weight_market: float
    weight_history: float
    weight_host: float
    media_gols: float
    offset: float
    elasticidade: float
    usar_dixon_coles: bool
    rho_dixon_coles: float
    usar_vetor_otimizado: bool = False


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
        "ELO_Chg_2A",
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
    result["momentum_force_01"] = minmax_scale(result["ELO_Chg_2A"])
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


OPTIMIZED_FORCE_VECTOR_PATH = BASE_DIR / "resultados" / "vetor_forca_otimo.json"


@st.cache_data
def _load_optimized_force_vector_cached(path: str, mtime: float) -> dict:
    # mtime entra na chave do cache: se o arquivo for reescrito (ex.: re-rodar
    # o buscador), o cache invalida sozinho e o vetor novo é carregado.
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    vector = data.get("vetor_forca")
    if not isinstance(vector, dict) or not vector:
        raise ValueError("Arquivo do vetor otimizado nao contem 'vetor_forca'.")
    return data


def load_optimized_force_vector(path: str | None = None) -> dict:
    """Carrega o vetor de força otimizado (cache invalida quando o arquivo muda)."""
    vector_path = Path(path) if path else OPTIMIZED_FORCE_VECTOR_PATH
    return _load_optimized_force_vector_cached(str(vector_path), vector_path.stat().st_mtime)


def _team_key_series(dataframe: pd.DataFrame) -> pd.Series:
    if "team_key" in dataframe.columns:
        return dataframe["team_key"].astype(str).map(canonical_team_key)
    if "NomeIngles" in dataframe.columns:
        return dataframe["NomeIngles"].astype(str).map(canonical_team_key)
    if "Seleção" in dataframe.columns:
        return dataframe["Seleção"].astype(str).map(canonical_team_key)
    if "SeleÃ§Ã£o" in dataframe.columns:
        return dataframe["SeleÃ§Ã£o"].astype(str).map(canonical_team_key)
    raise ValueError("DataFrame sem coluna de selecao para aplicar o vetor otimizado.")


def build_optimized_force_table(dataframe: pd.DataFrame, vector_data: dict | None = None) -> tuple[pd.DataFrame, float]:
    result = dataframe.copy()
    vector_data = vector_data or load_optimized_force_vector()
    vector = {canonical_team_key(str(key)): float(value) for key, value in vector_data["vetor_forca"].items()}
    team_keys = _team_key_series(result)
    missing = sorted(set(team_keys) - set(vector))
    if missing:
        raise ValueError("Vetor otimizado nao tem forca para: " + ", ".join(missing))

    result["team_key"] = team_keys
    result["forca_resultante_01"] = team_keys.map(vector).astype(float)
    result["forca_elastica"] = result["forca_resultante_01"]
    result["forca_com_offset"] = result["forca_resultante_01"]

    sort_columns = [
        column
        for column in ["forca_resultante_01", "fifa_force_01", "elo_force_01", "market_force_01"]
        if column in result.columns
    ]
    result = result.sort_values(by=sort_columns, ascending=False).reset_index(drop=True)
    if "market_prob" in result.columns:
        result["ranking_odds"] = (
            result["market_prob"].rank(method="min", ascending=False).fillna(len(result) + 1).astype(int)
        )
    result.index = result.index + 1
    result.insert(0, "ranking_forca", result.index)
    result["rank_forca"] = result["ranking_forca"]
    return result, 1.0


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


def render_param_sidebar() -> ModelParams:
    """Renderiza os pesos do indicador e os parâmetros do modelo na barra lateral.

    As chaves (``key=``) garantem que os valores fiquem sincronizados ao navegar
    entre as páginas de Indicador de Força, Partida e Simulação.
    """
    # Inicializa o dicionário de persistência no session_state caso não exista
    if "model_sidebar_params" not in st.session_state:
        st.session_state["model_sidebar_params"] = {
            "param_weight_fifa": DEFAULT_WEIGHT_FIFA,
            "param_weight_market": DEFAULT_WEIGHT_MARKET,
            "param_weight_elo": DEFAULT_WEIGHT_ELO,
            "param_weight_momentum": DEFAULT_WEIGHT_MOMENTUM,
            "param_weight_history": DEFAULT_WEIGHT_HISTORY,
            "param_weight_host": DEFAULT_WEIGHT_HOST,
            "param_media_gols": DEFAULT_MEDIA_GOLS,
            "param_media_gols_vetor_otimizado": 3.0,
            "param_offset": DEFAULT_OFFSET,
            "param_elasticidade": DEFAULT_ELASTICIDADE,
            "param_usar_dixon_coles": DEFAULT_USAR_DIXON_COLES,
            "param_rho_dixon_coles": DEFAULT_RHO_DIXON_COLES,
            "param_usar_vetor_forca_otimizado": False,
        }
    else:
        st.session_state["model_sidebar_params"].setdefault("param_usar_vetor_forca_otimizado", False)
        st.session_state["model_sidebar_params"].setdefault("param_media_gols_vetor_otimizado", 3.0)

    # Sincroniza do dicionário persistente para as chaves atuais do session_state
    for key, val in st.session_state["model_sidebar_params"].items():
        if key not in st.session_state:
            st.session_state[key] = val

    with st.sidebar:
        st.markdown("#### Vetor de Força")
        usar_vetor_otimizado = st.toggle(
            "Usar vetor de força otimizado",
            value=bool(st.session_state["model_sidebar_params"]["param_usar_vetor_forca_otimizado"]),
            key="param_usar_vetor_forca_otimizado",
        )

        optimized_defaults = {}
        if usar_vetor_otimizado:
            try:
                optimized_defaults = load_optimized_force_vector().get("parametros_simulacao", {})
            except Exception as error:  # noqa: BLE001
                st.error(f"Erro ao carregar o vetor de força otimizado: {error}")
                st.stop()

        if usar_vetor_otimizado:
            weight_fifa = float(st.session_state["model_sidebar_params"]["param_weight_fifa"])
            weight_elo = float(st.session_state["model_sidebar_params"]["param_weight_elo"])
            weight_momentum = float(st.session_state["model_sidebar_params"]["param_weight_momentum"])
            weight_market = float(st.session_state["model_sidebar_params"]["param_weight_market"])
            weight_history = float(st.session_state["model_sidebar_params"]["param_weight_history"])
            weight_host = float(st.session_state["model_sidebar_params"]["param_weight_host"])
            offset = 0.0
            elasticidade = 1.0
            media_gols = 3.0
            st.session_state["model_sidebar_params"]["param_media_gols_vetor_otimizado"] = media_gols
            st.session_state["param_media_gols_vetor_otimizado"] = media_gols
        else:
            st.markdown("#### Indicador de Força")
            
            # Primeira linha (3 colunas): Ranking ELO, Ranking FIFA, Mercado
            col_w1, col_w2, col_w3 = st.columns(3)
            with col_w1:
                weight_elo = st.slider("Ranking ELO", min_value=0.0, max_value=1.0, value=float(st.session_state["model_sidebar_params"]["param_weight_elo"]), step=0.01, key="param_weight_elo")
            with col_w2:
                weight_fifa = st.slider("Ranking FIFA", min_value=0.0, max_value=1.0, value=float(st.session_state["model_sidebar_params"]["param_weight_fifa"]), step=0.01, key="param_weight_fifa")
            with col_w3:
                weight_market = st.slider("Mercado", min_value=0.0, max_value=1.0, value=float(st.session_state["model_sidebar_params"]["param_weight_market"]), step=0.01, key="param_weight_market")

            # Segunda linha (3 colunas): Momento, Histórico, Anfitrião
            col_w4, col_w5, col_w6 = st.columns(3)
            with col_w4:
                weight_momentum = st.slider("Momento", min_value=0.0, max_value=1.0, value=float(st.session_state["model_sidebar_params"]["param_weight_momentum"]), step=0.01, key="param_weight_momentum")
            with col_w5:
                weight_history = st.slider("Histórico", min_value=0.0, max_value=1.0, value=float(st.session_state["model_sidebar_params"]["param_weight_history"]), step=0.01, key="param_weight_history")
            with col_w6:
                weight_host = st.slider("Anfitrião", min_value=0.0, max_value=1.0, value=float(st.session_state["model_sidebar_params"]["param_weight_host"]), step=0.01, key="param_weight_host")

            st.markdown("#### Parâmetros do Modelo")

            col7, col8, col9 = st.columns(3)
            with col7:
                media_gols = st.slider("Média de gols", min_value=0.5, max_value=5.0, value=float(st.session_state["model_sidebar_params"]["param_media_gols"]), step=0.05, key="param_media_gols")
            with col8:
                offset = st.slider("Offset", min_value=0.0, max_value=1.0, value=float(st.session_state["model_sidebar_params"]["param_offset"]), step=0.01, key="param_offset")
            with col9:
                elasticidade = st.slider("Elasticidade", min_value=0.1, max_value=5.0, value=float(st.session_state["model_sidebar_params"]["param_elasticidade"]), step=0.01, key="param_elasticidade")

        # Dixon-Coles e rho desativados da interface (mantidos internamente com usar_dixon_coles=True e rho=-0.13)
        # Descomente o bloco abaixo caso queira reativar os widgets na interface no futuro:
        # col10, col11 = st.columns([2, 3])
        # with col10:
        #     st.markdown("<div style='margin-top: 2rem;'></div>", unsafe_allow_html=True)
        #     usar_dixon_coles = st.toggle("Dixon-Coles", value=bool(st.session_state["model_sidebar_params"]["param_usar_dixon_coles"]), key="param_usar_dixon_coles")
        # with col11:
        #     rho_dixon_coles = st.slider("Parâmetro rho", min_value=-0.30, max_value=0.00, value=float(st.session_state["model_sidebar_params"]["param_rho_dixon_coles"]), step=0.01, disabled=not usar_dixon_coles, key="param_rho_dixon_coles")
        
        usar_dixon_coles = bool(optimized_defaults.get("dixon_coles", True)) if usar_vetor_otimizado else True
        rho_dixon_coles = float(optimized_defaults.get("rho", -0.13)) if usar_vetor_otimizado else -0.13

        st.markdown('<div class="reset-btn-container">', unsafe_allow_html=True)
        if not usar_vetor_otimizado and st.button("🔄 Resetar para Valores Iniciais", key="reset_model_params_btn", use_container_width=True):
            st.session_state["model_sidebar_params"] = {
                "param_weight_fifa": DEFAULT_WEIGHT_FIFA,
                "param_weight_market": DEFAULT_WEIGHT_MARKET,
                "param_weight_elo": DEFAULT_WEIGHT_ELO,
                "param_weight_momentum": DEFAULT_WEIGHT_MOMENTUM,
                "param_weight_history": DEFAULT_WEIGHT_HISTORY,
                "param_weight_host": DEFAULT_WEIGHT_HOST,
                "param_media_gols": DEFAULT_MEDIA_GOLS,
                "param_media_gols_vetor_otimizado": 3.0,
                "param_offset": DEFAULT_OFFSET,
                "param_elasticidade": DEFAULT_ELASTICIDADE,
                "param_usar_dixon_coles": DEFAULT_USAR_DIXON_COLES,
                "param_rho_dixon_coles": DEFAULT_RHO_DIXON_COLES,
                "param_usar_vetor_forca_otimizado": False,
            }
            # Remove chaves do session_state dos sliders para recarregarem os novos valores padrão
            for key in st.session_state["model_sidebar_params"].keys():
                if key in st.session_state:
                    del st.session_state[key]
            try:
                st.rerun()
            except AttributeError:
                st.experimental_rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    # Sincroniza de volta das chaves atuais do session_state para o dicionário persistente
    for key in st.session_state["model_sidebar_params"].keys():
        if key in st.session_state:
            st.session_state["model_sidebar_params"][key] = st.session_state[key]

    return ModelParams(
        weight_fifa=weight_fifa,
        weight_elo=weight_elo,
        weight_momentum=weight_momentum,
        weight_market=weight_market,
        weight_history=weight_history,
        weight_host=weight_host,
        media_gols=media_gols,
        offset=offset,
        elasticidade=elasticidade,
        usar_dixon_coles=usar_dixon_coles,
        rho_dixon_coles=rho_dixon_coles,
        usar_vetor_otimizado=usar_vetor_otimizado,
    )


def load_force_dataframe() -> pd.DataFrame:
    """Carrega a base enriquecida mais recente, interrompendo a página em caso de erro."""
    dataset_path = find_latest_enriched_dataset()
    try:
        return load_force_table(str(dataset_path))
    except Exception as error:  # noqa: BLE001
        st.error(f"Erro ao carregar a base enriquecida: {error}")
        st.stop()


def build_combined(base_df: pd.DataFrame, params: ModelParams) -> tuple[pd.DataFrame, float]:
    """Aplica os pesos/offset/elasticidade dos parâmetros e devolve (tabela, soma_pesos)."""
    if params.usar_vetor_otimizado:
        return build_optimized_force_table(base_df)

    return build_combined_table(
        dataframe=base_df,
        weight_fifa=params.weight_fifa,
        weight_elo=params.weight_elo,
        weight_momentum=params.weight_momentum,
        weight_market=params.weight_market,
        weight_history=params.weight_history,
        offset=params.offset,
        elasticidade=params.elasticidade,
        weight_host=params.weight_host,
    )
