"""
Página Dados — Visualização da tabela ELO_FIFA_DadosEnriquecidos
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.helpers import inject_custom_css
from utils.simulador_oficial import parse_world_cup_score

inject_custom_css()

# ============ CARREGAR A BASE ENRIQUECIDA ============
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "dataset"


@st.cache_data
def load_enriched_dataset() -> pd.DataFrame:
    """Carrega a versão mais recente do dataset enriquecido."""
    candidates = sorted(DATA_DIR.glob("FIFA_ELO_DadosSeleções_*.xlsx"))
    if candidates:
        path = candidates[-1]
    else:
        path = DATA_DIR / "FIFA_ELO_DadosSeleções_2026-04-15.xlsx"
    return pd.read_excel(path), path.name


df_raw, dataset_name = load_enriched_dataset()

# ============ HEADER ============
st.markdown("## 📋 Dados — Base Enriquecida")
st.caption(f"Fonte: `{dataset_name}` · {len(df_raw)} seleções · {len(df_raw.columns)} variáveis")

# ============ MÉTRICAS RESUMO ============
col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)

with col_m1:
    st.metric("🌍 Seleções", len(df_raw))
with col_m2:
    st.metric("🏆 Grupos", df_raw["Grupo"].nunique() if "Grupo" in df_raw.columns else "—")
with col_m3:
    if "Valor_Mercado_Milhoes_EUR" in df_raw.columns:
        total_market = pd.to_numeric(df_raw["Valor_Mercado_Milhoes_EUR"], errors="coerce").sum()
        st.metric("💰 Valor Total", f"€{total_market:,.0f}M")
    else:
        st.metric("💰 Valor Total", "—")
with col_m4:
    if "ELO_Rating" in df_raw.columns:
        avg_elo = pd.to_numeric(df_raw["ELO_Rating"], errors="coerce").mean()
        st.metric("📈 ELO Médio", f"{avg_elo:.0f}")
    else:
        st.metric("📈 ELO Médio", "—")
with col_m5:
    if "FIFA_Current_Points" in df_raw.columns:
        avg_fifa = pd.to_numeric(df_raw["FIFA_Current_Points"], errors="coerce").mean()
        st.metric("⭐ FIFA Pts Médio", f"{avg_fifa:.1f}")
    else:
        st.metric("⭐ FIFA Pts Médio", "—")

st.markdown("---")

# ============ FILTROS ============
col_filter_1, col_filter_2, col_filter_3 = st.columns(3)

with col_filter_1:
    conf_col = next(
        (c for c in ["Confederação", "Confederacao"] if c in df_raw.columns), None
    )
    if conf_col:
        all_confs = ["Todas"] + sorted(df_raw[conf_col].dropna().unique().tolist())
        filtro_conf = st.selectbox("Confederação", all_confs, key="dados_filtro_conf")
    else:
        filtro_conf = "Todas"

with col_filter_2:
    if "Grupo" in df_raw.columns:
        all_groups = ["Todos"] + sorted(df_raw["Grupo"].dropna().unique().tolist())
        filtro_grupo = st.selectbox("Grupo", all_groups, key="dados_filtro_grupo")
    else:
        filtro_grupo = "Todos"

with col_filter_3:
    if "Continente_Geo" in df_raw.columns:
        all_continents = ["Todos"] + sorted(df_raw["Continente_Geo"].dropna().unique().tolist())
        filtro_cont = st.selectbox("Continente", all_continents, key="dados_filtro_cont")
    else:
        filtro_cont = "Todos"

# Aplicar filtros
df = df_raw.copy()
if filtro_conf != "Todas" and conf_col:
    df = df[df[conf_col] == filtro_conf]
if filtro_grupo != "Todos" and "Grupo" in df.columns:
    df = df[df["Grupo"] == filtro_grupo]
if filtro_cont != "Todos" and "Continente_Geo" in df.columns:
    df = df[df["Continente_Geo"] == filtro_cont]


def minmax_scale(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    minimum = numeric.min()
    maximum = numeric.max()
    if pd.isna(minimum) or pd.isna(maximum) or maximum == minimum:
        return pd.Series(0.5, index=series.index, dtype=float)
    return ((numeric - minimum) / (maximum - minimum)).astype(float)


def first_existing_column(dataframe: pd.DataFrame, candidates: list[str]) -> str | None:
    return next((column for column in candidates if column in dataframe.columns), None)


def build_model_variables(dataframe: pd.DataFrame) -> pd.DataFrame:
    model_df = pd.DataFrame(index=dataframe.index)

    variable_specs = {
        "FIFA": first_existing_column(dataframe, ["FIFA_Current_Points"]),
        "ELO": first_existing_column(dataframe, ["ELO_Rating"]),
        "Momento": first_existing_column(dataframe, ["ELO_Chg_1A"]),
        "Mercado": first_existing_column(dataframe, ["Valor_Mercado_Milhoes_EUR"]),
    }

    for label, column in variable_specs.items():
        if column:
            model_df[label] = pd.to_numeric(dataframe[column], errors="coerce")

    apps_column = first_existing_column(
        dataframe,
        ["Participações_Copa_Mundo", "ParticipaÃ§Ãµes_Copa_Mundo", "Participacoes_Copa_Mundo"],
    )
    best_column = first_existing_column(
        dataframe,
        ["Melhor_Resultado_Copa_Mundo", "Melhor_Resultado"],
    )
    if apps_column and best_column:
        apps_score = minmax_scale(dataframe[apps_column])
        best_score = dataframe[best_column].map(parse_world_cup_score)
        model_df["Histórico Copas"] = 0.5 * apps_score + 0.5 * best_score
    elif apps_column:
        model_df["Histórico Copas"] = minmax_scale(dataframe[apps_column])
    elif best_column:
        model_df["Histórico Copas"] = dataframe[best_column].map(parse_world_cup_score)

    team_column = first_existing_column(dataframe, ["Seleção", "SeleÃ§Ã£o"])
    if team_column:
        hosts = {"Estados Unidos", "México", "MÃ©xico", "Canadá", "CanadÃ¡"}
        model_df["Anfitrião"] = dataframe[team_column].isin(hosts).astype(int)

    return model_df.dropna(axis=1, how="all")

# ============ SELEÇÃO DE COLUNAS ============
# Organizar colunas em categorias para facilitar a escolha
COLUMN_GROUPS = {
    "Identificação": ["Seleção", "NomeIngles", "Grupo", "Confederação", "Confederacao", "Continente_Geo", "Capital", "Regiao"],
    "FIFA": [c for c in df_raw.columns if c.startswith("FIFA_")],
    "ELO": [c for c in df_raw.columns if c.startswith("ELO_")],
    "Mercado & Demo": ["Valor_Mercado_Milhoes_EUR", "Media_Idade", "Tamanho_Elenco", "Populacao", "Area_km2"],
    "Copa do Mundo": ["Participações_Copa_Mundo", "Melhor_Resultado_Copa_Mundo", "Status_Qualificação", "Status_Qualificacao"],
}

# Colunas padrão para exibição
DEFAULT_DISPLAY = [
    "Seleção", "Grupo", "FIFA_Current_Rank", "FIFA_Current_Points",
    "ELO_Ranking", "ELO_Rating", "ELO_Chg_1A",
    "Valor_Mercado_Milhoes_EUR", "Participações_Copa_Mundo", "Melhor_Resultado_Copa_Mundo",
]
# Filtrar só as que existem
DEFAULT_DISPLAY = [c for c in DEFAULT_DISPLAY if c in df.columns]

with st.expander("⚙️ Selecionar colunas visíveis", expanded=False):
    available_cols = df.columns.tolist()
    # Remover colunas não úteis para exibição
    skip_cols = ["Link_Bandeira", "FIFA_Flag_URL", "Resumo_Wikipedia"]
    available_cols = [c for c in available_cols if c not in skip_cols]
    selected_cols = st.multiselect(
        "Colunas",
        options=available_cols,
        default=DEFAULT_DISPLAY,
        key="dados_colunas",
    )

if not selected_cols:
    selected_cols = DEFAULT_DISPLAY

# ============ TABELA PRINCIPAL ============
st.markdown("### 📊 Tabela Completa")

# Configuração inteligente de colunas
column_config = {}
for col in selected_cols:
    if col in df.columns:
        dtype = df[col].dtype
        if "float" in str(dtype):
            if "Points" in col or "Rating" in col or "Mercado" in col:
                column_config[col] = st.column_config.NumberColumn(format="%.1f")
            elif "Aproveitamento" in col or "Media" in col:
                column_config[col] = st.column_config.NumberColumn(format="%.2f")
            else:
                column_config[col] = st.column_config.NumberColumn(format="%.1f")

# Ordenar por FIFA Rank se disponível
display_df = df[selected_cols].copy()
if "FIFA_Current_Rank" in display_df.columns:
    display_df = display_df.sort_values("FIFA_Current_Rank").reset_index(drop=True)
elif "ELO_Ranking" in display_df.columns:
    display_df = display_df.sort_values("ELO_Ranking").reset_index(drop=True)

display_df.index = display_df.index + 1

st.dataframe(
    display_df,
    width='stretch',
    height=580,
    column_config=column_config,
)

# ============ VISUALIZAÇÕES ============
st.markdown("---")
st.markdown("### 📈 Visualizações")

viz_tab1, viz_tab2, viz_tab3 = st.tabs(
    [
        "Rankings",
        "Distribuição Variáveis",
        "Correlações",
    ]
)

with viz_tab1:
    col_v1, col_v2, col_v3, col_v4 = st.columns(4)

    with col_v1:
        if "ELO_Rating" in df.columns and "Seleção" in df.columns:
            df_elo_sorted = df.nlargest(48, "ELO_Rating")
            fig_elo = px.bar(
                df_elo_sorted,
                y="Seleção", x="ELO_Rating",
                orientation="h",
                color="ELO_Rating",
                color_continuous_scale=["#112015", "#209927", "#68E70F"],
                title="Top 48 — Rating ELO",
            )
            fig_elo.update_layout(
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font_color="#C9D1C9", title_font_color="#68E70F",

                yaxis={"categoryorder": "total ascending"},
                height=1150,
            )
            st.plotly_chart(fig_elo, width='stretch')

    with col_v2:
        if "FIFA_Current_Points" in df.columns and "Seleção" in df.columns:
            df_fifa_sorted = df.nlargest(48, "FIFA_Current_Points")
            fig_fifa = px.bar(
                df_fifa_sorted,
                y="Seleção", x="FIFA_Current_Points",
                orientation="h",
                color="FIFA_Current_Points",
                color_continuous_scale=["#112015", "#209927", "#68E70F"],
                title="Top 48 — FIFA Points",
            )
            fig_fifa.update_layout(
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font_color="#C9D1C9", title_font_color="#68E70F",

                yaxis={"categoryorder": "total ascending"},
                height=1150,
            )
            st.plotly_chart(fig_fifa, width='stretch')

    with col_v3:
        if "Valor_Mercado_Milhoes_EUR" in df.columns and "Seleção" in df.columns:
            df_mkt = df.nlargest(48, "Valor_Mercado_Milhoes_EUR")
            fig_mkt = px.bar(
                df_mkt,
                y="Seleção", x="Valor_Mercado_Milhoes_EUR",
                orientation="h",
                color="Valor_Mercado_Milhoes_EUR",
                color_continuous_scale=["#1d1b0f", "#7AB802", "#FFCF26"],
                title="Top 48 — Valor de Mercado (€M)",
            )
            fig_mkt.update_layout(
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font_color="#C9D1C9", title_font_color="#FFCF26",

                yaxis={"categoryorder": "total ascending"},
                height=1150,
            )
            st.plotly_chart(fig_mkt, width='stretch')

    with col_v4:
        if "ELO_Chg_1A" in df.columns and "Seleção" in df.columns:
            df_mom = df.copy()
            df_mom["ELO_Chg_1A_num"] = pd.to_numeric(df_mom["ELO_Chg_1A"], errors="coerce")
            df_mom = df_mom.dropna(subset=["ELO_Chg_1A_num"])
            df_momentum = (
                df_mom.assign(ELO_Chg_1A_abs=df_mom["ELO_Chg_1A_num"].abs())
                .nlargest(48, "ELO_Chg_1A_abs")
                .drop(columns=["ELO_Chg_1A_abs"])
            )
            df_momentum = df_momentum.sort_values("ELO_Chg_1A_num")

            colors = ["#BF1A1F" if x < 0 else "#209927" for x in df_momentum["ELO_Chg_1A_num"]]
            fig_mom = go.Figure(go.Bar(
                x=df_momentum["ELO_Chg_1A_num"],
                y=df_momentum["Seleção"],
                orientation="h",
                marker_color=colors,
            ))
            fig_mom.update_layout(
                title="Top 48 — Variação ELO (1 Ano)",
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font_color="#C9D1C9", title_font_color="#68E70F",

                yaxis={"categoryorder": "total ascending"},
                height=1150,
            )
            st.plotly_chart(fig_mom, width='stretch')

with viz_tab2:
    model_variables_df = build_model_variables(df)

    if model_variables_df.empty:
        st.info("Nenhuma das variáveis do modelo foi encontrada no dataset filtrado.")
    else:
        overview_cols = st.columns(min(3, len(model_variables_df.columns)))
        for index, column in enumerate(model_variables_df.columns):
            with overview_cols[index % len(overview_cols)]:
                valid_series = model_variables_df[column].dropna()
                if valid_series.empty:
                    continue
                fig_small = px.histogram(
                    model_variables_df,
                    x=column,
                    nbins=12,
                    color_discrete_sequence=["#209927"],
                    title=column,
                )
                fig_small.update_layout(
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font_color="#C9D1C9",
                    title_font_color="#68E70F",

                    height=260,
                    margin=dict(l=20, r=20, t=45, b=35),
                    showlegend=False,
                )
                fig_small.update_traces(
                    marker_line_color="rgba(241,241,241,0.55)",
                    marker_line_width=1.1,
                )
                st.plotly_chart(fig_small, width='stretch')

with viz_tab3:
    model_variables_df = build_model_variables(df)
    corr_df = model_variables_df.drop(columns=["Anfitrião"], errors="ignore").dropna(axis=1, how="all")

    if corr_df.shape[1] < 2:
        st.info("São necessárias pelo menos duas variáveis válidas para calcular correlações.")
    else:
        team_column = first_existing_column(df, ["Seleção", "SeleÃ§Ã£o"])
        corr_plot_df = corr_df.copy()
        if team_column:
            corr_plot_df["__team__"] = df.loc[corr_plot_df.index, team_column].astype(str)

        variable_columns = corr_df.columns.tolist()
        scatter_pairs = [
            (left, right)
            for left_index, left in enumerate(variable_columns)
            for right in variable_columns[left_index + 1 :]
        ]

        for index, (x_column, y_column) in enumerate(scatter_pairs):
            if index % 5 == 0:
                scatter_cols = st.columns(5)

            pair_columns = [x_column, y_column]
            if "__team__" in corr_plot_df.columns:
                pair_columns.append("__team__")
            pair_df = corr_plot_df[pair_columns].dropna()
            if pair_df.shape[0] < 2:
                continue

            correlation = pair_df[x_column].corr(pair_df[y_column])
            correlation_label = "n/a" if pd.isna(correlation) else f"{correlation:.2f}"

            with scatter_cols[index % len(scatter_cols)]:
                fig_scatter_pair = px.scatter(
                    pair_df,
                    x=x_column,
                    y=y_column,
                    color_discrete_sequence=["#68E70F"],
                    opacity=0.78,
                    title=f"{y_column} x {x_column}",
                    hover_name="__team__" if "__team__" in pair_df.columns else None,
                )
                fig_scatter_pair.update_traces(
                    marker=dict(
                        size=7,
                        line=dict(width=1, color="rgba(241,241,241,0.35)"),
                    )
                )
                if pair_df[x_column].nunique() > 1:
                    regression_coefficients = np.polyfit(pair_df[x_column], pair_df[y_column], 1)
                    regression_x = np.linspace(pair_df[x_column].min(), pair_df[x_column].max(), 100)
                    regression_y = (
                        regression_coefficients[0] * regression_x
                        + regression_coefficients[1]
                    )
                    fig_scatter_pair.add_trace(
                        go.Scatter(
                            x=regression_x,
                            y=regression_y,
                            mode="lines",
                            line=dict(color="#68E70F", width=1.2),
                            hoverinfo="skip",
                            showlegend=False,
                        )
                    )
                if "__team__" in pair_df.columns:
                    brasil_df = pair_df[pair_df["__team__"] == "Brasil"]
                    if not brasil_df.empty:
                        brasil_row = brasil_df.iloc[0]
                        fig_scatter_pair.add_trace(
                            go.Scatter(
                                x=[brasil_row[x_column]],
                                y=[brasil_row[y_column]],
                                mode="markers",
                                marker=dict(
                                    size=15,
                                    color="#FFCF26",
                                    symbol="star",
                                    line=dict(width=2, color="#F1F1F1"),
                                ),
                                hovertext=["Brasil"],
                                hoverinfo="text",
                                showlegend=False,
                            )
                        )
                        fig_scatter_pair.add_annotation(
                            x=brasil_row[x_column],
                            y=brasil_row[y_column],
                            text="🇧🇷 Brasil",
                            showarrow=True,
                            arrowcolor="#FFCF26",
                            arrowwidth=1.6,
                            ax=26,
                            ay=-28,
                            bgcolor="rgba(17,22,17,0.9)",
                            bordercolor="#FFCF26",
                            borderwidth=1,
                            font=dict(color="#F1F1F1", size=11),
                        )
                fig_scatter_pair.add_annotation(
                    x=0.04,
                    y=0.94,
                    xref="paper",
                    yref="paper",
                    text=f"r = {correlation_label}",
                    showarrow=False,
                    align="left",
                    bgcolor="rgba(17,22,17,0.86)",
                    bordercolor="rgba(104,231,15,0.45)",
                    borderwidth=1,
                    font=dict(color="#F1F1F1", size=13),
                )
                fig_scatter_pair.update_layout(
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font_color="#C9D1C9",
                    title_font_color="#68E70F",

                    height=300,
                    margin=dict(l=15, r=15, t=52, b=42),
                    showlegend=False,
                )
                st.plotly_chart(fig_scatter_pair, width='stretch')

# ============ DOWNLOAD ============
st.markdown("---")
st.download_button(
    label="⬇️ Baixar dados filtrados (CSV)",
    data=df[selected_cols].to_csv(index=False).encode("utf-8"),
    file_name="ELO_FIFA_DadosEnriquecidos.csv",
    mime="text/csv",
    key="download_enriched",
)
