from __future__ import annotations

import os
import re
import sys
from io import BytesIO
import json
from datetime import datetime as _dt

import pandas as pd
import streamlit as st
from openpyxl.styles import Font, Alignment, PatternFill

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.helpers import inject_custom_css
from utils.simulador_oficial import simulate_one_cup_oficial, PoissonMatchSimulator
from utils.simulador_analitico import run_detailed_simulation
from utils.forca_core import (
    BASE_DIR,
    DATA_DIR,
    build_combined,
    compute_match_probabilities,
    load_force_dataframe,
    render_param_sidebar,
)


SIM_STAGE_COLUMNS = ["pos_1", "pos_2", "pos_3", "pos_4", "Top32", "Oitavas", "Quartas", "Semifinal", "Final", "Campeao"]


def build_match_cache(
    dataframe: pd.DataFrame,
    media_gols: float,
    usar_dixon_coles: bool,
    rho_dixon_coles: float,
) -> dict[tuple[str, str], dict[str, float]]:
    cache: dict[tuple[str, str], dict[str, float]] = {}
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
                media_gols=media_gols,
                max_goals=10,
                usar_dixon_coles=usar_dixon_coles,
                rho_dixon_coles=rho_dixon_coles,
            )
    return cache


def build_simulation_table(
    dataframe: pd.DataFrame,
    accumulated: dict[str, dict[str, int]],
    n_sims: int,
) -> pd.DataFrame:
    result = dataframe[
        [
            "Seleção",
            "team_key",
            "forca_com_offset",
        ]
    ].copy()

    for stage in SIM_STAGE_COLUMNS:
        result[f"{stage}_pct"] = result["team_key"].map(lambda key: accumulated[key][stage] / n_sims)

    # Ordenar pelos favoritos (Campeão, Final, etc.)
    result = result.sort_values(
        by=["Campeao_pct", "Final_pct", "Semifinal_pct", "forca_com_offset"],
        ascending=False,
    ).reset_index(drop=True)

    result.index = result.index + 1
    result.insert(0, "Rank Sim", result.index)

    return result


# Fases eliminatórias usadas nas tabelas agregadas por categoria e nº de vagas em cada fase.
# Somando a probabilidade por seleção e dividindo pelas vagas, cada coluna vira uma
# "participação esperada" da categoria na fase (todas as colunas somam 100% entre categorias).
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


def _aggregate_category(
    merged: pd.DataFrame,
    group_col: str,
    sort_alpha: bool = False,
    order_key=None,
) -> pd.DataFrame:
    """Agrega as probabilidades por fase somando entre as seleções de cada categoria.

    Cada coluna de fase é dividida pelo nº de vagas da fase, virando a fração esperada
    daquela fase ocupada pela categoria (a coluna ``Campeão`` é literalmente a
    probabilidade de o campeão sair da categoria).
    """
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
    if order_key is not None:
        result["_ord"] = result["Categoria"].map(order_key)
        result = result.sort_values(by="_ord").drop(columns="_ord")
    elif sort_alpha:
        result = result.sort_values(by="Categoria")
    else:
        result = result.sort_values(by="Campeão", ascending=False)
    return result.reset_index(drop=True)


def _contar_titulos_mundiais(melhor: object) -> int:
    """Conta quantos títulos mundiais a seleção tem a partir de 'Campeão (anos...)'."""
    texto = str(melhor or "").strip()
    if not texto.startswith("Campe"):
        return 0
    match = re.search(r"\(([^)]*)\)", texto)
    if not match:
        return 1
    anos = [ano for ano in re.split(r"[,/;]", match.group(1)) if ano.strip()]
    return len(anos) if anos else 1


def _ordem_titulos(categoria: str) -> tuple[int, int]:
    """Ordem complementar: estreantes, nunca campeãs e depois 1, 2, 3... títulos."""
    if categoria.startswith("Estre"):
        return (0, 0)
    if categoria.startswith("Nunca"):
        return (1, 0)
    match = re.match(r"(\d+)", categoria)
    return (2, int(match.group(1)) if match else 99)


# Colunas da distribuição de eliminação, na ordem das fases (somam 100% por seleção).
ELIMINATION_COLUMNS = [
    "Fase de Grupos",
    "16-avos",
    "Oitavas",
    "Quartas",
    "Semifinal",
    "Vice (Final)",
    "Campeã",
]


def build_elimination_table(sim_table: pd.DataFrame) -> pd.DataFrame:
    """Distribuição da fase em que cada seleção é eliminada.

    Derivada das probabilidades acumuladas ("chegou pelo menos à fase X"): a chance de
    sair exatamente numa fase é a diferença entre chegar nela e chegar na seguinte.
    Cada linha soma ~100%.
    """
    df = sim_table
    out = pd.DataFrame({"Seleção": df["Seleção"].values})
    out["Fase de Grupos"] = 1.0 - df["Top32_pct"].values
    out["16-avos"] = df["Top32_pct"].values - df["Oitavas_pct"].values
    out["Oitavas"] = df["Oitavas_pct"].values - df["Quartas_pct"].values
    out["Quartas"] = df["Quartas_pct"].values - df["Semifinal_pct"].values
    out["Semifinal"] = df["Semifinal_pct"].values - df["Final_pct"].values
    out["Vice (Final)"] = df["Final_pct"].values - df["Campeao_pct"].values
    out["Campeã"] = df["Campeao_pct"].values

    # Arredondamentos de ponto flutuante podem gerar diferenças levemente negativas.
    for col in ELIMINATION_COLUMNS:
        out[col] = out[col].clip(lower=0.0)

    out = out.sort_values(
        by=["Campeã", "Vice (Final)", "Semifinal", "Quartas"], ascending=False
    ).reset_index(drop=True)
    out.index = out.index + 1
    out.insert(0, "Rank", out.index)
    return out


GROUP_STAGE_PROB_COLUMNS = [
    "1º",
    "2º",
    "3º",
    "4º",
    "Avança como 3º",
    "Classifica (Top 32)",
    "Cai na fase de grupos",
]


def build_group_stage_table(
    sim_table: pd.DataFrame, meta_df: pd.DataFrame
) -> pd.DataFrame:
    """Detalhamento da fase de grupos por seleção, organizado por grupo.

    Tudo derivado das probabilidades já acumuladas: ``Avança como 3º`` é a parcela do
    Top 32 que sobra depois de 1º e 2º (os 8 melhores terceiros), e ``Cai na fase de
    grupos`` é o complemento do Top 32.
    """
    merged = sim_table.merge(meta_df[["team_key", "Grupo"]], on="team_key", how="left")

    out = pd.DataFrame({"Grupo": merged["Grupo"].values, "Seleção": merged["Seleção"].values})
    out["1º"] = merged["pos_1_pct"].values
    out["2º"] = merged["pos_2_pct"].values
    out["3º"] = merged["pos_3_pct"].values
    out["4º"] = merged["pos_4_pct"].values
    out["Avança como 3º"] = (
        merged["Top32_pct"].values - merged["pos_1_pct"].values - merged["pos_2_pct"].values
    )
    out["Classifica (Top 32)"] = merged["Top32_pct"].values
    out["Cai na fase de grupos"] = 1.0 - merged["Top32_pct"].values

    for col in GROUP_STAGE_PROB_COLUMNS:
        out[col] = out[col].clip(lower=0.0)

    out = out.sort_values(
        by=["Grupo", "Classifica (Top 32)"], ascending=[True, False]
    ).reset_index(drop=True)
    return out


def build_category_tables(
    sim_table: pd.DataFrame, meta_df: pd.DataFrame
) -> dict[str, pd.DataFrame]:
    """Constrói as tabelas de probabilidade agregadas por grupo, confederação,
    estreia em Copas e tradição de título a partir da tabela de simulação."""
    meta_cols = [
        "team_key",
        "Grupo",
        "Confederação",
        "Participações_Copa_Mundo",
        "Melhor_Resultado_Copa_Mundo",
    ]
    merged = sim_table.merge(meta_df[meta_cols], on="team_key", how="left")

    merged["cat_grupo"] = "Grupo " + merged["Grupo"].astype(str)
    merged["cat_confed"] = merged["Confederação"].fillna("Sem confederação")

    apps = pd.to_numeric(merged["Participações_Copa_Mundo"], errors="coerce").fillna(-1)
    titulos = merged["Melhor_Resultado_Copa_Mundo"].apply(_contar_titulos_mundiais)

    def _cat_titulos(n_apps: float, n_titulos: int) -> str:
        if n_apps == 0:
            return "Estreantes (1ª Copa)"
        if n_titulos == 0:
            return "Nunca campeãs"
        return f"{n_titulos} título" + ("s" if n_titulos > 1 else "")

    merged["cat_titulos"] = [_cat_titulos(a, t) for a, t in zip(apps, titulos)]

    return {
        "Por grupo": _aggregate_category(merged, "cat_grupo", sort_alpha=True),
        "Por confederação": _aggregate_category(merged, "cat_confed"),
        "Por títulos em Copas": _aggregate_category(merged, "cat_titulos", order_key=_ordem_titulos),
    }


_MESES_PT = {
    "janeiro": 1, "fevereiro": 2, "março": 3, "abril": 4,
    "maio": 5, "junho": 6, "julho": 7, "agosto": 8,
    "setembro": 9, "outubro": 10, "novembro": 11, "dezembro": 12,
}


def _parse_date_pt(raw: str) -> str:
    """'Quinta-feira, 11 de junho de 2026' -> '11/06/2026' (sortable in Excel)."""
    m = re.search(r"(\d{1,2})\s+de\s+(\w+)\s+de\s+(\d{4})", raw)
    if m:
        day = int(m.group(1))
        month = _MESES_PT.get(m.group(2).lower(), 0)
        year = int(m.group(3))
        if month:
            return _dt(year, month, day).strftime("%d/%m/%Y")
    return raw


def _extract_local(raw: str) -> str:
    """'Cidade do México, no México — 13h00 no horário local' -> 'Cidade do México, México'."""
    # Remove tudo a partir de ' —' ou ' –' (travessão)
    local = re.split(r"\s*[—–]\s*", raw)[0].strip()
    # Limpar ', no/nos/na/nas PAIS' -> ', PAIS'
    local = re.sub(r",\s*n[oa]s?\s+", ", ", local)
    return local


def _extract_br_time(raw: str) -> str:
    """'(16h00 em Brasília / 18h00 em Praia / ...)' -> '16h00'.
    Also handles '(1h00 de 14 de junho em Brasília ...) -> '01h00'."""
    m = re.search(r"\((\d{1,2}h\d{2})", raw)
    if m:
        t = m.group(1)
        # Normalizar hora de 1 dígito -> 2 dígitos (1h00 -> 01h00)
        parts = t.split("h")
        return f"{int(parts[0]):02d}h{parts[1]}"
    return raw


def generate_group_predictions(
    dataframe: pd.DataFrame, media_gols: float, usar_dixon_coles: bool, rho_dixon_coles: float
) -> pd.DataFrame:
    try:
        with open(DATA_DIR / "calendario_copa_2026.json", "r", encoding="utf-8") as f:
            schedule = json.load(f)
    except FileNotFoundError:
        return pd.DataFrame()

    name_map = {
        "República da Coreia": "Coreia do Sul",
        "República Democrática do Congo": "RD do Congo",
        "República Tcheca": "Tcheca"
    }

    rows = []
    for match in schedule:
        team_a_str = name_map.get(match["team_a"], match["team_a"])
        team_b_str = name_map.get(match["team_b"], match["team_b"])

        # Obter forca se times existirem
        if team_a_str in dataframe["Seleção"].values and team_b_str in dataframe["Seleção"].values:
            forca_a = float(dataframe.loc[dataframe["Seleção"] == team_a_str, "forca_com_offset"].iloc[0])
            forca_b = float(dataframe.loc[dataframe["Seleção"] == team_b_str, "forca_com_offset"].iloc[0])

            # A função compute_match_probabilities já usa Poisson, média e rho do slider
            probs = compute_match_probabilities(
                force_a=forca_a, force_b=forca_b, media_gols=media_gols,
                usar_dixon_coles=usar_dixon_coles, rho_dixon_coles=rho_dixon_coles
            )

            wa, d, wb = probs["win_a"], probs["draw"], probs["win_b"]

            rows.append({
                "Grupo": f"Grupo {match['group']}",
                "Data": _parse_date_pt(match["date"]),
                "Local": _extract_local(match["location_time"]),
                "Horário Brasília": _extract_br_time(match["br_time"]),
                "Seleção A": team_a_str,
                "Vitória A": wa,
                "Empate": d,
                "Vitória B": wb,
                "Seleção B": team_b_str,
                "Força A": forca_a
            })

    df_res = pd.DataFrame(rows)
    if not df_res.empty:
        df_res = df_res.sort_values(by=["Grupo", "Força A"], ascending=[True, False]).reset_index(drop=True)
        df_res = df_res.drop(columns=["Força A"])
    return df_res


def simulation_excel_bytes(
    simulation_df: pd.DataFrame,
    info_df: pd.DataFrame,
    matches_df: pd.DataFrame = None,
    extra_sheets: dict[str, pd.DataFrame] | None = None,
) -> bytes:
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        simulation_df.to_excel(writer, sheet_name="Simulações", index=False)
        info_df.to_excel(writer, sheet_name="Parâmetros", index=False)

        if matches_df is not None and not matches_df.empty:
            # Formatar percentuais para exibicao sem converter pra string pra não estragar
            # Mas pandas to_excel precisa ou da string ou que apliquemos style.
            export_matches = matches_df.copy()
            for col in ["Vitória A", "Empate", "Vitória B"]:
                export_matches[col] = export_matches[col].apply(lambda x: f"{x:.1%}")

            export_matches.to_excel(writer, sheet_name="Previsão Jogos", index=False)
            workbook = writer.book
            worksheet = writer.sheets["Previsão Jogos"]

            header_fill = PatternFill(start_color="1A1A2E", end_color="1A1A2E", fill_type="solid")
            header_font = Font(color="FFFFFF", bold=True)
            center_align = Alignment(horizontal="center", vertical="center")

            # Novo esquema de cores
            base_fill = PatternFill(start_color="EBF4FA", end_color="EBF4FA", fill_type="solid") # Azul bem clarinho para toda a planilha
            win_fill = PatternFill(start_color="CCE5FF", end_color="CCE5FF", fill_type="solid") # Azul um pouco mais forte para vitórias
            draw_fill = PatternFill(start_color="E8E8E8", end_color="E8E8E8", fill_type="solid") # Cinza claro para empate

            for col in range(1, 10):
                cell = worksheet.cell(row=1, column=col)
                cell.fill = header_fill
                cell.font = header_font
                cell.alignment = center_align

            worksheet.column_dimensions['A'].width = 10   # Grupo
            worksheet.column_dimensions['B'].width = 14   # Data (dd/mm/yyyy)
            worksheet.column_dimensions['C'].width = 35   # Local (cidade, país)
            worksheet.column_dimensions['D'].width = 16   # Horário Brasília (HHhMM)
            worksheet.column_dimensions['E'].width = 20   # Seleção A
            worksheet.column_dimensions['F'].width = 12   # Vitória A
            worksheet.column_dimensions['G'].width = 12   # Empate
            worksheet.column_dimensions['H'].width = 12   # Vitória B
            worksheet.column_dimensions['I'].width = 20   # Seleção B

            for row in range(2, len(export_matches) + 2):
                for col in range(1, 10):
                    # Preenchimento base (azul claro)
                    worksheet.cell(row=row, column=col).fill = base_fill

                    if col in [1, 2, 3, 4, 6, 7, 8]:
                        worksheet.cell(row=row, column=col).alignment = center_align

                worksheet.cell(row=row, column=5).alignment = Alignment(horizontal="right", vertical="center")
                worksheet.cell(row=row, column=5).font = Font(bold=True)
                worksheet.cell(row=row, column=9).alignment = Alignment(horizontal="left", vertical="center")
                worksheet.cell(row=row, column=9).font = Font(bold=True)

                worksheet.cell(row=row, column=6).fill = win_fill
                worksheet.cell(row=row, column=7).fill = draw_fill
                worksheet.cell(row=row, column=8).fill = win_fill

        if extra_sheets:
            for sheet_name, sheet_df in extra_sheets.items():
                safe_name = sheet_name[:31]
                sheet_df.to_excel(writer, sheet_name=safe_name, index=False)

    buffer.seek(0)
    return buffer.getvalue()


def run_complete_simulation_progressive(
    dataframe: pd.DataFrame,
    media_gols: float,
    n_sims: int,
    usar_dixon_coles: bool,
    rho_dixon_coles: float,
    tipo_chaveamento: str = "Sorteio Oficial",
    chunk_size: int = 10000,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    strengths = dict(zip(dataframe["team_key"], dataframe["forca_com_offset"]))
    match_cache = build_match_cache(
        dataframe=dataframe,
        media_gols=media_gols,
        usar_dixon_coles=usar_dixon_coles,
        rho_dixon_coles=rho_dixon_coles,
    )
    match_simulator = PoissonMatchSimulator(match_cache=match_cache, strengths=strengths)
    status_placeholder = st.empty()

    def update_progress(completed: int, total: int) -> None:
        status_placeholder.markdown(f"**Progresso detalhado:** {completed:,} / {total:,} copas")

    with st.spinner("Processando simulação completa...", show_time=True):
        detailed_result = run_detailed_simulation(
            dataframe=dataframe,
            n_sims=n_sims,
            match_simulator=match_simulator,
            strengths=strengths,
            tipo_chaveamento=tipo_chaveamento,
            chunk_size=chunk_size,
            progress_callback=update_progress,
        )

    status_placeholder.success(f"Simulação completa concluída: {n_sims:,} copas!")
    sim_table = build_simulation_table(
        dataframe=dataframe,
        accumulated=detailed_result["accumulated"],
        n_sims=n_sims,
    )
    return sim_table, detailed_result["tables"]


inject_custom_css()

st.markdown("## Simulação Copa do Mundo 2026")
st.markdown(
    """
<p style="font-size: 1rem; margin-bottom: 1.5rem;">
Simulação completa da Copa do Mundo de 2026 a partir do indicador de força e dos
parâmetros do modelo definidos na barra lateral. Ajuste o número de copas e o tipo
de chaveamento abaixo e rode a simulação.
</p>
""",
    unsafe_allow_html=True,
)

params = render_param_sidebar()
base_df = load_force_dataframe()
combined_df, weight_sum = build_combined(base_df, params)

if weight_sum <= 0:
    st.warning("A soma dos pesos está zerada. Ajuste ao menos um peso na barra lateral para construir a força resultante.")

media_gols = params.media_gols
usar_dixon_coles = params.usar_dixon_coles
rho_dixon_coles = params.rho_dixon_coles
tipo_simulacao = "Completa"

if "explorador_sim_excel_bytes" not in st.session_state:
    st.session_state["explorador_sim_excel_bytes"] = None

st.markdown("### Parâmetros da Simulação")
col_param_1, col_param_2 = st.columns([3, 2])
with col_param_1:
    n_sims = st.slider(
        "Nº de Copas", min_value=10000, max_value=1000000, value=10000, step=10000, key="sim_n_sims"
    )
with col_param_2:
    tipo_chaveamento = st.radio(
        "Chaveamento", ["Sorteio Oficial", "Sorteio Aleatório"], horizontal=True, key="sim_tipo_chaveamento"
    )

run_simulation = st.button("Rodar simulação", width='stretch')

st.markdown("---")
st.markdown("### Simulação")

if "explorador_sim_display" not in st.session_state:
    st.session_state["explorador_sim_display"] = None
if "explorador_detailed_tables" not in st.session_state:
    st.session_state["explorador_detailed_tables"] = None
if "explorador_category_tables" not in st.session_state:
    st.session_state["explorador_category_tables"] = None
if "explorador_elimination_table" not in st.session_state:
    st.session_state["explorador_elimination_table"] = None
if "explorador_group_stage_table" not in st.session_state:
    st.session_state["explorador_group_stage_table"] = None

if run_simulation:
    sim_table, detailed_tables = run_complete_simulation_progressive(
        dataframe=combined_df,
        media_gols=media_gols,
        n_sims=int(n_sims),
        usar_dixon_coles=usar_dixon_coles,
        rho_dixon_coles=rho_dixon_coles,
        tipo_chaveamento=tipo_chaveamento,
        chunk_size=10000,
    )

    sim_display = sim_table[
        [
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

    info_df = pd.DataFrame(
        [
            {"Parametro": "Etapa", "Valor": "Pré-Torneio"},
            {"Parametro": "Peso FIFA", "Valor": params.weight_fifa},
            {"Parametro": "Peso ELO", "Valor": params.weight_elo},
            {"Parametro": "Peso Momento", "Valor": params.weight_momentum},
            {"Parametro": "Peso Mercado", "Valor": params.weight_market},
            {"Parametro": "Peso Histórico Copas", "Valor": params.weight_history},
            {"Parametro": "Peso Anfitrião", "Valor": params.weight_host},
            {"Parametro": "Offset", "Valor": params.offset},
            {"Parametro": "Elasticidade", "Valor": params.elasticidade},
            {"Parametro": "Média de gols", "Valor": media_gols},
            {"Parametro": "Usar Dixon-Coles", "Valor": usar_dixon_coles},
            {"Parametro": "Rho Dixon-Coles", "Valor": rho_dixon_coles},
            {"Parametro": "Número de Copas", "Valor": int(n_sims)},
            {"Parametro": "Tipo de Simulação", "Valor": tipo_simulacao},
        ]
    )

    matches_df = generate_group_predictions(
        dataframe=combined_df, media_gols=media_gols,
        usar_dixon_coles=usar_dixon_coles, rho_dixon_coles=rho_dixon_coles
    )

    elimination_table = build_elimination_table(sim_table)
    group_stage_table = build_group_stage_table(sim_table, combined_df)
    category_tables = build_category_tables(sim_table, combined_df)

    extra_sheets = None
    if detailed_tables:
        extra_sheets = {
            "Finais": detailed_tables["finais"],
            "Brasil 1o Top32": detailed_tables["brasil_1o_grupo_top32"],
            "Brasil 2o Top32": detailed_tables["brasil_2o_grupo_top32"],
            "Brasil 3o Top32": detailed_tables["brasil_3o_grupo_top32"],
            "Brasil Adv 16avos": detailed_tables["brasil_adversarios_16avos"],
            "Brasil Adv Oitavas": detailed_tables["brasil_adversarios_oitavas"],
            "Brasil Adv Quartas": detailed_tables["brasil_adversarios_quartas"],
            "Brasil Adv Semi": detailed_tables["brasil_adversarios_semifinal"],
            "Brasil Adv Final": detailed_tables["brasil_adversarios_final"],
            "Eliminadores Brasil": detailed_tables["eliminadores_brasil"],
            "Carrascos Brasil": detailed_tables["eliminadores_brasil_agrupado"],
            "Titulo Cond Brasil": detailed_tables["titulo_condicional_brasil"],
            "Impacto Pos Grupo": detailed_tables["impacto_posicao_grupo"],
            "Bottom16 Surpresa": detailed_tables["bottom16_surpresa"],
            "Bottom16 Lista": detailed_tables["bottom16_lista"],
            "Semifinais": detailed_tables["semifinais"],
        }

    if extra_sheets is None:
        extra_sheets = {}
    extra_sheets["Fase de Grupos Detalhe"] = group_stage_table
    extra_sheets["Fase de Eliminacao"] = elimination_table
    extra_sheets["Cat Por Grupo"] = category_tables["Por grupo"]
    extra_sheets["Cat Por Confederacao"] = category_tables["Por confederação"]
    extra_sheets["Cat Por Titulos"] = category_tables["Por títulos em Copas"]

    st.session_state["explorador_sim_excel_bytes"] = simulation_excel_bytes(
        sim_display,
        info_df,
        matches_df,
        extra_sheets=extra_sheets,
    )

    if int(n_sims) >= 100000:
        import datetime
        os.makedirs(BASE_DIR / "resultados", exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%d-%m-%Y")
        filepath = BASE_DIR / "resultados" / f"simulacao_previsao_esportiva_{tipo_simulacao}_Pre-Torneio_{timestamp}.xlsx"
        with open(filepath, "wb") as f:
            f.write(st.session_state["explorador_sim_excel_bytes"])

    st.session_state["explorador_sim_display"] = sim_display
    st.session_state["explorador_detailed_tables"] = detailed_tables
    st.session_state["explorador_category_tables"] = category_tables
    st.session_state["explorador_elimination_table"] = elimination_table
    st.session_state["explorador_group_stage_table"] = group_stage_table
    st.rerun()

if st.session_state.get("explorador_sim_display") is not None:
    st.dataframe(
        st.session_state["explorador_sim_display"],
        width='stretch',
        height=520,
        column_config={
            "1º Grupo": st.column_config.NumberColumn(format="%.3f"),
            "2º Grupo": st.column_config.NumberColumn(format="%.3f"),
            "3º Grupo": st.column_config.NumberColumn(format="%.3f"),
            "4º Grupo": st.column_config.NumberColumn(format="%.3f"),
            "Top 32": st.column_config.NumberColumn(format="%.3f"),
            "Oitavas": st.column_config.NumberColumn(format="%.3f"),
            "Quartas": st.column_config.NumberColumn(format="%.3f"),
            "Semi": st.column_config.NumberColumn(format="%.3f"),
            "Final": st.column_config.NumberColumn(format="%.3f"),
            "Campeão": st.column_config.NumberColumn(format="%.3f"),
        },
    )

group_stage_table = st.session_state.get("explorador_group_stage_table")
if group_stage_table is not None:
    st.markdown("### Fase de grupos detalhada")
    st.markdown(
        """
<p style="font-size: 0.95rem; color: #555; margin-bottom: 1rem;">
Probabilidade de cada seleção terminar em <b>1º</b>, <b>2º</b>, <b>3º</b> ou <b>4º</b> do
grupo. <b>Avança como 3º</b> é a chance de se classificar entre os 8 melhores terceiros;
<b>Classifica (Top 32)</b> é a chance total de ir ao mata-mata; <b>Cai na fase de grupos</b>
é o complemento.
</p>
""",
        unsafe_allow_html=True,
    )
    st.dataframe(
        group_stage_table,
        width="stretch",
        height=560,
        column_config={
            col: st.column_config.NumberColumn(format="%.3f")
            for col in GROUP_STAGE_PROB_COLUMNS
        },
    )

elimination_table = st.session_state.get("explorador_elimination_table")
if elimination_table is not None:
    st.markdown("### Fase de eliminação por seleção")
    st.markdown(
        """
<p style="font-size: 0.95rem; color: #555; margin-bottom: 1rem;">
Em qual fase cada seleção é eliminada — da fase de grupos ao título. Cada linha soma
100%: <b>16-avos</b> = chegou ao mata-mata de 32 e perdeu; <b>Vice</b> = perdeu a final;
<b>Campeã</b> = venceu o torneio.
</p>
""",
        unsafe_allow_html=True,
    )
    st.dataframe(
        elimination_table,
        width="stretch",
        height=520,
        column_config={
            "Rank": st.column_config.NumberColumn(format="%d"),
            **{
                col: st.column_config.NumberColumn(format="%.3f")
                for col in ELIMINATION_COLUMNS
            },
        },
    )

detailed_tables = st.session_state.get("explorador_detailed_tables")
if detailed_tables:
    def show_table(df: pd.DataFrame, height: int = 360) -> None:
        probability_columns = [
            col
            for col in df.columns
            if col.startswith("Prob") or col.startswith("Titulo ") or col == "Prob titulo"
        ]
        st.dataframe(
            df,
            width='stretch',
            height=height,
            column_config={
                col: st.column_config.NumberColumn(format="%.3f")
                for col in probability_columns
            },
        )

    st.markdown("### Análises da simulação completa")
    st.markdown("#### Finais")
    st.markdown("##### Finais mais prováveis")
    show_table(detailed_tables["finais"], height=360)

    st.markdown("#### Brasil")
    st.markdown("##### Primeiro mata-mata do Brasil por posição no grupo")
    col_brasil_1, col_brasil_2, col_brasil_3 = st.columns(3)
    with col_brasil_1:
        st.markdown("###### Brasil 1º do grupo")
        show_table(detailed_tables["brasil_1o_grupo_top32"], height=320)
    with col_brasil_2:
        st.markdown("###### Brasil 2º do grupo")
        show_table(detailed_tables["brasil_2o_grupo_top32"], height=320)
    with col_brasil_3:
        st.markdown("###### Brasil 3º do grupo")
        show_table(detailed_tables["brasil_3o_grupo_top32"], height=320)

    st.markdown("##### Adversários mais prováveis do Brasil por fase alcançada")
    col_adv_1, col_adv_2, col_adv_3, col_adv_4, col_adv_5 = st.columns(5)
    with col_adv_1:
        st.markdown("###### Dado que o Brasil avançou para 16 avos")
        show_table(detailed_tables["brasil_adversarios_16avos"], height=300)
    with col_adv_2:
        st.markdown("###### Dado que o Brasil avançou para oitavas")
        show_table(detailed_tables["brasil_adversarios_oitavas"], height=300)
    with col_adv_3:
        st.markdown("###### Dado que o Brasil avançou para quartas")
        show_table(detailed_tables["brasil_adversarios_quartas"], height=300)
    with col_adv_4:
        st.markdown("###### Dado que o Brasil avançou para semi")
        show_table(detailed_tables["brasil_adversarios_semifinal"], height=300)
    with col_adv_5:
        st.markdown("###### Dado que o Brasil avançou para final")
        show_table(detailed_tables["brasil_adversarios_final"], height=300)

    st.markdown("##### Eliminações do Brasil")
    col_elim_1, col_elim_2 = st.columns([1, 2])
    with col_elim_1:
        st.markdown("###### Top carrascos")
        show_table(detailed_tables["eliminadores_brasil_agrupado"], height=320)
    with col_elim_2:
        st.markdown("###### Carrascos por fase")
        show_table(detailed_tables["eliminadores_brasil"], height=320)

    st.markdown("#### Condicionais")
    col_cond_1, col_cond_2 = st.columns([1, 2])
    with col_cond_1:
        st.markdown("##### Título do Brasil por condição")
        show_table(detailed_tables["titulo_condicional_brasil"], height=360)
    with col_cond_2:
        st.markdown("##### Chance de título pela posição de avanço no grupo")
        show_table(detailed_tables["impacto_posicao_grupo"], height=360)

    st.markdown("#### Bottom 16")
    col_bottom_1, col_bottom_2 = st.columns([2, 1])
    with col_bottom_1:
        st.markdown("##### Pelo menos uma bottom 16 por fase")
        show_table(detailed_tables["bottom16_surpresa"], height=260)
    with col_bottom_2:
        st.markdown("##### Lista pelo indicador atual")
        show_table(detailed_tables["bottom16_lista"], height=260)

category_tables = st.session_state.get("explorador_category_tables")
if category_tables:
    def show_category_table(df: pd.DataFrame, height: int = 320) -> None:
        prob_cols = ["Força Média"] + list(CATEGORY_STAGE_LABELS.values())
        st.dataframe(
            df,
            width="stretch",
            height=height,
            column_config={
                "Nº Seleções": st.column_config.NumberColumn(format="%d"),
                **{
                    col: st.column_config.NumberColumn(format="%.3f")
                    for col in prob_cols
                    if col in df.columns
                },
            },
        )

    st.markdown("### Probabilidades agregadas por categoria")
    st.markdown(
        """
<p style="font-size: 0.95rem; color: #555; margin-bottom: 1rem;">
Cada coluna de fase soma a probabilidade das seleções da categoria e divide pelo nº de
vagas da fase, virando a <b>participação esperada</b> da categoria naquela fase (todas as
colunas somam 100% entre as categorias). A coluna <b>Campeão</b> é a probabilidade de o
campeão sair da categoria.
</p>
""",
        unsafe_allow_html=True,
    )

    st.markdown("#### Por grupo")
    show_category_table(category_tables["Por grupo"], height=460)

    st.markdown("#### Por confederação")
    show_category_table(category_tables["Por confederação"], height=280)

    st.markdown("#### Por títulos mundiais")
    show_category_table(category_tables["Por títulos em Copas"], height=300)

st.markdown("---")
_sim_bytes = st.session_state["explorador_sim_excel_bytes"]
_col_dl, _ = st.columns([1, 3])
with _col_dl:
    st.download_button(
        label="Baixar Excel",
        data=_sim_bytes if _sim_bytes else "",
        file_name="simulacao_explorador_forca.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        width='stretch',
        disabled=_sim_bytes is None,
        help="Disponível após rodar uma simulação." if _sim_bytes is None else None,
    )
