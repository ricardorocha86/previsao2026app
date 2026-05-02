import numpy as np
from scipy.stats import poisson
from utils.config import MEDIA_GOLS_COPA

# ============================================================
# 1. FUNÇÕES MATEMÁTICAS E AUXILIARES (ex-simulation.py)
# ============================================================

def parse_world_cup_score(text: object) -> float:
    """Converte o texto do melhor resultado em uma pontuação de força (0.1 a 1.0)."""
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

def dixon_coles_correction(gols_casa, gols_fora, lambda_casa, lambda_fora, rho=-0.13):
    """Aplica a correção de Dixon-Coles para placares baixos (0-0, 1-1, etc)."""
    if gols_casa == 0 and gols_fora == 0:
        return 1 - lambda_casa * lambda_fora * rho
    elif gols_casa == 0 and gols_fora == 1:
        return 1 + lambda_casa * rho
    elif gols_casa == 1 and gols_fora == 0:
        return 1 + lambda_fora * rho
    elif gols_casa == 1 and gols_fora == 1:
        return 1 - rho
    return 1.0

def elo_to_forca(elo, k_scale=400):
    """Transforma o rating Elo em Força exponencial."""
    return 10 ** (elo / k_scale)

def randomizar_grupos(grupos_dict, elo_dict=None):
    """Randomiza os grupos mantendo os cabeças de chave (Top 12/Hosts)."""
    CABECAS_2026 = [
        "Espanha", "Argentina", "França", "Inglaterra", "Brasil", "Portugal", 
        "Holanda", "Bélgica", "Croácia", "Estados Unidos", "México", "Marrocos"
    ]
    todos_times = []
    for teams in grupos_dict.values():
        todos_times.extend(teams)
    
    times_set = set(todos_times)
    cabecas_base = [t for t in CABECAS_2026 if t in times_set]
    
    if len(cabecas_base) < 12 and elo_dict:
        restantes = [t for t in todos_times if t not in cabecas_base]
        times_top_elo = sorted(restantes, key=lambda t: elo_dict.get(t, 0), reverse=True)
        cabecas_base.extend(times_top_elo[:(12 - len(cabecas_base))])
    
    num_grupos = len(grupos_dict)
    cabecas_final = cabecas_base[:num_grupos]
    cabecas_set = set(cabecas_final)
    outros_times = [t for t in todos_times if t not in cabecas_set]
    
    np.random.shuffle(cabecas_final)
    np.random.shuffle(outros_times)
    
    nomes_grupos = sorted(grupos_dict.keys())
    novo_grupos = {}
    idx_outros = 0
    for i, g in enumerate(nomes_grupos):
        cabeca = cabecas_final[i]
        num_vagas = len(grupos_dict[g]) - 1
        novo_grupos[g] = [cabeca] + outros_times[idx_outros : idx_outros + num_vagas]
        idx_outros += num_vagas
    return novo_grupos

def simular_jogo_simples(elo_a, elo_b, mata_mata=False, media_gols=MEDIA_GOLS_COPA, k_scale=400, **kwargs):
    """
    Simulação rápida de um único jogo. 
    Retorna 7 valores para compatibilidade com o modo Ao Vivo:
    (pontos_a, pontos_b, gols_a, gols_b, fairplay_a, fairplay_b, resultado)
    """
    f_a = elo_to_forca(elo_a, k_scale)
    f_b = elo_to_forca(elo_b, k_scale)
    share_a = f_a / (f_a + f_b)
    
    lambda_a = media_gols * share_a
    lambda_b = media_gols * (1.0 - share_a)
    
    gols_a = np.random.poisson(lambda_a)
    gols_b = np.random.poisson(lambda_b)
    
    # Fair Play (usado como critério de desempate aleatório no app)
    fp_a, fp_b = np.random.random(), np.random.random()
    
    if gols_a > gols_b:
        return 3, 0, gols_a, gols_b, fp_a, fp_b, 0
    elif gols_b > gols_a:
        return 0, 3, gols_a, gols_b, fp_a, fp_b, 1
    else:
        if not mata_mata:
            return 1, 1, gols_a, gols_b, fp_a, fp_b, 2
        # No mata-mata, decide nos pênaltis via share de força
        vence_a = np.random.random() < share_a
        return (0, 0, gols_a, gols_b, fp_a, fp_b, 0 if vence_a else 1)

# ============================================================
# 2. MOTOR DE ALTA PERFORMANCE (ex-simulador_v4.py)
# ============================================================

class BaseMatchSimulator:
    def simulate_match(self, team_a: str, team_b: str, rng: np.random.Generator, mata_mata: bool = False):
        raise NotImplementedError

class PoissonMatchSimulator(BaseMatchSimulator):
    def __init__(self, match_cache: dict, strengths: dict = None):
        self.match_cache = match_cache
        self.strengths = strengths
        for data in self.match_cache.values():
            if "flat_p" not in data:
                data["flat_p"] = data["matrix"].ravel()
                data["matrix_size"] = data["matrix"].size
                data["shape_cols"] = data["matrix"].shape[1]
            if "cumsum_p" not in data:
                data["cumsum_p"] = np.cumsum(data["flat_p"])

    def simulate_match(self, team_a: str, team_b: str, rng: np.random.Generator, mata_mata: bool = False):
        match_data = self.match_cache[(team_a, team_b)]
        flat_index = np.searchsorted(match_data["cumsum_p"], rng.random())
        cols = match_data["shape_cols"]
        gols_a, gols_b = flat_index // cols, flat_index % cols
        
        if gols_a > gols_b: return int(gols_a), int(gols_b), 1
        if gols_b > gols_a: return int(gols_a), int(gols_b), 2
        if not mata_mata: return int(gols_a), int(gols_b), 0
            
        # Mata-Mata: Prorrogação (30% da média original)
        lambda_a_extra = float(match_data["lambda_a"]) * 0.3
        lambda_b_extra = float(match_data["lambda_b"]) * 0.3
        gols_a_total = int(gols_a + rng.poisson(lambda_a_extra))
        gols_b_total = int(gols_b + rng.poisson(lambda_b_extra))
        
        if gols_a_total > gols_b_total: return gols_a_total, gols_b_total, 1
        if gols_b_total > gols_a_total: return gols_a_total, gols_b_total, 2
            
        # Pênaltis baseados no share de força
        share_a = float(match_data["share_a"])
        return gols_a_total, gols_b_total, 1 if rng.random() < share_a else 2

def ordenar_grupo_oficial(tabela_grupo, strengths, rng):
    """Ordenação FIFA: Pontos -> Confronto Direto -> Saldo -> Gols -> Fair Play."""
    blocos = {}
    for t in tabela_grupo:
        t["fair_play"] = rng.random()
        pts = t["points"]
        if pts not in blocos: blocos[pts] = []
        blocos[pts].append(t)

    resultado = []
    for pts in sorted(blocos.keys(), reverse=True):
        empatados = blocos[pts]
        if len(empatados) == 1:
            resultado.extend(empatados)
            continue

        nomes_empatados = set(t["team_key"] for t in empatados)
        for t in empatados:
            pts_h2h = saldo_h2h = gols_h2h = 0
            for adv, jogo in t["confrontos"].items():
                if adv in nomes_empatados:
                    gp, gc = jogo
                    pts_h2h += 3 if gp > gc else (1 if gp == gc else 0)
                    saldo_h2h += gp - gc
                    gols_h2h += gp
            t.update({"pts_h2h": pts_h2h, "saldo_h2h": saldo_h2h, "gols_h2h": gols_h2h})

        resultado.extend(sorted(empatados, key=lambda t: (
            t["pts_h2h"], t["saldo_h2h"], t["gols_h2h"], 
            t["goal_diff"], t["goals_for"], t["fair_play"], strengths[t["team_key"]]
        ), reverse=True))
    return resultado

def simulate_one_cup_oficial(groups, strengths, rng, match_simulator: BaseMatchSimulator):
    """Simula uma Copa completa seguindo as regras de 2026 (48 times)."""
    SIM_STAGE_COLUMNS = ["pos_1", "pos_2", "pos_3", "pos_4", "Top32", "Oitavas", "Quartas", "Semifinal", "Final", "Campeao"]
    history = {tk: {s: 0 for s in SIM_STAGE_COLUMNS} for teams in groups.values() for tk in teams}

    # Fase de Grupos
    records = []
    for g_name, teams in sorted(groups.items()):
        table = {tk: {"team_key": tk, "group": g_name, "points": 0, "goal_diff": 0, "goals_for": 0, "fair_play": 0, "confrontos": {}} for tk in teams}
        for i, t_a in enumerate(teams):
            for t_b in teams[i+1:]:
                ga, gb, win = match_simulator.simulate_match(t_a, t_b, rng, False)
                table[t_a]["confrontos"][t_b], table[t_b]["confrontos"][t_a] = (ga, gb), (gb, ga)
                table[t_a]["goal_diff"] += ga - gb
                table[t_b]["goal_diff"] += gb - ga
                table[t_a]["goals_for"] += ga
                table[t_b]["goals_for"] += gb
                if win == 1: table[t_a]["points"] += 3
                elif win == 2: table[t_b]["points"] += 3
                else: table[t_a]["points"] += 1; table[t_b]["points"] += 1

        ranking = ordenar_grupo_oficial(list(table.values()), strengths, rng)
        for pos, rec in enumerate(ranking, start=1):
            rec["group_position"] = pos
            records.append(rec)
            history[rec["team_key"]][f"pos_{pos}"] = 1

    # Classificação Mata-Mata
    primeiros = {r["group"]: r for r in records if r["group_position"] == 1}
    segundos = {r["group"]: r for r in records if r["group_position"] == 2}
    terceiros = sorted([r for r in records if r["group_position"] == 3], key=lambda c: (c["points"], c["goal_diff"], c["goals_for"], c["fair_play"], strengths[c["team_key"]]), reverse=True)[:8]
    
    # Chaveamento Pools (Artigo 12.6)
    pools = {"E":["A","B","C","D","F"],"I":["C","D","F","G","H"],"A":["C","E","F","H","I"],"L":["E","H","I","J","K"],"D":["B","E","F","I","J"],"G":["A","E","H","I","J"],"B":["E","F","G","I","J"],"K":["D","E","I","J","L"]}
    confrontos_3 = {}
    t_disp = terceiros.copy()
    for g_p in ["E","I","A","L","D","G","B","K"]:
        for i, t in enumerate(t_disp):
            if t["group"] in pools[g_p]:
                confrontos_3[g_p] = t_disp.pop(i); break
        else:
            if t_disp: confrontos_3[g_p] = t_disp.pop(0)

    current_round = [segundos.get("A"), segundos.get("B"), primeiros.get("E"), confrontos_3.get("E"), primeiros.get("F"), segundos.get("C"), primeiros.get("C"), segundos.get("F"), primeiros.get("I"), confrontos_3.get("I"), segundos.get("E"), segundos.get("I"), primeiros.get("A"), confrontos_3.get("A"), primeiros.get("L"), confrontos_3.get("L"), primeiros.get("D"), confrontos_3.get("D"), primeiros.get("G"), confrontos_3.get("G"), segundos.get("K"), segundos.get("L"), primeiros.get("H"), segundos.get("J"), primeiros.get("B"), confrontos_3.get("B"), primeiros.get("J"), segundos.get("H"), primeiros.get("K"), confrontos_3.get("K"), segundos.get("D"), segundos.get("G")]
    current_round = [t for t in current_round if t is not None]

    for r in current_round: history[r["team_key"]]["Top32"] = 1
    
    stages = {32:"Oitavas", 16:"Quartas", 8:"Semifinal", 4:"Final", 2:"Campeao"}
    while len(current_round) > 1:
        next_s = stages[len(current_round)]
        next_r = []
        for i in range(0, len(current_round), 2):
            _, _, win = match_simulator.simulate_match(current_round[i]["team_key"], current_round[i+1]["team_key"], rng, True)
            w_rec = current_round[i] if win == 1 else current_round[i+1]
            history[w_rec["team_key"]][next_s] = 1
            next_r.append(w_rec)
        current_round = next_r
    return history
