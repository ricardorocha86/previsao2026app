import pandas as pd
import numpy as np
import random

# ============ CONFIGURAÇÕES ============
MEDIA_GOLS_COPA = 2.75  # média histórica de gols por jogo
DIFF_FORCA = 500.0     # diferença de pontos considerada significativa
FORCA_PADRAO = 1350.0  # força para seleções sem ranking

# ============ 1. PREPARAÇÃO DOS DADOS ============
def limpar_pontuacao(x):
    if pd.isna(x) or x == 'N/A': return np.nan
    if isinstance(x, str): return float(x.replace('.', '').replace(',', '.'))
    return float(x)

def preparar_dados(caminho_csv):
    df = pd.read_csv(caminho_csv, sep = ';', skiprows = 1)  # pula linha "sep=;"
    df['Forca'] = df['Pontuacao_Ranking_FIFA'].apply(limpar_pontuacao).fillna(FORCA_PADRAO)
    df['Grupo'] = df['Grupo'].str.strip()
    print(f"[DADOS] {len(df)} seleções carregadas | {df['Grupo'].nunique()} grupos")
    return df

# ============ 2. MOTOR DE JOGO (Poisson) ============
def simular_jogo(forca_a, forca_b, mata_mata = False):
    diff = (forca_a - forca_b) / DIFF_FORCA
    lambda_a = max(0.1, (MEDIA_GOLS_COPA / 2) * (1 + diff))
    lambda_b = max(0.1, (MEDIA_GOLS_COPA / 2) * (1 - diff))
    
    gols_a, gols_b = np.random.poisson(lambda_a), np.random.poisson(lambda_b)
    fp_a, fp_b = -np.random.poisson(1.5), -np.random.poisson(1.5)  # fair play
    
    if gols_a > gols_b: return gols_a, gols_b, 3, 0, fp_a, fp_b, 'A'
    if gols_b > gols_a: return gols_a, gols_b, 0, 3, fp_a, fp_b, 'B'
    if not mata_mata:   return gols_a, gols_b, 1, 1, fp_a, fp_b, 'E'  # empate grupos
    
    vencedor = 'A' if random.random() < (0.5 + diff * 0.1) else 'B'  # pênaltis
    return gols_a, gols_b, 0, 0, fp_a, fp_b, vencedor

# ============ 3. FASE DE GRUPOS ============
def simular_fase_grupos(df, historico):
    tabela_grupos = []
    
    for grupo in sorted(df['Grupo'].unique()):
        times = df[df['Grupo'] == grupo]['Seleção'].tolist()
        stats = {t: {'Pts': 0, 'SG': 0, 'GP': 0, 'FP': 0} for t in times}
        
        for i in range(len(times)):  # todos contra todos
            for j in range(i + 1, len(times)):
                t1, t2 = times[i], times[j]
                f1, f2 = df.loc[df['Seleção'] == t1, 'Forca'].values[0], df.loc[df['Seleção'] == t2, 'Forca'].values[0]
                ga, gb, p1, p2, fp1, fp2, _ = simular_jogo(f1, f2)
                
                stats[t1]['Pts'] += p1; stats[t1]['GP'] += ga; stats[t1]['SG'] += (ga - gb); stats[t1]['FP'] += fp1
                stats[t2]['Pts'] += p2; stats[t2]['GP'] += gb; stats[t2]['SG'] += (gb - ga); stats[t2]['FP'] += fp2
        
        df_grupo = pd.DataFrame.from_dict(stats, orient = 'index')
        df_grupo['Seleção'], df_grupo['Grupo'] = df_grupo.index, grupo
        df_grupo = df_grupo.sort_values(['Pts', 'SG', 'GP', 'FP'], ascending = False)
        df_grupo['Posicao'] = range(1, 5)
        tabela_grupos.append(df_grupo)
        
        for t in times: historico[t][0] = 1  # marca fase de grupos
    
    return pd.concat(tabela_grupos)

# ============ 4. CLASSIFICAÇÃO PARA MATA-MATA ============
def definir_classificados_32(df_grupos, df_original, historico):
    primeiros = df_grupos[df_grupos['Posicao'] == 1]
    segundos = df_grupos[df_grupos['Posicao'] == 2]
    terceiros = df_grupos[df_grupos['Posicao'] == 3].sort_values(['Pts', 'SG', 'GP', 'FP'], ascending = False).head(8)
    
    classificados = pd.concat([primeiros, segundos, terceiros])
    classificados = classificados.merge(df_original[['Seleção', 'Forca']], on = 'Seleção')
    classificados = classificados.sort_values(['Pts', 'SG', 'GP', 'Forca'], ascending = False)
    
    for t in classificados['Seleção']: historico[t][1] = 1  # marca top 32
    return classificados

# ============ 5. MATA-MATA (Recursivo) ============
def rodar_mata_mata(df_times, historico, indice_fase):
    if len(df_times) < 2: return df_times  # campeão
    
    nomes_fases = {2: '32avos', 3: 'Oitavas', 4: 'Quartas', 5: 'Semis', 6: 'Final'}
    vencedores = []
    
    for i in range(len(df_times) // 2):
        time_a, time_b = df_times.iloc[i], df_times.iloc[len(df_times) - 1 - i]
        _, _, _, _, _, _, resultado = simular_jogo(time_a['Forca'], time_b['Forca'], mata_mata = True)
        
        ganhador = time_a if resultado == 'A' else time_b
        vencedores.append(ganhador)
        historico[ganhador['Seleção']][indice_fase] = 1
    
    return rodar_mata_mata(pd.DataFrame(vencedores), historico, indice_fase + 1)

# ============ 6. SIMULAR UMA COPA COMPLETA ============
def simular_uma_copa(df):
    historico = {time: [0] * 7 for time in df['Seleção']}
    
    df_grupos = simular_fase_grupos(df, historico)
    classificados = definir_classificados_32(df_grupos, df, historico)
    rodar_mata_mata(classificados, historico, indice_fase = 2)
    
    return historico

# ============ 7. MONTE CARLO (Múltiplas Simulações) ============
def gerar_analise_completa(caminho_csv, n_simulacoes = 100):
    df = preparar_dados(caminho_csv)
    colunas = ['Fase Grupos', 'Top 32', 'Oitavas', 'Quartas', 'Semis', 'Final', 'Campeão']
    df_resultado = pd.DataFrame(0, index = df['Seleção'], columns = colunas)
    
    print(f"\n[MONTE CARLO] Iniciando {n_simulacoes} simulações da Copa 2026...")
    
    for i in range(n_simulacoes):
        if (i + 1) % 100 == 0 or i == 0: print(f"[PROGRESSO] Simulação {i + 1}/{n_simulacoes}")
        resultado = simular_uma_copa(df)
        for time, stats in resultado.items(): df_resultado.loc[time] += stats
    
    df_resultado = (df_resultado / n_simulacoes).sort_values('Campeão', ascending = False)
    
    print(f"\n[CONCLUÍDO] {n_simulacoes} simulações finalizadas!")
    print(f"[TOP 5 FAVORITOS]")
    print(df_resultado.head(5)[['Campeão', 'Final', 'Semis']].to_string())
    
    return df_resultado

# ============ EXECUÇÃO ============
resultado = gerar_analise_completa('dados.csv', n_simulacoes = 10000)
print(resultado)