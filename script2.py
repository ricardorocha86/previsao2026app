import pandas as pd
import numpy as np

# ============ CONFIGURAÇÕES ============
MEDIA_GOLS_COPA = 2.75
DIFF_FORCA = 500.0
FORCA_PADRAO = 1350.0

# Pesos para o indicador composto de força
PESO_RANKING_FIFA = 0.60      # 60% ranking FIFA
PESO_PARTICIPACOES = 0.15     # 15% participações em copas
PESO_MELHOR_RESULTADO = 0.25  # 25% melhor resultado histórico

# Mapeamento de melhor resultado para pontuação (0 a 1)
RESULTADO_COPA_PONTOS = {
    'Campeão': 1.0,
    '2º Lugar': 0.85,
    '3º Lugar': 0.75,
    '4º Lugar': 0.65,
    'Quartas de Final': 0.50,
    'Oitavas de Final': 0.35,
    'Fase de Grupos': 0.20,
    'Estreante': 0.10,
    'N/A': 0.10
}

# ============ 1. PREPARAÇÃO DOS DADOS ============
def extrair_melhor_resultado(texto):
    """Extrai o melhor resultado da string (ex: 'Campeão (1978, 1986, 2022)' -> 'Campeão')"""
    if pd.isna(texto) or texto == 'N/A':
        return 'N/A'
    # Pega apenas a parte antes do parêntese
    resultado = str(texto).split('(')[0].strip()
    return resultado if resultado in RESULTADO_COPA_PONTOS else 'N/A'

def calcular_forca_composta(row, max_fifa, max_participacoes):
    """Calcula o indicador composto de força da seleção"""
    
    # 1. Componente Ranking FIFA (normalizado)
    pontos_fifa = row['Pontuacao_FIFA_Num']
    fifa_normalizado = pontos_fifa / max_fifa if max_fifa > 0 else 0.5
    
    # 2. Componente Participações (normalizado com escala logarítmica para suavizar)
    participacoes = row['Participacoes_Num']
    if max_participacoes > 0:
        # Usa escala logarítmica para não penalizar demais estreantes
        participacoes_normalizado = np.log1p(participacoes) / np.log1p(max_participacoes)
    else:
        participacoes_normalizado = 0.1
    
    # 3. Componente Melhor Resultado
    melhor_resultado = row['Melhor_Resultado_Limpo']
    resultado_normalizado = RESULTADO_COPA_PONTOS.get(melhor_resultado, 0.10)
    
    # Combina os componentes com os pesos
    forca_normalizada = (
        PESO_RANKING_FIFA * fifa_normalizado +
        PESO_PARTICIPACOES * participacoes_normalizado +
        PESO_MELHOR_RESULTADO * resultado_normalizado
    )
    
    # Converte para escala similar ao ranking FIFA (aprox. 1200-1900)
    forca_final = 1200 + (forca_normalizada * 700)
    
    return forca_final

def preparar_dados(caminho_csv):
    df = pd.read_csv(caminho_csv, sep = ';', skiprows = 1)
    
    # Processa Pontuação FIFA
    df['Pontuacao_FIFA_Num'] = df['Pontuacao_Ranking_FIFA'].apply(
        lambda x: float(str(x).replace('.', '').replace(',', '.')) if pd.notna(x) and x != 'N/A' else FORCA_PADRAO
    )
    
    # Processa Participações em Copa
    df['Participacoes_Num'] = df['Participações_Copa_Mundo'].apply(
        lambda x: int(x) if pd.notna(x) and str(x) != 'N/A' else 0
    )
    
    # Processa Melhor Resultado
    df['Melhor_Resultado_Limpo'] = df['Melhor_Resultado_Copa_Mundo'].apply(extrair_melhor_resultado)
    
    # Calcula valores máximos para normalização
    max_fifa = df['Pontuacao_FIFA_Num'].max()
    max_participacoes = df['Participacoes_Num'].max()
    
    # Calcula a força composta
    df['Forca'] = df.apply(lambda row: calcular_forca_composta(row, max_fifa, max_participacoes), axis=1)
    df['Grupo'] = df['Grupo'].str.strip()
    
    forca_dict = dict(zip(df['Seleção'], df['Forca']))  # lookup O(1)
    grupos_dict = df.groupby('Grupo')['Seleção'].apply(list).to_dict()
    selecoes = df['Seleção'].tolist()
    
    # Mostra as seleções com maior força calculada
    print(f"\n[DADOS] {len(df)} seleções | {len(grupos_dict)} grupos")
    print(f"[PESOS] FIFA: {PESO_RANKING_FIFA*100:.0f}% | Participações: {PESO_PARTICIPACOES*100:.0f}% | Melhor Resultado: {PESO_MELHOR_RESULTADO*100:.0f}%")
    print("\n[TOP 10 FORÇA COMPOSTA]")
    top10 = df.nlargest(10, 'Forca')[['Seleção', 'Pontuacao_FIFA_Num', 'Participacoes_Num', 'Melhor_Resultado_Limpo', 'Forca']]
    print(top10.to_string(index=False))
    
    return selecoes, forca_dict, grupos_dict

# ============ 2. MOTOR DE JOGO ============
def simular_jogo(forca_a, forca_b, mata_mata = False):
    diff = (forca_a - forca_b) / DIFF_FORCA
    lambda_a = max(0.1, (MEDIA_GOLS_COPA / 2) * (1 + diff))
    lambda_b = max(0.1, (MEDIA_GOLS_COPA / 2) * (1 - diff))
    
    gols_a, gols_b = np.random.poisson(lambda_a), np.random.poisson(lambda_b)
    fp_a, fp_b = -np.random.poisson(1.5), -np.random.poisson(1.5)
    
    if gols_a > gols_b: return 3, 0, gols_a, gols_b, fp_a, fp_b, 0  # A venceu
    if gols_b > gols_a: return 0, 3, gols_a, gols_b, fp_a, fp_b, 1  # B venceu
    if not mata_mata:   return 1, 1, gols_a, gols_b, fp_a, fp_b, -1 # empate
    return 0, 0, gols_a, gols_b, fp_a, fp_b, 0 if np.random.random() < (0.5 + diff * 0.1) else 1

# ============ 3. FASE DE GRUPOS ============
def simular_fase_grupos(grupos_dict, forca_dict):
    resultados = []
    
    for grupo, times in grupos_dict.items():
        stats = {t: [0, 0, 0, 0] for t in times}  # [Pts, SG, GP, FP]
        
        for i in range(len(times)):
            for j in range(i + 1, len(times)):
                t1, t2 = times[i], times[j]
                p1, p2, ga, gb, fp1, fp2, _ = simular_jogo(forca_dict[t1], forca_dict[t2])
                
                stats[t1][0] += p1; stats[t1][1] += ga - gb; stats[t1][2] += ga; stats[t1][3] += fp1
                stats[t2][0] += p2; stats[t2][1] += gb - ga; stats[t2][2] += gb; stats[t2][3] += fp2
        
        ranking = sorted(stats.items(), key = lambda x: (x[1][0], x[1][1], x[1][2], x[1][3]), reverse = True)
        for pos, (time, stat) in enumerate(ranking):
            resultados.append((time, grupo, pos + 1, stat[0], stat[1], stat[2], stat[3]))
    
    return resultados

# ============ 4. CLASSIFICAÇÃO PARA MATA-MATA ============
def definir_classificados_32(resultados_grupos, forca_dict):
    primeiros = [(t, s[0], s[1], s[2], s[3]) for t, g, p, *s in resultados_grupos if p == 1]
    segundos = [(t, s[0], s[1], s[2], s[3]) for t, g, p, *s in resultados_grupos if p == 2]
    terceiros = sorted(
        [(t, s[0], s[1], s[2], s[3]) for t, g, p, *s in resultados_grupos if p == 3],
        key = lambda x: (x[1], x[2], x[3], x[4]), reverse = True
    )[:8]
    
    todos = primeiros + segundos + terceiros
    return sorted(todos, key = lambda x: (x[1], x[2], x[3], forca_dict[x[0]]), reverse = True)

# ============ 5. MATA-MATA ============
def rodar_mata_mata(classificados, forca_dict, historico, fase):
    if len(classificados) < 2: return classificados
    
    vencedores = []
    n = len(classificados)
    
    for i in range(n // 2):
        t1, t2 = classificados[i][0], classificados[n - 1 - i][0]
        *_, resultado = simular_jogo(forca_dict[t1], forca_dict[t2], mata_mata = True)
        
        ganhador = t1 if resultado == 0 else t2
        dados = classificados[i] if resultado == 0 else classificados[n - 1 - i]
        vencedores.append(dados)
        historico[ganhador][fase] = 1
    
    return rodar_mata_mata(vencedores, forca_dict, historico, fase + 1)

# ============ 6. SIMULAR UMA COPA ============
def simular_uma_copa(selecoes, forca_dict, grupos_dict):
    historico = {t: [0] * 7 for t in selecoes}
    
    resultados = simular_fase_grupos(grupos_dict, forca_dict)
    for t, *_ in resultados: historico[t][0] = 1
    
    classificados = definir_classificados_32(resultados, forca_dict)
    for t, *_ in classificados: historico[t][1] = 1
    
    rodar_mata_mata(classificados, forca_dict, historico, 2)
    return historico

# ============ 7. MONTE CARLO ============
def gerar_analise_completa(caminho_csv, n_simulacoes = 1000):
    selecoes, forca_dict, grupos_dict = preparar_dados(caminho_csv)
    
    idx = {t: i for i, t in enumerate(selecoes)}  # mapeamento rápido
    acumulador = np.zeros((len(selecoes), 7), dtype = np.int32)
    
    print(f"\n[MONTE CARLO] {n_simulacoes} simulações...")
    
    for i in range(n_simulacoes):
        if (i + 1) % 500 == 0: print(f"[PROGRESSO] {i + 1}/{n_simulacoes}")
        resultado = simular_uma_copa(selecoes, forca_dict, grupos_dict)
        for time, stats in resultado.items():
            acumulador[idx[time]] += stats
    
    colunas = ['Fase Grupos', 'Top 32', 'Oitavas', 'Quartas', 'Semis', 'Final', 'Campeão']
    df = pd.DataFrame(acumulador / n_simulacoes, index = selecoes, columns = colunas)
    df = df.sort_values('Campeão', ascending = False)
    
    print(f"\n[CONCLUÍDO] {n_simulacoes} simulações!")
    print(df.head(5)[['Campeão', 'Final', 'Semis']].to_string())
    return df

# ============ EXECUÇÃO ============
resultado = gerar_analise_completa('dados.csv', n_simulacoes = 10000)
print(resultado)

