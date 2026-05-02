import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time
from datetime import datetime
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.helpers import inject_custom_css, get_bandeira_url
from utils.data_loader import carregar_dados, carregar_dados_elo, preparar_estruturas
from utils.simulador_oficial import simular_jogo_simples as simular_jogo
from utils.config import MEDIA_GOLS_COPA

inject_custom_css()

st.markdown("## 🎬 Simulação Ao Vivo")
st.markdown("Acompanhe uma Copa do Mundo completa, vendo cada jogo acontecer em tempo real!")

# Carregar dados
try:
    df_dados = carregar_dados()
    df_elo = carregar_dados_elo()
    selecoes, elo_dict, grupos_dict, bandeiras_dict, stats_gols_dict = preparar_estruturas(df_dados, df_elo)
except Exception as e:
    st.error(f"❌ Erro ao carregar dados: {e}")
    st.stop()

# Controles
col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([1, 1, 1])

with col_ctrl1:
    velocidade = st.select_slider("⏱️ Velocidade", options=["Lento", "Normal", "Rápido", "Instantâneo"], value="Normal")
    delay_map = {"Lento": 1.0, "Normal": 0.3, "Rápido": 0.1, "Instantâneo": 0.0}
    delay = delay_map[velocidade]

with col_ctrl2:
    if st.button("🎲 NOVA COPA", width='stretch', type="primary"):
        st.session_state['live_fase'] = 'grupos'
        st.session_state['live_jogos'] = []
        st.session_state['live_stats'] = {g: {t: [0, 0, 0, 0, 0, 0] for t in times} for g, times in grupos_dict.items()}
        st.session_state['live_running'] = True
        st.session_state['live_classificados'] = []
        st.session_state['live_campeao'] = None
        st.session_state['live_vice'] = None
        st.session_state['live_semifinalistas'] = []
        st.session_state['live_final_placar'] = None

with col_ctrl3:
    if st.button("⏹️ PARAR", width='stretch'):
        st.session_state['live_running'] = False

# Inicializar estado
if 'live_running' not in st.session_state:
    st.session_state['live_running'] = False
    st.session_state['live_fase'] = 'grupos'
    st.session_state['live_jogos'] = []
    st.session_state['live_stats'] = {}
    st.session_state['live_classificados'] = []
    st.session_state['live_campeao'] = None
    st.session_state['live_vice'] = None
    st.session_state['live_semifinalistas'] = []
    st.session_state['live_final_placar'] = None

if 'historico_copas' not in st.session_state:
    st.session_state['historico_copas'] = []

# Config de simulação
live_config = {'media_gols': MEDIA_GOLS_COPA, 'k_scale': 400, 'usar_dixon_coles': False, 'rho_dixon_coles': -0.13}

if st.session_state.get('live_running', False):
    
    # ========== FASE DE GRUPOS ==========
    if st.session_state['live_fase'] == 'grupos':
        st.markdown("## 🏟️ FASE DE GRUPOS")
        
        stats = {g: {t: [0, 0, 0, 0, 0, 0] for t in times} for g, times in grupos_dict.items()}
        todos_jogos = []
        
        for grupo in sorted(grupos_dict.keys()):
            times = grupos_dict[grupo]
            for i in range(len(times)):
                for j in range(i + 1, len(times)):
                    todos_jogos.append((grupo, times[i], times[j]))
        
        jogos_lista = []
        col_jogos, col_tabela = st.columns([3, 2])
        jogos_container = col_jogos.empty()
        tabela_container = col_tabela.empty()
        
        for idx, (grupo, t1, t2) in enumerate(todos_jogos):
            p1, p2, ga, gb, fp1, fp2, resultado = simular_jogo(elo_dict[t1], elo_dict[t2], config=live_config)
            
            stats[grupo][t1][0] += p1
            stats[grupo][t2][0] += p2
            stats[grupo][t1][4] += ga
            stats[grupo][t1][5] += gb
            stats[grupo][t2][4] += gb
            stats[grupo][t2][5] += ga
            
            if resultado == 0:
                stats[grupo][t1][1] += 1; stats[grupo][t2][3] += 1
            elif resultado == 1:
                stats[grupo][t2][1] += 1; stats[grupo][t1][3] += 1
            else:
                stats[grupo][t1][2] += 1; stats[grupo][t2][2] += 1
            
            jogos_lista.insert(0, {'Grupo': grupo, 'Jogo': f"{idx+1}/{len(todos_jogos)}", 'Time 1': t1, 'Placar': f"{ga} x {gb}", 'Time 2': t2})
            
            with jogos_container.container():
                st.markdown(f"""
                <div style="background: #ffffff; border: 1px solid #e0e0e0; border-left: 4px solid #209927; border-radius: 12px; padding: 1rem; margin-bottom: 1rem; box-shadow: 0 2px 8px rgba(0,0,0,0.06);">
                    <div style="text-align: center; color: #035C88; font-size: 0.9rem; margin-bottom: 0.5rem; font-weight: 700; font-style: italic;">GRUPO {grupo}</div>
                    <div style="display: flex; justify-content: center; align-items: center; gap: 1rem;">
                        <div style="display: flex; align-items: center; gap: 8px;">
                            <img src="{get_bandeira_url(t1, bandeiras_dict)}" style="width: 36px; border-radius: 3px;">
                            <span style="color: #2E2E2E; font-size: 1.1rem; font-weight: 600;">{t1}</span>
                        </div>
                        <div style="font-size: 2rem; font-weight: 800; color: #209927; padding: 0 1rem;">{ga} x {gb}</div>
                        <div style="display: flex; align-items: center; gap: 8px;">
                            <span style="color: #2E2E2E; font-size: 1.1rem; font-weight: 600;">{t2}</span>
                            <img src="{get_bandeira_url(t2, bandeiras_dict)}" style="width: 36px; border-radius: 3px;">
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                df_jogos_display = pd.DataFrame(jogos_lista[:15])
                st.dataframe(df_jogos_display, width='stretch', height=300, hide_index=True)
            
            with tabela_container.container():
                for g in sorted(grupos_dict.keys())[:4]:
                    ranking = sorted(stats[g].items(), key=lambda x: (x[1][0], x[1][4] - x[1][5], x[1][4]), reverse=True)
                    df_tab = pd.DataFrame([
                        {'Pos': i+1, 'Seleção': sel, 'P': s[0], 'V': s[1], 'E': s[2], 'D': s[3], 'SG': s[4] - s[5]}
                        for i, (sel, s) in enumerate(ranking)
                    ])
                    st.markdown(f"**Grupo {g}**")
                    st.dataframe(df_tab, width='stretch', height=140, hide_index=True)
            
            if delay > 0:
                time.sleep(delay)
            
            if not st.session_state.get('live_running', False):
                break
        
        if st.session_state.get('live_running', False):
            resultados_grupos = []
            for grupo in sorted(grupos_dict.keys()):
                ranking = sorted(stats[grupo].items(), key=lambda x: (x[1][0], x[1][4] - x[1][5], x[1][4]), reverse=True)
                for pos, (sel, s) in enumerate(ranking):
                    resultados_grupos.append((sel, grupo, pos + 1, s[0], s[4] - s[5], s[4], 0))
            
            st.session_state['live_resultados_grupos'] = resultados_grupos
            st.session_state['live_fase'] = 'oitavas'
            st.rerun()
    
    # ========== MATA-MATA ==========
    elif st.session_state['live_fase'] in ['oitavas', 'quartas', 'semis', 'final']:
        nomes_fases = {'oitavas': '⚔️ OITAVAS DE FINAL', 'quartas': '🔥 QUARTAS DE FINAL', 'semis': '💥 SEMIFINAIS', 'final': '🏆 FINAL'}
        st.markdown(f"## {nomes_fases[st.session_state['live_fase']]}")
        
        if st.session_state['live_fase'] == 'oitavas':
            resultados = st.session_state['live_resultados_grupos']
            primeiros = [(t, s[0], s[1], s[2], s[3]) for t, g, p, *s in resultados if p == 1]
            segundos = [(t, s[0], s[1], s[2], s[3]) for t, g, p, *s in resultados if p == 2]
            terceiros = sorted([(t, s[0], s[1], s[2], s[3]) for t, g, p, *s in resultados if p == 3],
                               key=lambda x: (x[1], x[2], x[3], x[4]), reverse=True)[:8]
            todos = primeiros + segundos + terceiros
            classificados = sorted(todos, key=lambda x: (x[1], x[2], x[3], elo_dict[x[0]]), reverse=True)
        else:
            classificados = st.session_state['live_classificados']
        
        if st.session_state['live_fase'] == 'semis':
            st.session_state['live_semifinalistas'] = [c[0] for c in classificados]
        
        vencedores = []
        n = len(classificados)
        jogos_fase = []
        
        col_jogos, col_class = st.columns([3, 2])
        jogos_container = col_jogos.empty()
        class_container = col_class.empty()
        
        for i in range(n // 2):
            t1, t2 = classificados[i][0], classificados[n - 1 - i][0]
            p1, p2, ga, gb, fp1, fp2, resultado = simular_jogo(elo_dict[t1], elo_dict[t2], mata_mata=True, config=live_config)
            
            ganhador = t1 if resultado == 0 else t2
            perdedor = t2 if resultado == 0 else t1
            dados = classificados[i] if resultado == 0 else classificados[n - 1 - i]
            vencedores.append(dados)
            
            penaltis = " (pen)" if ga == gb else ""
            
            if st.session_state['live_fase'] == 'final':
                st.session_state['live_final_placar'] = f"{t1} {ga} x {gb} {t2}{penaltis}"
                st.session_state['live_vice'] = perdedor
            
            jogos_fase.append({'Time 1': t1, 'Placar': f"{ga} x {gb}", 'Time 2': t2, 'Vencedor': f"{ganhador}{penaltis}"})
            
            with jogos_container.container():
                st.markdown(f"""
                <div style="background: #ffffff; border: 1px solid #e0e0e0; border-left: 4px solid #209927; border-radius: 12px; padding: 1rem; margin-bottom: 1rem; box-shadow: 0 2px 8px rgba(0,0,0,0.06);">
                    <div style="display: flex; justify-content: center; align-items: center; gap: 1rem;">
                        <div style="display: flex; align-items: center; gap: 8px;">
                            <img src="{get_bandeira_url(t1, bandeiras_dict)}" style="width: 40px; border-radius: 3px;">
                            <span style="color: #2E2E2E; font-size: 1.1rem; font-weight: 600;">{t1}</span>
                        </div>
                        <div style="font-size: 2.2rem; font-weight: 800; color: #209927; padding: 0 1rem;">{ga} x {gb}</div>
                        <div style="display: flex; align-items: center; gap: 8px;">
                            <span style="color: #2E2E2E; font-size: 1.1rem; font-weight: 600;">{t2}</span>
                            <img src="{get_bandeira_url(t2, bandeiras_dict)}" style="width: 40px; border-radius: 3px;">
                        </div>
                    </div>
                    <div style="text-align: center; margin-top: 0.5rem; color: #209927; font-size: 1rem; font-weight: 700;">
                        🏆 {ganhador}{penaltis}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                df_fase = pd.DataFrame(jogos_fase)
                st.dataframe(df_fase, width='stretch', hide_index=True)
            
            with class_container.container():
                st.markdown("### ✅ Classificados")
                for v in vencedores:
                    st.success(f"🏆 {v[0]}")
            
            if delay > 0:
                time.sleep(delay * 2)
        
        st.session_state['live_classificados'] = vencedores
        
        proxima = {'oitavas': 'quartas', 'quartas': 'semis', 'semis': 'final', 'final': 'campeao'}
        
        if st.session_state['live_fase'] == 'final':
            st.session_state['live_campeao'] = vencedores[0][0]
            st.session_state['live_fase'] = 'campeao'
        else:
            st.session_state['live_fase'] = proxima[st.session_state['live_fase']]
        
        time.sleep(1)
        st.rerun()
    
    # ========== CAMPEÃO ==========
    elif st.session_state['live_fase'] == 'campeao':
        st.session_state['live_running'] = False
        
        nova_copa = {
            'edicao': len(st.session_state['historico_copas']) + 1,
            'campeao': st.session_state['live_campeao'],
            'vice': st.session_state.get('live_vice', 'N/A'),
            'semifinalistas': st.session_state.get('live_semifinalistas', []),
            'final_placar': st.session_state.get('live_final_placar', 'N/A'),
            'timestamp': datetime.now().strftime('%d/%m/%Y %H:%M:%S')
        }
        st.session_state['historico_copas'].append(nova_copa)
        
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        campeao = st.session_state['live_campeao']
        vice = st.session_state.get('live_vice', 'N/A')
        with col2:
            st.markdown(f"""
            <div style="background: #ffffff; border: 3px solid #FFCF26; border-radius: 20px; padding: 3rem; text-align: center; box-shadow: 0 8px 32px rgba(255,207,38,0.3);">
                <div style="font-size: 5rem;">🏆</div>
                <img src="{get_bandeira_url(campeao, bandeiras_dict)}" style="width: 120px; height: auto; border-radius: 8px; margin: 1rem 0; box-shadow: 0 4px 20px rgba(32,153,39,0.3);">
                <div style="font-size: 3.5rem; font-weight: 900; color: #209927; margin: 0.5rem 0;">{campeao}</div>
                <div style="color: #2E2E2E; font-size: 1.5rem; font-weight: 700; font-style: italic;">CAMPEÃO DA COPA DO MUNDO 2026!</div>
                <div style="display: flex; justify-content: center; align-items: center; gap: 8px; margin-top: 1rem; color: #5a5a6a; font-size: 1rem;">
                    🥈 Vice: <img src="{get_bandeira_url(vice, bandeiras_dict)}" style="width: 24px; border-radius: 2px;"> {vice}
                </div>
                <div style="color: #8a8a9a; font-size: 0.9rem; margin-top: 0.5rem;">Final: {st.session_state.get('live_final_placar', 'N/A')}</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.balloons()

else:
    st.info("👆 Clique em **NOVA COPA** para iniciar uma simulação ao vivo!")
    
    if st.session_state.get('live_campeao'):
        col1, col2, col3 = st.columns([1, 2, 1])
        campeao = st.session_state['live_campeao']
        vice = st.session_state.get('live_vice', 'N/A')
        with col2:
            st.markdown(f"""
            <div style="background: #ffffff; border: 3px solid #FFCF26; border-radius: 20px; padding: 3rem; text-align: center; box-shadow: 0 8px 32px rgba(255,207,38,0.3);">
                <div style="font-size: 5rem;">🏆</div>
                <img src="{get_bandeira_url(campeao, bandeiras_dict)}" style="width: 120px; height: auto; border-radius: 8px; margin: 1rem 0;">
                <div style="font-size: 3.5rem; font-weight: 900; color: #209927; margin: 0.5rem 0;">{campeao}</div>
                <div style="color: #2E2E2E; font-size: 1.5rem; font-weight: 700; font-style: italic;">CAMPEÃO DA COPA DO MUNDO 2026!</div>
                <div style="display: flex; justify-content: center; align-items: center; gap: 8px; margin-top: 1rem; color: #5a5a6a; font-size: 1rem;">
                    🥈 Vice: <img src="{get_bandeira_url(vice, bandeiras_dict)}" style="width: 24px; border-radius: 2px;"> {vice}
                </div>
            </div>
            """, unsafe_allow_html=True)

# ========== HISTÓRICO ==========
st.markdown("---")
st.markdown("### 📜 Histórico de Copas Simuladas")

if len(st.session_state.get('historico_copas', [])) == 0:
    st.info("Nenhuma copa simulada ainda! Clique em 'NOVA COPA' para começar.")
else:
    historico = st.session_state['historico_copas']
    
    campeoes_count = {}
    for copa in historico:
        campeoes_count[copa['campeao']] = campeoes_count.get(copa['campeao'], 0) + 1
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🏆 Copas Simuladas", len(historico))
    with col2:
        st.metric("🥇 Campeões Únicos", len(campeoes_count))
    with col3:
        if campeoes_count:
            maior = max(campeoes_count.items(), key=lambda x: x[1])
            st.metric("👑 Maior Campeão", f"{maior[0]} ({maior[1]}x)")
    
    for copa in reversed(historico):
        st.markdown(f"""
        <div style="background: #ffffff; border: 1px solid #e0e0e0; border-left: 4px solid #209927; border-radius: 12px; padding: 1rem; margin: 0.5rem 0; box-shadow: 0 2px 8px rgba(0,0,0,0.06);">
            <div style="color: #035C88; font-size: 1rem; font-weight: 700; font-style: italic;">🏆 Copa #{copa['edicao']}</div>
            <div style="color: #209927; font-size: 1.2rem; font-weight: bold; display: flex; align-items: center; gap: 8px;">
                <img src="{get_bandeira_url(copa['campeao'], bandeiras_dict)}" style="width: 28px; border-radius: 3px;"> {copa['campeao']}
            </div>
            <div style="color: #8a8a9a; font-size: 0.85rem;">🥈 Vice: {copa['vice']} | Final: {copa['final_placar']}</div>
        </div>
        """, unsafe_allow_html=True)
    
    if st.button("🗑️ Limpar Histórico", width='stretch'):
        st.session_state['historico_copas'] = []
        st.rerun()




