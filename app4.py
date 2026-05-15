import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os

# ============================================================
# CONFIGURAÇÃO
# ============================================================
st.set_page_config(
    page_title="Desenrola Brasil - Análise de Renegociação",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CORES BANCÁRIAS / FINANCEIRAS
# ============================================================
# Paleta inspirada em instituições financeiras
COR_PRIMARIA = "#0A2540"      # Azul escuro (executivo, confiança)
COR_SECUNDARIA = "#0066CC"    # Azul médio (bancário)
COR_DESTAQUE = "#00A86B"      # Verde (crescimento, dinheiro)
COR_ALERTA = "#FF6B35"        # Laranja (atenção)
COR_FUNDO = "#F8FAFC"         # Cinza claro (clean)
COR_TEXTO = "#1E293B"         # Cinza escuro
COR_SUBTEXTO = "#64748B"      # Cinza médio

# ============================================================
# CSS PERSONALIZADO (visual bancário)
# ============================================================
st.markdown(f"""
<style>
.stApp {{ background: {COR_FUNDO} !important; }}
html, body, .stApp {{ color: {COR_TEXTO} !important; font-family: 'Inter', 'Segoe UI', sans-serif !important; }}
#MainMenu, footer, header {{ visibility: hidden; }}
.block-container {{ padding: 2rem 2.5rem !important; max-width: 1400px !important; }}

/* Header bancário */
.page-header {{
    background: linear-gradient(135deg, {COR_PRIMARIA}, {COR_SECUNDARIA});
    padding: 1.5rem 2rem;
    border-radius: 12px;
    margin-bottom: 28px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
}}
.page-header h1 {{
    font-size: 28px !important;
    font-weight: 700 !important;
    color: white !important;
    margin: 0 0 8px 0 !important;
}}
.page-header p {{
    color: rgba(255,255,255,0.85);
    font-size: 14px;
    margin: 0;
}}
.update-badge {{
    background: {COR_DESTAQUE};
    color: white;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 600;
    display: inline-block;
    margin-top: 12px;
}}

/* Cards de métricas (estilo dashboard financeiro) */
.kpi-grid {{ display: flex; gap: 20px; margin-bottom: 32px; flex-wrap: wrap; }}
.kpi-card {{
    flex: 1; min-width: 180px; background: white; border-radius: 12px;
    padding: 24px 20px; text-align: center; border: 1px solid #E2E8F0;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    border-bottom: 3px solid {COR_SECUNDARIA};
    transition: transform 0.2s;
}}
.kpi-card:hover {{
    transform: translateY(-3px);
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
}}
.kpi-label {{ font-size: 12px; text-transform: uppercase; color: {COR_SUBTEXTO}; margin-bottom: 8px; font-weight: 600; letter-spacing: 0.5px; }}
.kpi-value {{ font-size: 32px; font-weight: 800; color: {COR_PRIMARIA}; margin: 8px 0; }}
.kpi-sub {{ font-size: 11px; color: {COR_SUBTEXTO}; }}

/* Cards de insight */
.insight-card {{
    background: white;
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 20px;
    border: 1px solid #E2E8F0;
    box-shadow: 0 1px 2px rgba(0,0,0,0.03);
}}
.insight-title {{ font-size: 13px; font-weight: 700; color: {COR_SECUNDARIA}; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 0.5px; }}
.insight-value {{ font-size: 24px; font-weight: 800; color: {COR_PRIMARIA}; margin: 10px 0; }}
.insight-text {{ font-size: 13px; color: {COR_SUBTEXTO}; line-height: 1.5; }}

/* Headers de seção */
.section-header {{
    display: flex; align-items: center; justify-content: space-between;
    margin: 32px 0 20px 0; border-bottom: 2px solid #E2E8F0;
    padding-bottom: 12px;
}}
.section-header h2 {{ font-size: 20px; font-weight: 700; color: {COR_PRIMARIA}; margin: 0; }}
.section-tag {{
    background: #E8F4FD; padding: 4px 12px; border-radius: 20px;
    font-size: 11px; color: {COR_SECUNDARIA}; font-weight: 600;
}}

/* Tabelas */
[data-testid="stDataFrame"] {{
    border-radius: 12px;
    border: 1px solid #E2E8F0;
}}

/* Footer */
.footer {{
    text-align: center; padding: 30px 0 20px; margin-top: 40px;
    border-top: 1px solid #E2E8F0; color: {COR_SUBTEXTO}; font-size: 12px;
}}
</style>
""", unsafe_allow_html=True)

# ============================================================
# HEADER
# ============================================================
st.markdown(f"""
<div class="page-header">
    <h1>🏦 Desenrola Brasil</h1>
    <p>Análise de Renegociação de Dívidas | Dados Oficiais do Banco Central</p>
    <div class="update-badge">📅 Dados atualizados até março/2026</div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# FUNÇÕES
# ============================================================
def fmt_brl(valor):
    if pd.isna(valor) or valor == 0:
        return "R$ 0"
    if abs(valor) >= 1_000_000_000:
        return f"R$ {valor/1_000_000_000:.1f}B".replace(".", ",")
    if abs(valor) >= 1_000_000:
        return f"R$ {valor/1_000_000:.1f}M".replace(".", ",")
    return f"R$ {valor:,.0f}".replace(",", ".")

def fmt_num(n):
    if pd.isna(n):
        return "0"
    return f"{int(n):,}".replace(",", ".")

@st.cache_data
def carregar_dados():
    """Carrega o CSV diretamente do repositório"""
    encodings = ['latin1', 'ISO-8859-1', 'cp1252', 'utf-8', 'WIN1252']
    
    for encoding in encodings:
        try:
            df = pd.read_csv('dados_desenrola.csv', delimiter=';', encoding=encoding, low_memory=False)
            df.columns = df.columns.str.lower().str.strip()
            
            if 'numero_operacoes' in df.columns:
                df['numero_operacoes'] = pd.to_numeric(
                    df['numero_operacoes'].astype(str).str.replace('.', '').str.replace(',', '.'),
                    errors='coerce'
                )
            
            if 'volume_operacoes' in df.columns:
                df['volume_operacoes'] = pd.to_numeric(
                    df['volume_operacoes'].astype(str).str.replace(',', '.').str.extract(r'(\d+\.?\d*)', expand=False),
                    errors='coerce'
                )
            
            df = df.dropna(subset=['numero_operacoes', 'volume_operacoes'])
            return df, encoding
            
        except Exception:
            continue
    
    return None, None

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("### 🎯 Informações")
    st.markdown("""
    **Programa Desenrola Brasil**
    
    - **Fonte:** Banco Central do Brasil (SCR)
    - **Período:** set/2023 - mar/2026
    - **Cobertura:** Nacional
    
    ---
    
    **Tipos do programa:**
    
    🔵 **Tipo 1:** Faixa 1 (Pessoa Física)
    🟢 **Tipo 2:** Faixa 2 (Pessoa Física)
    🟡 **Tipo 3:** Pequenos Negócios
    
    ---
    
    📥 [Dados oficiais do BC](https://www.bcb.gov.br/)
    """)
    
    st.markdown("---")
    st.caption("Dashboard desenvolvido para portfólio de Análise de Dados")

# ============================================================
# MAIN
# ============================================================
with st.spinner("🔄 Carregando dados do Banco Central..."):
    df, encoding = carregar_dados()

if df is not None and len(df) > 0:
    st.success(f"✅ Dados carregados com sucesso! {len(df):,} registros processados")
    
    # ===== DADOS AGREGADOS =====
    total_operacoes = df['numero_operacoes'].sum()
    total_volume = df['volume_operacoes'].sum()
    ticket_medio = total_volume / total_operacoes if total_operacoes > 0 else 0
    
    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">💰 VOLUME RENEGOCIADO</div>
            <div class="kpi-value">{fmt_brl(total_volume)}</div>
            <div class="kpi-sub">total de dívidas renegociadas</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">📋 OPERAÇÕES</div>
            <div class="kpi-value">{fmt_num(total_operacoes)}</div>
            <div class="kpi-sub">renegociações realizadas</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">🎫 TICKET MÉDIO</div>
            <div class="kpi-value">{fmt_brl(ticket_medio)}</div>
            <div class="kpi-sub">por operação</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">📅 REGISTROS</div>
            <div class="kpi-value">{fmt_num(len(df))}</div>
            <div class="kpi-sub">linhas processadas</div>
        </div>
        """, unsafe_allow_html=True)
    
    # ===== INSIGHTS PRINCIPAIS =====
    st.markdown("""
    <div class="section-header">
        <h2>🔍 Principais Insights</h2>
        <span class="section-tag">Análise Estratégica</span>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="insight-card">
            <div class="insight-title">🏦 LIDERANÇA DE MERCADO</div>
            <div class="insight-value">Nubank</div>
            <div class="insight-text">Banco com o maior volume de renegociações, superando instituições tradicionais.</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="insight-card">
            <div class="insight-title">📍 CONCENTRAÇÃO REGIONAL</div>
            <div class="insight-value">Sudeste</div>
            <div class="insight-text">SP, RJ e MG respondem por quase metade do volume financeiro do programa.</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="insight-card">
            <div class="insight-title">📊 TICKET MÉDIO</div>
            <div class="insight-value">{fmt_brl(ticket_medio)}</div>
            <div class="insight-text">Valor médio renegociado por operação em todo o país.</div>
        </div>
        """, unsafe_allow_html=True)
    
    # ===== GRÁFICO 1: POR TIPO =====
    if 'tipo_desenrola' in df.columns:
        st.markdown("""
        <div class="section-header">
            <h2>📊 Análise por Tipo do Programa</h2>
            <span class="section-tag">Distribuição</span>
        </div>
        """, unsafe_allow_html=True)
        
        tipo_data = df.groupby('tipo_desenrola').agg({
            'numero_operacoes': 'sum',
            'volume_operacoes': 'sum'
        }).reset_index()
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig1 = px.pie(tipo_data, names='tipo_desenrola', values='numero_operacoes', hole=0.4,
                         title='Distribuição de Operações',
                         color_discrete_sequence=[COR_PRIMARIA, COR_SECUNDARIA, COR_DESTAQUE])
            fig1.update_layout(template="plotly_white", height=420, paper_bgcolor='white', font=dict(color=COR_TEXTO))
            fig1.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            fig2 = px.bar(tipo_data, x='tipo_desenrola', y='volume_operacoes', 
                         title='Volume Financeiro por Tipo (R$)',
                         color='tipo_desenrola',
                         color_discrete_sequence=[COR_PRIMARIA, COR_SECUNDARIA, COR_DESTAQUE],
                         text_auto='.2s')
            fig2.update_layout(template="plotly_white", height=420, paper_bgcolor='white', font=dict(color=COR_TEXTO))
            st.plotly_chart(fig2, use_container_width=True)
    
    # ===== GRÁFICO 2: TOP ESTADOS =====
    if 'unidade_federacao' in df.columns:
        st.markdown("""
        <div class="section-header">
            <h2>🗺️ Ranking por Unidade da Federação</h2>
            <span class="section-tag">Top 10</span>
        </div>
        """, unsafe_allow_html=True)
        
        uf_data = df.groupby('unidade_federacao').agg({
            'numero_operacoes': 'sum',
            'volume_operacoes': 'sum'
        }).reset_index().sort_values('numero_operacoes', ascending=False).head(10)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig3 = px.bar(uf_data, x='unidade_federacao', y='numero_operacoes', 
                         title='Operações por UF',
                         color='numero_operacoes', color_continuous_scale='Blues')
            fig3.update_layout(template="plotly_white", height=420, xaxis_title="UF", yaxis_title="Operações")
            st.plotly_chart(fig3, use_container_width=True)
        
        with col2:
            fig4 = px.bar(uf_data, x='unidade_federacao', y='volume_operacoes', 
                         title='Volume Financeiro por UF (R$)',
                         color='volume_operacoes', color_continuous_scale='Greens')
            fig4.update_layout(template="plotly_white", height=420, xaxis_title="UF", yaxis_title="Volume (R$)")
            st.plotly_chart(fig4, use_container_width=True)
    
    # ===== GRÁFICO 3: TOP BANCOS =====
    if 'nome_conglomerado_financeiro' in df.columns:
        st.markdown("""
        <div class="section-header">
            <h2>🏦 Ranking de Instituições Financeiras</h2>
            <span class="section-tag">Top 10</span>
        </div>
        """, unsafe_allow_html=True)
        
        banco_data = df.groupby('nome_conglomerado_financeiro')['numero_operacoes'].sum().sort_values(ascending=False).head(10).reset_index()
        banco_data.columns = ['Instituição', 'Operações']
        
        fig5 = px.bar(banco_data, x='Operações', y='Instituição', orientation='h',
                     title='Top 10 Instituições por Renegociações',
                     color='Operações', color_continuous_scale='Viridis',
                     text='Operações')
        fig5.update_layout(template="plotly_white", height=500, paper_bgcolor='white')
        fig5.update_traces(texttemplate='%{text:,}', textposition='outside')
        st.plotly_chart(fig5, use_container_width=True)
    
    # ===== CONCLUSÃO =====
    st.markdown("""
    <div class="section-header">
        <h2>📈 Conclusão</h2>
        <span class="section-tag">Resumo Executivo</span>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="insight-card" style="background: linear-gradient(135deg, #0A2540, #0066CC); color: white; border: none;">
        <div class="insight-text" style="color: rgba(255,255,255,0.9);">
        O Programa <strong>Desenrola Brasil</strong> já renegociou <strong>{fmt_brl(total_volume)}</strong> em dívidas, 
        com <strong>{fmt_num(total_operacoes)} operações</strong> realizadas. O <strong>Nubank</strong> lidera as renegociações,
        enquanto a região <strong>Sudeste</strong> concentra a maior parte do volume financeiro.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # ===== EXPORTAÇÃO =====
    st.markdown("---")
    st.markdown("### 📥 Exportar Dados")
    
    relatorio = f"""RELATÓRIO DESENROLA BRASIL
Data: {datetime.now().strftime('%d/%m/%Y %H:%M')}

VISÃO GERAL:
- Total Renegociado: {fmt_brl(total_volume)}
- Total de Operações: {fmt_num(total_operacoes)}
- Ticket Médio: {fmt_brl(ticket_medio)}

Fonte: Banco Central do Brasil (SCR/Desenrola)
"""
    
    st.download_button(
        label="📝 Baixar Relatório (TXT)",
        data=relatorio,
        file_name=f"relatorio_desenrola_{datetime.now().strftime('%Y%m%d')}.txt",
        mime="text/plain"
    )
    
    # ===== FOOTER =====
    st.markdown(f"""
    <div class="footer">
        🏦 Desenrola Brasil · Fonte: Banco Central do Brasil (SCR)<br>
        Dashboard desenvolvido para portfólio de Análise de Dados
    </div>
    """, unsafe_allow_html=True)

else:
    st.error("❌ Erro ao carregar o arquivo. Verifique se o arquivo 'dados_desenrola.csv' está no repositório.")

# ============================================================
# FIM
# ============================================================
