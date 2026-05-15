import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import unicodedata
import re

# ============================================================
# CONFIGURAÇÃO DA PÁGINA
# ============================================================
st.set_page_config(
    page_title="Desenrola Brasil - Programa de Renegociação de Dívidas",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# INICIALIZAÇÃO DO TEMA
# ============================================================
if 'tema' not in st.session_state:
    st.session_state.tema = 'claro'

# ============================================================
# CORES POR TEMA
# ============================================================
if st.session_state.tema == 'claro':
    COR_PRIMARIA = "#00A86B"
    COR_SECUNDARIA = "#0052CC"
    COR_DESTAQUE = "#FFB800"
    COR_ALERTA = "#FF6B35"
    COR_FUNDO = "#F8FAFC"
    COR_CARD = "#FFFFFF"
    COR_TEXTO = "#1E293B"
    COR_SUBTEXTO = "#64748B"
    COR_BORDA = "#E2E8F0"
    COR_NORTE = "#3B82F6"
    COR_NORDESTE = "#10B981"
    COR_CENTRO_OESTE = "#F59E0B"
    COR_SUDESTE = "#EF4444"
    COR_SUL = "#8B5CF6"
else:
    COR_PRIMARIA = "#10B981"
    COR_SECUNDARIA = "#60A5FA"
    COR_DESTAQUE = "#FBBF24"
    COR_ALERTA = "#FB923C"
    COR_FUNDO = "#0F172A"
    COR_CARD = "#1E293B"
    COR_TEXTO = "#F8FAFC"
    COR_SUBTEXTO = "#94A3B8"
    COR_BORDA = "#334155"
    COR_NORTE = "#60A5FA"
    COR_NORDESTE = "#34D399"
    COR_CENTRO_OESTE = "#FBBF24"
    COR_SUDESTE = "#F87171"
    COR_SUL = "#A78BFA"

# ============================================================
# CSS
# ============================================================
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

* {{ margin: 0; padding: 0; box-sizing: border-box; }}

html, body, .stApp {{
    background: {COR_FUNDO} !important;
    font-family: 'Inter', sans-serif !important;
    color: {COR_TEXTO} !important;
}}

.block-container {{ padding: 1rem 1.5rem !important; max-width: 1400px !important; margin: 0 auto !important; }}

/* Header */
.main-header {{
    background: linear-gradient(135deg, {COR_PRIMARIA}, {COR_SECUNDARIA});
    border-radius: 16px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1.5rem;
}}
.main-header h1 {{ font-size: 1.6rem; font-weight: 800; color: white; margin: 0; }}
.main-header p {{ color: rgba(255,255,255,0.9); font-size: 0.8rem; margin-top: 0.3rem; }}
.header-badges {{ display: flex; gap: 0.4rem; margin-top: 0.8rem; flex-wrap: wrap; }}
.header-badge {{ background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); padding: 0.2rem 0.6rem; border-radius: 20px; font-size: 0.6rem; font-weight: 500; color: white; }}

/* KPIs */
.kpi-grid {{ display: flex; gap: 0.8rem; margin-bottom: 1.5rem; flex-wrap: wrap; }}
.kpi-card {{ flex: 1; min-width: 140px; background: {COR_CARD}; border-radius: 14px; padding: 0.8rem; border: 1px solid {COR_BORDA}; border-bottom: 3px solid {COR_PRIMARIA}; transition: transform 0.2s; }}
.kpi-card:hover {{ transform: translateY(-2px); }}
.kpi-icon {{ font-size: 1.3rem; margin-bottom: 0.3rem; }}
.kpi-label {{ font-size: 0.6rem; text-transform: uppercase; color: {COR_SUBTEXTO}; font-weight: 600; }}
.kpi-value {{ font-size: 1.3rem; font-weight: 800; color: {COR_TEXTO}; margin: 0.3rem 0; }}
.kpi-sub {{ font-size: 0.55rem; color: {COR_SUBTEXTO}; }}

/* Seções */
.section-header {{ display: flex; align-items: center; justify-content: space-between; margin: 1rem 0 0.8rem 0; border-bottom: 2px solid {COR_BORDA}; padding-bottom: 0.4rem; }}
.section-header h2 {{ font-size: 1rem; font-weight: 700; color: {COR_TEXTO}; margin: 0; }}
.section-badge {{ background: {COR_PRIMARIA}; color: white; padding: 0.15rem 0.5rem; border-radius: 20px; font-size: 0.55rem; font-weight: 600; }}

/* Insight cards */
.insight-card {{ background: {COR_CARD}; border-radius: 12px; padding: 0.8rem; margin-bottom: 0.6rem; border-left: 3px solid {COR_DESTAQUE}; }}
.insight-title {{ font-size: 0.55rem; font-weight: 700; text-transform: uppercase; color: {COR_SECUNDARIA}; }}
.insight-value {{ font-size: 1.1rem; font-weight: 800; color: {COR_TEXTO}; margin: 0.3rem 0; }}
.insight-text {{ font-size: 0.65rem; color: {COR_SUBTEXTO}; }}

/* Tabelas */
[data-testid="stDataFrame"] th {{ background: {COR_SECUNDARIA} !important; color: white !important; padding: 6px 8px !important; font-size: 0.65rem !important; }}
[data-testid="stDataFrame"] td {{ background: {COR_CARD} !important; color: {COR_TEXTO} !important; padding: 5px 8px !important; font-size: 0.6rem !important; }}

/* Sidebar */
section[data-testid="stSidebar"] {{ background: {COR_CARD} !important; border-right: 1px solid {COR_BORDA} !important; padding: 0.8rem !important; }}

/* Footer */
.footer {{ text-align: center; padding: 0.8rem; margin-top: 1rem; border-top: 1px solid {COR_BORDA}; font-size: 0.55rem; color: {COR_SUBTEXTO}; }}
</style>
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

def fmt_percentual(valor, total):
    if total == 0:
        return "0%"
    return f"{(valor/total*100):.1f}%"

def classificar_banco_corrigido(nome):
    nome_upper = str(nome).upper().strip()
    nome_upper = re.sub(r'\s*-\s*PRUDENCIAL$', '', nome_upper)
    
    if any(digital in nome_upper for digital in ['NUBANK', 'INTER', 'C6']):
        return 'digital'
    if any(tradicional in nome_upper for tradicional in ['ITAU', 'BRADESCO', 'SANTANDER', 'BB', 'CAIXA', 'BANCO DO BRASIL']):
        return 'tradicional'
    if any(outro in nome_upper for outro in ['BTG', 'BMG', 'BANRISUL', 'VOTORANTIM', 'BCO DO NORDESTE', 'BCO DO EST']):
        return 'outros'
    return 'outros'

def agrupar_por_regiao(uf):
    regioes = {
        'NORTE': ['AC', 'AP', 'AM', 'PA', 'RO', 'RR', 'TO'],
        'NORDESTE': ['AL', 'BA', 'CE', 'MA', 'PB', 'PE', 'PI', 'RN', 'SE'],
        'CENTRO_OESTE': ['DF', 'GO', 'MS', 'MT'],
        'SUDESTE': ['ES', 'MG', 'RJ', 'SP'],
        'SUL': ['PR', 'RS', 'SC']
    }
    for regiao, ufs in regioes.items():
        if uf in ufs:
            return regiao
    return 'OUTROS'

def calcular_hhi(data, col_valor='numero_operacoes'):
    total = data[col_valor].sum()
    if total == 0:
        return 0
    participacoes = (data[col_valor] / total) ** 2
    hhi = participacoes.sum() * 10000
    return hhi

@st.cache_data
def carregar_dados():
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
            if 'data_base' in df.columns:
                df['data_base'] = pd.to_datetime(df['data_base'].astype(str), format='%Y%m', errors='coerce')
            
            if 'nome_conglomerado_financeiro' in df.columns:
                df['tipo_banco'] = df['nome_conglomerado_financeiro'].apply(classificar_banco_corrigido)
            
            if 'unidade_federacao' in df.columns:
                df['macrorregiao'] = df['unidade_federacao'].apply(agrupar_por_regiao)
            
            df = df.dropna(subset=['numero_operacoes', 'volume_operacoes'])
            return df, encoding
        except:
            continue
    return None, None

# ============================================================
# CARREGAR DADOS
# ============================================================
with st.spinner("🔄 Carregando dados..."):
    df, encoding = carregar_dados()

if df is None:
    st.error("❌ Erro ao carregar os dados. Verifique o arquivo 'dados_desenrola.csv'")
    st.stop()

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("### 🎛️ Controles")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("☀️ Claro", use_container_width=True):
            st.session_state.tema = 'claro'
            st.rerun()
    with col2:
        if st.button("🌙 Escuro", use_container_width=True):
            st.session_state.tema = 'escuro'
            st.rerun()
    
    st.markdown("---")
    st.markdown("### 🎯 Filtros")
    
    df_filtrado = df.copy()
    
    if 'tipo_desenrola' in df.columns:
        tipos = sorted(df['tipo_desenrola'].dropna().unique())
        tipo_filtro = st.multiselect("Tipo do Programa", tipos, default=tipos)
        if tipo_filtro:
            df_filtrado = df_filtrado[df_filtrado['tipo_desenrola'].isin(tipo_filtro)]
    
    if 'unidade_federacao' in df.columns:
        ufs = sorted(df['unidade_federacao'].dropna().unique())
        uf_filtro = st.multiselect("UF", ufs, default=[])
        if uf_filtro:
            df_filtrado = df_filtrado[df_filtrado['unidade_federacao'].isin(uf_filtro)]
    
    if 'macrorregiao' in df.columns:
        regioes = sorted(df['macrorregiao'].dropna().unique())
        regiao_filtro = st.multiselect("Macrorregião", regioes, default=[])
        if regiao_filtro:
            df_filtrado = df_filtrado[df_filtrado['macrorregiao'].isin(regiao_filtro)]
    
    if 'nome_conglomerado_financeiro' in df.columns:
        bancos = sorted(df['nome_conglomerado_financeiro'].dropna().unique())
        banco_filtro = st.multiselect("Instituição", bancos, default=[])
        if banco_filtro:
            df_filtrado = df_filtrado[df_filtrado['nome_conglomerado_financeiro'].isin(banco_filtro)]
    
    if 'data_base' in df.columns and not df['data_base'].isna().all():
        min_date = df['data_base'].min().date()
        max_date = df['data_base'].max().date()
        periodo_filtro = st.slider("Período", min_date, max_date, (min_date, max_date))
        df_filtrado = df_filtrado[(df_filtrado['data_base'].dt.date >= periodo_filtro[0]) & 
                                   (df_filtrado['data_base'].dt.date <= periodo_filtro[1])]
    
    st.markdown("---")
    st.markdown("### 📊 Tipos do programa")
    st.markdown("🔵 Tipo 1: Faixa 1 (PF)")
    st.markdown("🟢 Tipo 2: Faixa 2 (PF)")
    st.markdown("🟡 Tipo 3: Pequenos Negócios")

# ============================================================
# HEADER
# ============================================================
st.markdown(f"""
<div class="main-header">
    <h1>🏦 Desenrola Brasil</h1>
    <p>Programa de Renegociação de Dívidas - Dados oficiais do Banco Central</p>
    <div class="header-badges">
        <span class="header-badge">🏛️ Fonte Oficial</span>
        <span class="header-badge">📊 Dados Abertos</span>
        <span class="header-badge">✅ SCR</span>
        <span class="header-badge">📋 Lei 14.690/2023</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# MAIN
# ============================================================
if df_filtrado is not None and len(df_filtrado) > 0:
    st.success(f"✅ {len(df_filtrado):,} registros processados")
    
    # ===== KPIs =====
    total_operacoes = df_filtrado['numero_operacoes'].sum()
    total_volume = df_filtrado['volume_operacoes'].sum()
    ticket_medio = total_volume / total_operacoes if total_operacoes > 0 else 0
    num_bancos = df_filtrado['nome_conglomerado_financeiro'].nunique() if 'nome_conglomerado_financeiro' in df_filtrado.columns else 0
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'<div class="kpi-card"><div class="kpi-icon">💰</div><div class="kpi-label">VOLUME</div><div class="kpi-value">{fmt_brl(total_volume)}</div><div class="kpi-sub">renegociado</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="kpi-card"><div class="kpi-icon">📋</div><div class="kpi-label">OPERAÇÕES</div><div class="kpi-value">{fmt_num(total_operacoes)}</div><div class="kpi-sub">realizadas</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="kpi-card"><div class="kpi-icon">🎫</div><div class="kpi-label">TICKET MÉDIO</div><div class="kpi-value">{fmt_brl(ticket_medio)}</div><div class="kpi-sub">por operação</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="kpi-card"><div class="kpi-icon">🏛️</div><div class="kpi-label">INSTITUIÇÕES</div><div class="kpi-value">{fmt_num(num_bancos)}</div><div class="kpi-sub">bancos</div></div>', unsafe_allow_html=True)
    
    # ===== HHI =====
    if 'nome_conglomerado_financeiro' in df_filtrado.columns:
        st.markdown('<div class="section-header"><h2>📊 Concentração de Mercado (HHI)</h2><span class="section-badge">Risco</span></div>', unsafe_allow_html=True)
        
        hhi_operacoes = calcular_hhi(df_filtrado, 'numero_operacoes')
        hhi_volume = calcular_hhi(df_filtrado, 'volume_operacoes')
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f'<div class="insight-card"><div class="insight-title">🏦 HHI (Operações)</div><div class="insight-value">{hhi_operacoes:.0f}</div><div class="insight-text">Concentração por número de operações</div></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="insight-card"><div class="insight-title">💰 HHI (Volume)</div><div class="insight-value">{hhi_volume:.0f}</div><div class="insight-text">Concentração por valor financeiro</div></div>', unsafe_allow_html=True)
    
    # ===== ANÁLISE POR MACRORREGIÃO =====
    if 'macrorregiao' in df_filtrado.columns:
        st.markdown('<div class="section-header"><h2>🗺️ Análise por Macrorregião</h2><span class="section-badge">Geográfico</span></div>', unsafe_allow_html=True)
        
        regiao_data = df_filtrado.groupby('macrorregiao')['volume_operacoes'].sum().reset_index()
        regiao_data = regiao_data[regiao_data['macrorregiao'] != 'OUTROS']
        
        fig_regiao = px.bar(regiao_data, x='macrorregiao', y='volume_operacoes',
                           title='Volume Renegociado por Macrorregião (R$)',
                           color='macrorregiao',
                           color_discrete_map={'NORTE': COR_NORTE, 'NORDESTE': COR_NORDESTE,
                                               'CENTRO_OESTE': COR_CENTRO_OESTE, 'SUDESTE': COR_SUDESTE, 'SUL': COR_SUL})
        fig_regiao.update_layout(template="plotly_white", height=400, paper_bgcolor=COR_CARD, font=dict(color=COR_TEXTO))
        st.plotly_chart(fig_regiao, use_container_width=True)
    
    # ===== EVOLUÇÃO MENSAL =====
    if 'data_base' in df_filtrado.columns and len(df_filtrado['data_base'].dropna()) > 1:
        st.markdown('<div class="section-header"><h2>📈 Evolução Mensal</h2><span class="section-badge">Temporal</span></div>', unsafe_allow_html=True)
        
        evolucao = df_filtrado.groupby('data_base').agg({
            'numero_operacoes': 'sum',
            'volume_operacoes': 'sum'
        }).reset_index().sort_values('data_base')
        
        tab1, tab2 = st.tabs(["💰 Volume (R$)", "📋 Operações"])
        
        with tab1:
            fig_vol = px.line(evolucao, x='data_base', y='volume_operacoes', markers=True, title='Evolução do Volume por Mês')
            fig_vol.update_layout(template="plotly_white", height=350, paper_bgcolor=COR_CARD, font=dict(color=COR_TEXTO))
            fig_vol.update_traces(line=dict(width=2, color=COR_PRIMARIA))
            st.plotly_chart(fig_vol, use_container_width=True)
        
        with tab2:
            fig_op = px.line(evolucao, x='data_base', y='numero_operacoes', markers=True, title='Evolução das Operações por Mês')
            fig_op.update_layout(template="plotly_white", height=350, paper_bgcolor=COR_CARD, font=dict(color=COR_TEXTO))
            fig_op.update_traces(line=dict(width=2, color=COR_SECUNDARIA))
            st.plotly_chart(fig_op, use_container_width=True)
    
    # ===== ANÁLISE POR TIPO =====
    if 'tipo_desenrola' in df_filtrado.columns:
        st.markdown('<div class="section-header"><h2>📊 Análise por Tipo do Programa</h2><span class="section-badge">Distribuição</span></div>', unsafe_allow_html=True)
        
        tipo_data = df_filtrado.groupby('tipo_desenrola').agg({
            'numero_operacoes': 'sum',
            'volume_operacoes': 'sum'
        }).reset_index()
        tipo_data['ticket_medio'] = tipo_data['volume_operacoes'] / tipo_data['numero_operacoes']
        
        col1, col2 = st.columns(2)
        with col1:
            fig_pie = px.pie(tipo_data, names='tipo_desenrola', values='numero_operacoes', hole=0.4,
                            title='Distribuição de Operações',
                            color_discrete_sequence=[COR_PRIMARIA, COR_SECUNDARIA, COR_DESTAQUE])
            fig_pie.update_layout(template="plotly_white", height=380, paper_bgcolor=COR_CARD)
            st.plotly_chart(fig_pie, use_container_width=True)
            
            # Destaque Tipo 3 - CORRIGIDO
            tipo3 = tipo_data[tipo_data['tipo_desenrola'] == 3]
            if len(tipo3) > 0:
                pct_oper = (tipo3['numero_operacoes'].iloc[0] / total_operacoes) * 100
                pct_vol = (tipo3['volume_operacoes'].iloc[0] / total_volume) * 100
                st.info(f"🟡 **Tipo 3 (Pequenos Negócios)** representa {pct_oper:.1f}% das operações, mas {pct_vol:.1f}% do volume financeiro")
        
        with col2:
            fig_ticket = px.bar(tipo_data, x='tipo_desenrola', y='ticket_medio',
                               title='Ticket Médio por Tipo (R$)',
                               color='tipo_desenrola',
                               color_discrete_sequence=[COR_PRIMARIA, COR_SECUNDARIA, COR_DESTAQUE])
            fig_ticket.update_layout(template="plotly_white", height=380, paper_bgcolor=COR_CARD)
            st.plotly_chart(fig_ticket, use_container_width=True)
    
    # ===== PARTICIPAÇÃO POR BANCO =====
    if 'nome_conglomerado_financeiro' in df_filtrado.columns:
        st.markdown('<div class="section-header"><h2>📊 Participação por Instituição</h2><span class="section-badge">Market Share</span></div>', unsafe_allow_html=True)
        
        participacao = df_filtrado.groupby('nome_conglomerado_financeiro')['numero_operacoes'].sum().sort_values(ascending=False).head(10).reset_index()
        participacao.columns = ['Instituição', 'Operações']
        
        fig_participacao = px.pie(participacao.head(7), names='Instituição', values='Operações', hole=0.4,
                                   title='Top 7 Instituições - Market Share',
                                   color_discrete_sequence=[COR_PRIMARIA, COR_SECUNDARIA, COR_DESTAQUE, COR_ALERTA, '#8B5CF6', '#EC4899', '#06B6D4'])
        fig_participacao.update_layout(template="plotly_white", height=400, paper_bgcolor=COR_CARD, font=dict(color=COR_TEXTO))
        st.plotly_chart(fig_participacao, use_container_width=True)
        
        participacao['Participação'] = (participacao['Operações'] / participacao['Operações'].sum() * 100).round(1).astype(str) + '%'
        participacao['Operações'] = participacao['Operações'].apply(fmt_num)
        st.dataframe(participacao, use_container_width=True, hide_index=True)
    
    # ===== RANKING ESTADOS =====
    if 'unidade_federacao' in df_filtrado.columns:
        st.markdown('<div class="section-header"><h2>🗺️ Top 10 Estados</h2><span class="section-badge">Ranking</span></div>', unsafe_allow_html=True)
        
        uf_data = df_filtrado.groupby('unidade_federacao')['numero_operacoes'].sum().sort_values(ascending=False).head(10).reset_index()
        uf_data.columns = ['UF', 'Operações']
        
        fig_uf = px.bar(uf_data, x='UF', y='Operações', title='Operações por UF', color='Operações', color_continuous_scale='Blues')
        fig_uf.update_layout(template="plotly_white", height=400, paper_bgcolor=COR_CARD)
        st.plotly_chart(fig_uf, use_container_width=True)
    
    # ===== RANKING BANCOS =====
    if 'nome_conglomerado_financeiro' in df_filtrado.columns:
        st.markdown('<div class="section-header"><h2>🏦 Top 10 Instituições</h2><span class="section-badge">Ranking</span></div>', unsafe_allow_html=True)
        
        banco_data = df_filtrado.groupby('nome_conglomerado_financeiro')['numero_operacoes'].sum().sort_values(ascending=False).head(10).reset_index()
        banco_data.columns = ['Instituição', 'Operações']
        
        fig_banco = px.bar(banco_data, x='Operações', y='Instituição', orientation='h', 
                          title='Top 10 Instituições por Renegociações', color='Operações', color_continuous_scale='Viridis')
        fig_banco.update_layout(template="plotly_white", height=450, paper_bgcolor=COR_CARD)
        st.plotly_chart(fig_banco, use_container_width=True)
    
    # ===== TICKET MÉDIO POR ESTADO =====
    if 'unidade_federacao' in df_filtrado.columns:
        st.markdown('<div class="section-header"><h2>🎫 Ticket Médio por Estado</h2><span class="section-badge">Ranking</span></div>', unsafe_allow_html=True)
        
        ticket_uf = df_filtrado.groupby('unidade_federacao').agg({
            'numero_operacoes': 'sum',
            'volume_operacoes': 'sum'
        }).reset_index()
        ticket_uf['ticket_medio'] = ticket_uf['volume_operacoes'] / ticket_uf['numero_operacoes']
        ticket_uf = ticket_uf.sort_values('ticket_medio', ascending=False).head(10)
        ticket_uf['ticket_formatado'] = ticket_uf['ticket_medio'].apply(fmt_brl)
        
        fig_ticket_uf = px.bar(ticket_uf, x='unidade_federacao', y='ticket_medio',
                               title='Top 10 Estados por Ticket Médio (R$)',
                               color='ticket_medio', color_continuous_scale='Greens',
                               text='ticket_formatado')
        fig_ticket_uf.update_layout(template="plotly_white", height=400, paper_bgcolor=COR_CARD)
        st.plotly_chart(fig_ticket_uf, use_container_width=True)
    
    # ===== CONCLUSÃO =====
    st.markdown('<div class="section-header"><h2>📈 Conclusão</h2><span class="section-badge">Resumo</span></div>', unsafe_allow_html=True)
    
    banco_lider = df_filtrado.groupby('nome_conglomerado_financeiro')['numero_operacoes'].sum().idxmax() if 'nome_conglomerado_financeiro' in df_filtrado.columns else "N/A"
    
    st.markdown(f"""
    <div class="insight-card" style="background: linear-gradient(135deg, {COR_PRIMARIA}, {COR_SECUNDARIA}); color: white; border: none;">
        <div class="insight-text" style="color: white;">O Programa <strong>Desenrola Brasil</strong> já renegociou <strong>{fmt_brl(total_volume)}</strong> em dívidas, com <strong>{fmt_num(total_operacoes)} operações</strong>. <strong>{banco_lider}</strong> lidera as renegociações.</div>
    </div>
    """, unsafe_allow_html=True)
    
    # ===== EXPORTAÇÃO =====
    st.markdown("---")
    st.markdown("### 📥 Exportar Dados")
    
    col1, col2 = st.columns(2)
    with col1:
        relatorio = f"""RELATÓRIO DESENROLA BRASIL - {datetime.now().strftime('%d/%m/%Y %H:%M')}
Total Renegociado: {fmt_brl(total_volume)}
Total de Operações: {fmt_num(total_operacoes)}
Ticket Médio: {fmt_brl(ticket_medio)}
Banco Líder: {banco_lider}
Fonte: Banco Central do Brasil (SCR)"""
        st.download_button("📝 Baixar Relatório", relatorio, f"relatorio_{datetime.now().strftime('%Y%m%d')}.txt", "text/plain")
    
    with col2:
        csv = df_filtrado.to_csv(index=False).encode('utf-8')
        st.download_button("📊 Baixar CSV", csv, f"dados_{datetime.now().strftime('%Y%m%d')}.csv", "text/csv")
    
    # ===== FOOTER =====
    st.markdown(f"""
    <div class="footer">
        🏦 Desenrola Brasil · Fonte: Banco Central do Brasil (SCR)<br>
        Dashboard para portfólio de Análise de Dados
        <div class="footer-disclaimer">⚠️ Dados de domínio público. Dashboard para fins de portfólio.</div>
    </div>
    """, unsafe_allow_html=True)

else:
    st.warning("⚠️ Nenhum dado encontrado com os filtros selecionados. Ajuste os filtros para visualizar os dados.")
