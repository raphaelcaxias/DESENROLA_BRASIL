import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import unicodedata
import io

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
# CORES POR TEMA (contraste melhorado)
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
    COR_TABELA_HEADER = "#0052CC"
    COR_TABELA_LINHA = "#FFFFFF"
    COR_TABELA_HOVER = "#F1F5F9"
else:
    COR_PRIMARIA = "#10B981"      # Verde mais claro
    COR_SECUNDARIA = "#60A5FA"    # Azul mais claro
    COR_DESTAQUE = "#FBBF24"
    COR_ALERTA = "#FB923C"
    COR_FUNDO = "#0F172A"
    COR_CARD = "#1E293B"
    COR_TEXTO = "#F8FAFC"         # Mais branco
    COR_SUBTEXTO = "#94A3B8"
    COR_BORDA = "#334155"
    COR_TABELA_HEADER = "#3B82F6"
    COR_TABELA_LINHA = "#1E293B"
    COR_TABELA_HOVER = "#334155"

# ============================================================
# CSS PREMIUM (contraste melhorado)
# ============================================================
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

* {{
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}}

html, body, .stApp {{
    background: {COR_FUNDO} !important;
    font-family: 'Inter', sans-serif !important;
    color: {COR_TEXTO} !important;
}}

.block-container {{
    padding: 1.5rem 2rem !important;
    max-width: 1400px !important;
    margin: 0 auto !important;
}}

/* Header */
.main-header {{
    background: linear-gradient(135deg, {COR_PRIMARIA}, {COR_SECUNDARIA});
    border-radius: 20px;
    padding: 1.5rem 2rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}}

.main-header h1 {{
    font-size: 1.8rem;
    font-weight: 800;
    color: white;
    margin: 0;
}}

.main-header p {{
    color: rgba(255,255,255,0.9);
    font-size: 0.85rem;
    margin-top: 0.5rem;
}}

.header-badges {{
    display: flex;
    gap: 0.5rem;
    margin-top: 1rem;
    flex-wrap: wrap;
}}

.header-badge {{
    background: rgba(255,255,255,0.15);
    backdrop-filter: blur(10px);
    padding: 0.2rem 0.7rem;
    border-radius: 20px;
    font-size: 0.65rem;
    font-weight: 500;
    color: white;
}}

/* KPIs */
.kpi-grid {{ display: flex; gap: 1rem; margin-bottom: 2rem; flex-wrap: wrap; }}
.kpi-card {{
    flex: 1; min-width: 160px; background: {COR_CARD}; border-radius: 16px;
    padding: 1.2rem; border: 1px solid {COR_BORDA};
    border-bottom: 3px solid {COR_PRIMARIA};
    transition: transform 0.2s;
}}
.kpi-card:hover {{ transform: translateY(-3px); }}
.kpi-icon {{ font-size: 1.6rem; margin-bottom: 0.5rem; }}
.kpi-label {{ font-size: 0.65rem; text-transform: uppercase; color: {COR_SUBTEXTO}; font-weight: 600; letter-spacing: 0.5px; }}
.kpi-value {{ font-size: 1.6rem; font-weight: 800; color: {COR_TEXTO}; margin: 0.4rem 0; }}
.kpi-sub {{ font-size: 0.6rem; color: {COR_SUBTEXTO}; }}

/* Seções */
.section-header {{
    display: flex; align-items: center; justify-content: space-between;
    margin: 1.5rem 0 1rem 0; border-bottom: 2px solid {COR_BORDA}; padding-bottom: 0.5rem;
}}
.section-header h2 {{ font-size: 1.1rem; font-weight: 700; color: {COR_TEXTO}; margin: 0; }}
.section-badge {{
    background: {COR_PRIMARIA}; color: white; padding: 0.2rem 0.6rem;
    border-radius: 20px; font-size: 0.6rem; font-weight: 600;
}}

/* Insight cards */
.insight-card {{
    background: {COR_CARD}; border-radius: 14px; padding: 1rem;
    margin-bottom: 0.8rem; border-left: 4px solid {COR_DESTAQUE};
}}
.insight-title {{ font-size: 0.6rem; font-weight: 700; text-transform: uppercase; color: {COR_SECUNDARIA}; letter-spacing: 0.5px; }}
.insight-value {{ font-size: 1.2rem; font-weight: 800; color: {COR_TEXTO}; margin: 0.4rem 0; }}
.insight-text {{ font-size: 0.7rem; color: {COR_SUBTEXTO}; line-height: 1.4; }}

/* Tabelas - estilo profissional */
[data-testid="stDataFrame"] {{
    border-radius: 12px !important;
    overflow: auto !important;
}}

[data-testid="stDataFrame"] th {{
    background: {COR_TABELA_HEADER} !important;
    color: white !important;
    font-weight: 600 !important;
    padding: 10px 12px !important;
    font-size: 0.75rem !important;
    text-align: center !important;
}}

[data-testid="stDataFrame"] td {{
    background: {COR_TABELA_LINHA} !important;
    color: {COR_TEXTO} !important;
    border-bottom: 1px solid {COR_BORDA} !important;
    padding: 8px 12px !important;
    font-size: 0.7rem !important;
}}

[data-testid="stDataFrame"] tr:hover td {{
    background: {COR_TABELA_HOVER} !important;
}}

/* Botões */
.stDownloadButton button,
.stButton button {{
    background: linear-gradient(135deg, {COR_PRIMARIA}, {COR_SECUNDARIA}) !important;
    color: white !important;
    border: none !important;
    border-radius: 40px !important;
    padding: 0.5rem 1rem !important;
    font-weight: 600 !important;
    font-size: 0.75rem !important;
    transition: all 0.3s ease !important;
    width: 100% !important;
}}

.stDownloadButton button:hover,
.stButton button:hover {{
    transform: scale(1.02);
    opacity: 0.95;
}}

/* Sidebar */
section[data-testid="stSidebar"] {{
    background: {COR_CARD} !important;
    border-right: 1px solid {COR_BORDA} !important;
}}

section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stMultiSelect label {{
    font-weight: 600;
    color: {COR_TEXTO} !important;
    font-size: 0.7rem;
}}

/* Métricas de crescimento */
.metric-positive {{
    color: #10B981;
    background: rgba(16,185,129,0.1);
    padding: 2px 6px;
    border-radius: 20px;
    font-size: 0.65rem;
    font-weight: 600;
}}

.metric-negative {{
    color: #EF4444;
    background: rgba(239,68,68,0.1);
    padding: 2px 6px;
    border-radius: 20px;
    font-size: 0.65rem;
    font-weight: 600;
}}

/* Footer */
.footer {{
    text-align: center; padding: 1.5rem 0 1rem; margin-top: 1.5rem;
    border-top: 1px solid {COR_BORDA}; font-size: 0.65rem; color: {COR_SUBTEXTO};
}}
.footer-disclaimer {{ font-size: 0.6rem; margin-top: 0.3rem; }}
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

def normalizar_texto(texto):
    """Remove caracteres especiais e normaliza texto"""
    if pd.isna(texto):
        return texto
    texto = str(texto)
    texto = unicodedata.normalize('NFKD', texto).encode('ASCII', 'ignore').decode('ASCII')
    # Substituições manuais
    substituicoes = {
        'ECONOMICA': 'ECONÔMICA',
        'FEDERAL': 'FEDERAL',
        'CAIXA ECONOMICA FEDERAL': 'CAIXA ECONÔMICA FEDERAL'
    }
    for k, v in substituicoes.items():
        texto = texto.replace(k, v)
    return texto

def calcular_hhi(data):
    total = data['numero_operacoes'].sum()
    if total == 0:
        return 0
    participacoes = (data['numero_operacoes'] / total) ** 2
    hhi = participacoes.sum() * 10000
    return hhi

def classificar_hhi(hhi):
    if hhi < 1500:
        return "Competitivo", "🟢"
    elif hhi < 2500:
        return "Moderadamente concentrado", "🟡"
    else:
        return "Altamente concentrado", "🔴"

def classificar_banco(nome):
    nome_upper = str(nome).upper().strip()
    if any(digital in nome_upper for digital in ['NUBANK', 'INTER', 'C6 BANK']):
        return 'digital'
    elif any(tradicional in nome_upper for tradicional in ['ITAU', 'BRADESCO', 'SANTANDER', 'BB', 'CAIXA', 'BANCO DO BRASIL']):
        return 'tradicional'
    return 'outros'

@st.cache_data
def carregar_dados():
    """Carrega CSV com tratamento de erro robusto"""
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
                df['ano_mes'] = df['data_base'].dt.strftime('%Y-%m')
            
            if 'nome_conglomerado_financeiro' in df.columns:
                df['tipo_banco'] = df['nome_conglomerado_financeiro'].apply(classificar_banco)
            
            df = df.dropna(subset=['numero_operacoes', 'volume_operacoes'])
            
            # Tratamento de erro: se DataFrame vazio
            if len(df) == 0:
                return None, None
                
            return df, encoding
        except Exception as e:
            continue
    
    return None, None

# ============================================================
# CARREGAR DADOS COM TRATAMENTO DE ERRO
# ============================================================
with st.spinner("🔄 Carregando dados do Banco Central..."):
    df, encoding = carregar_dados()

if df is None:
    st.error("""
    ❌ **Erro ao carregar os dados**
    
    Verifique se o arquivo `dados_desenrola.csv` está no repositório.
    
    **Possíveis soluções:**
    1. Verifique se o arquivo existe na pasta do projeto
    2. Confirme se o nome do arquivo está correto
    3. Verifique o formato do CSV (separador ponto e vírgula)
    """)
    st.stop()

# ============================================================
# SIDEBAR (com toggle de tema e filtros)
# ============================================================
with st.sidebar:
    st.markdown("### 🎛️ Controles")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("☀️ Modo Claro", use_container_width=True):
            st.session_state.tema = 'claro'
            st.rerun()
    with col2:
        if st.button("🌙 Modo Escuro", use_container_width=True):
            st.session_state.tema = 'escuro'
            st.rerun()
    
    st.markdown("---")
    st.markdown("### 🎯 Filtros")
    
    # Filtro por Tipo
    if 'tipo_desenrola' in df.columns:
        tipos = sorted(df['tipo_desenrola'].dropna().unique())
        tipo_filtro = st.multiselect("Tipo do Programa", tipos, default=tipos)
        df_filtrado = df[df['tipo_desenrola'].isin(tipo_filtro)] if tipo_filtro else df.copy()
    else:
        df_filtrado = df.copy()
    
    # Filtro por UF
    if 'unidade_federacao' in df.columns:
        ufs = sorted(df['unidade_federacao'].dropna().unique())
        uf_filtro = st.multiselect("UF", ufs, default=[])
        if uf_filtro:
            df_filtrado = df_filtrado[df_filtrado['unidade_federacao'].isin(uf_filtro)]
    
    # Filtro por Banco
    if 'nome_conglomerado_financeiro' in df.columns:
        bancos = sorted(df['nome_conglomerado_financeiro'].dropna().unique())
        banco_filtro = st.multiselect("Instituição Financeira", bancos, default=[])
        if banco_filtro:
            df_filtrado = df_filtrado[df_filtrado['nome_conglomerado_financeiro'].isin(banco_filtro)]
    
    # Filtro por Período
    if 'data_base' in df.columns and not df['data_base'].isna().all():
        min_date = df['data_base'].min().date()
        max_date = df['data_base'].max().date()
        periodo_filtro = st.slider("Período", min_date, max_date, (min_date, max_date))
        df_filtrado = df_filtrado[(df_filtrado['data_base'].dt.date >= periodo_filtro[0]) & 
                                   (df_filtrado['data_base'].dt.date <= periodo_filtro[1])]
    
    st.markdown("---")
    st.markdown("### 📊 Tipos do programa")
    st.markdown("🔵 **Tipo 1:** Faixa 1 (Pessoa Física)")
    st.markdown("🟢 **Tipo 2:** Faixa 2 (Pessoa Física)")
    st.markdown("🟡 **Tipo 3:** Pequenos Negócios")

# ============================================================
# HEADER
# ============================================================
st.markdown(f"""
<div class="main-header">
    <h1>🏦 Desenrola Brasil</h1>
    <p>Programa de Renegociação de Dívidas - Dados oficiais do Banco Central do Brasil</p>
    <div class="header-badges">
        <span class="header-badge">🏛️ Fonte Oficial</span>
        <span class="header-badge">📊 Dados Abertos</span>
        <span class="header-badge">✅ SCR - Sistema de Informações de Crédito</span>
        <span class="header-badge">📋 Lei nº 14.690/2023</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# MAIN
# ============================================================
if df_filtrado is not None and len(df_filtrado) > 0:
    st.success(f"✅ {len(df_filtrado):,} registros processados com sucesso!")
    
    # ===== KPIs =====
    total_operacoes = df_filtrado['numero_operacoes'].sum()
    total_volume = df_filtrado['volume_operacoes'].sum()
    ticket_medio = total_volume / total_operacoes if total_operacoes > 0 else 0
    num_bancos = df_filtrado['nome_conglomerado_financeiro'].nunique() if 'nome_conglomerado_financeiro' in df_filtrado.columns else 0
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-icon">💰</div>
            <div class="kpi-label">VOLUME RENEGOCIADO</div>
            <div class="kpi-value">{fmt_brl(total_volume)}</div>
            <div class="kpi-sub">total de dívidas renegociadas</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-icon">📋</div>
            <div class="kpi-label">OPERAÇÕES</div>
            <div class="kpi-value">{fmt_num(total_operacoes)}</div>
            <div class="kpi-sub">renegociações realizadas</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-icon">🎫</div>
            <div class="kpi-label">TICKET MÉDIO</div>
            <div class="kpi-value">{fmt_brl(ticket_medio)}</div>
            <div class="kpi-sub">por operação</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-icon">🏛️</div>
            <div class="kpi-label">INSTITUIÇÕES</div>
            <div class="kpi-value">{fmt_num(num_bancos)}</div>
            <div class="kpi-sub">bancos participantes</div>
        </div>
        """, unsafe_allow_html=True)
    
    # ===== ANÁLISE HHI =====
    if 'nome_conglomerado_financeiro' in df_filtrado.columns:
        st.markdown(f"""
        <div class="section-header">
            <h2>📊 Concentração de Mercado (HHI)</h2>
            <span class="section-badge">Risco</span>
        </div>
        """, unsafe_allow_html=True)
        
        hhi_valor = calcular_hhi(df_filtrado)
        classificacao, cor_icon = classificar_hhi(hhi_valor)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="insight-card">
                <div class="insight-title">🏦 ÍNDICE HHI</div>
                <div class="insight-value">{hhi_valor:.0f}</div>
                <div class="insight-text">Índice Herfindahl-Hirschman - Mede concentração de mercado</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="insight-card">
                <div class="insight-title">📈 CLASSIFICAÇÃO</div>
                <div class="insight-value">{cor_icon} {classificacao}</div>
                <div class="insight-text">Mercado de renegociação de dívidas</div>
            </div>
            """, unsafe_allow_html=True)
    
    # ===== PARTICIPAÇÃO PERCENTUAL POR BANCO (NOVO) =====
    if 'nome_conglomerado_financeiro' in df_filtrado.columns:
        st.markdown(f"""
        <div class="section-header">
            <h2>📊 Participação por Instituição Financeira</h2>
            <span class="section-badge">Market Share</span>
        </div>
        """, unsafe_allow_html=True)
        
        participacao = df_filtrado.groupby('nome_conglomerado_financeiro')['numero_operacoes'].sum().sort_values(ascending=False).head(10).reset_index()
        participacao.columns = ['Instituição', 'Operações']
        total_part = participacao['Operações'].sum()
        participacao['Participação'] = participacao['Operações'].apply(lambda x: fmt_percentual(x, total_operacoes))
        participacao['Operações'] = participacao['Operações'].apply(fmt_num)
        
        # Gráfico de rosca
        fig_participacao = px.pie(participacao.head(7), names='Instituição', values='Operações'.replace('.', ''), 
                                   hole=0.4, title='Top 7 Instituições - Market Share',
                                   color_discrete_sequence=[COR_PRIMARIA, COR_SECUNDARIA, COR_DESTAQUE, COR_ALERTA, '#8B5CF6', '#EC4899', '#06B6D4'])
        fig_participacao.update_layout(template="plotly_white", height=450, paper_bgcolor=COR_CARD, font=dict(color=COR_TEXTO))
        st.plotly_chart(fig_participacao, use_container_width=True)
        
        # Tabela de participação
        st.dataframe(participacao, use_container_width=True, hide_index=True)
    
    # ===== BANCOS DIGITAIS VS TRADICIONAIS =====
    if 'tipo_banco' in df_filtrado.columns and len(df_filtrado['tipo_banco'].unique()) > 0:
        st.markdown(f"""
        <div class="section-header">
            <h2>📱 Bancos Digitais vs Tradicionais</h2>
            <span class="section-badge">Segmentação</span>
        </div>
        """, unsafe_allow_html=True)
        
        tipo_banco_data = df_filtrado.groupby('tipo_banco').agg({
            'numero_operacoes': 'sum',
            'volume_operacoes': 'sum'
        }).reset_index()
        tipo_banco_data['ticket_medio'] = tipo_banco_data['volume_operacoes'] / tipo_banco_data['numero_operacoes']
        
        col1, col2 = st.columns(2)
        with col1:
            fig_digital = px.bar(tipo_banco_data, x='tipo_banco', y='numero_operacoes',
                                  title='Operações por Tipo de Banco',
                                  color='tipo_banco',
                                  color_discrete_map={'digital': '#00A86B', 'tradicional': '#0052CC', 'outros': '#94A3B8'},
                                  text_auto='.0f')
            fig_digital.update_layout(template="plotly_white", height=400, paper_bgcolor=COR_CARD, font=dict(color=COR_TEXTO))
            st.plotly_chart(fig_digital, use_container_width=True)
        with col2:
            fig_ticket_digital = px.bar(tipo_banco_data, x='tipo_banco', y='ticket_medio',
                                         title='Ticket Médio por Tipo de Banco (R$)',
                                         color='tipo_banco',
                                         color_discrete_map={'digital': '#00A86B', 'tradicional': '#0052CC', 'outros': '#94A3B8'},
                                         text_auto='.2s')
            fig_ticket_digital.update_layout(template="plotly_white", height=400, paper_bgcolor=COR_CARD, font=dict(color=COR_TEXTO))
            st.plotly_chart(fig_ticket_digital, use_container_width=True)
    
    # ===== LIDERANÇA REGIONAL =====
    if 'unidade_federacao' in df_filtrado.columns and 'nome_conglomerado_financeiro' in df_filtrado.columns:
        st.markdown(f"""
        <div class="section-header">
            <h2>🗺️ Liderança Regional por Estado</h2>
            <span class="section-badge">Top 3 Bancos</span>
        </div>
        """, unsafe_allow_html=True)
        
        banco_por_uf = df_filtrado.groupby(['unidade_federacao', 'nome_conglomerado_financeiro'])['numero_operacoes'].sum().reset_index()
        banco_por_uf = banco_por_uf.sort_values(['unidade_federacao', 'numero_operacoes'], ascending=[True, False])
        top3_por_uf = banco_por_uf.groupby('unidade_federacao').head(3).reset_index(drop=True)
        top3_por_uf['ranking'] = top3_por_uf.groupby('unidade_federacao').cumcount() + 1
        top3_por_uf['nome_normalizado'] = top3_por_uf['nome_conglomerado_financeiro'].apply(normalizar_texto)
        top3_por_uf['exibicao'] = top3_por_uf.apply(lambda x: f"{x['ranking']}º - {x['nome_normalizado']} ({fmt_num(x['numero_operacoes'])})", axis=1)
        
        if len(top3_por_uf) > 0:
            lideranca = top3_por_uf[top3_por_uf['ranking'] == 1].groupby('nome_normalizado').size().sort_values(ascending=False)
            if len(lideranca) > 0:
                st.info(f"🏆 **{lideranca.index[0]}** lidera em {lideranca.iloc[0]} estados, seguido por **{lideranca.index[1] if len(lideranca) > 1 else 'N/A'}**")
            
            tabela_lideranca = top3_por_uf.pivot_table(index='unidade_federacao', columns='ranking', values='exibicao', aggfunc='first').reset_index()
            tabela_lideranca.columns = ['UF', '🥇 1º Lugar', '🥈 2º Lugar', '🥉 3º Lugar']
            
            st.dataframe(
                tabela_lideranca,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "UF": st.column_config.TextColumn("Estado", width="small"),
                    "🥇 1º Lugar": st.column_config.TextColumn("Líder", width="medium"),
                    "🥈 2º Lugar": st.column_config.TextColumn("Segundo", width="medium"),
                    "🥉 3º Lugar": st.column_config.TextColumn("Terceiro", width="medium"),
                }
            )
    
    # ===== TICKET MÉDIO POR ESTADO (NOVO) =====
    if 'unidade_federacao' in df_filtrado.columns:
        st.markdown(f"""
        <div class="section-header">
            <h2>🎫 Ticket Médio por Estado</h2>
            <span class="section-badge">Ranking</span>
        </div>
        """, unsafe_allow_html=True)
        
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
        fig_ticket_uf.update_layout(template="plotly_white", height=450, paper_bgcolor=COR_CARD, font=dict(color=COR_TEXTO))
        fig_ticket_uf.update_traces(textposition='outside')
        st.plotly_chart(fig_ticket_uf, use_container_width=True)
    
    # ===== EVOLUÇÃO MENSAL COM CRESCIMENTO =====
    if 'data_base' in df_filtrado.columns and len(df_filtrado['data_base'].dropna()) > 1:
        st.markdown(f"""
        <div class="section-header">
            <h2>📈 Evolução Mensal do Programa</h2>
            <span class="section-badge">Temporal</span>
        </div>
        """, unsafe_allow_html=True)
        
        evolucao = df_filtrado.groupby('data_base').agg({
            'numero_operacoes': 'sum',
            'volume_operacoes': 'sum'
        }).reset_index().sort_values('data_base')
        
        # Calcular crescimento
        evolucao['crescimento'] = evolucao['volume_operacoes'].pct_change() * 100
        
        fig_evolucao = px.line(evolucao, x='data_base', y='volume_operacoes', 
                               title='Volume Renegociado por Mês (R$)',
                               markers=True, line_shape='linear')
        fig_evolucao.update_layout(template="plotly_white", height=450,
                                   xaxis_title="Mês", yaxis_title="Volume (R$)",
                                   hovermode='x unified',
                                   paper_bgcolor=COR_CARD,
                                   font=dict(color=COR_TEXTO))
        fig_evolucao.update_traces(line=dict(width=3, color=COR_PRIMARIA), 
                                   marker=dict(size=6, color=COR_SECUNDARIA))
        
        # Marco regulatório
        if len(evolucao[evolucao['data_base'] >= pd.Timestamp('2024-04-01')]) > 0:
            fig_evolucao.add_annotation(
                x=pd.Timestamp('2024-04-01'),
                y=evolucao['volume_operacoes'].max() * 0.9,
                text="📅 MP 1.213/2024<br>Inclusão Pequenos Negócios",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor=COR_ALERTA,
                font=dict(size=9, color=COR_ALERTA),
                bgcolor=COR_CARD,
                bordercolor=COR_ALERTA,
                borderwidth=1
            )
        
        st.plotly_chart(fig_evolucao, use_container_width=True)
        
        # Tabela de crescimento mensal
        st.markdown("#### 📊 Crescimento Mensal (%)")
        crescimento_tabela = evolucao[['data_base', 'crescimento']].dropna().tail(6).copy()
        crescimento_tabela['data_base'] = crescimento_tabela['data_base'].dt.strftime('%b/%Y')
        crescimento_tabela['crescimento'] = crescimento_tabela['crescimento'].apply(lambda x: f"{x:+.1f}%")
        crescimento_tabela.columns = ['Mês', 'Variação']
        st.dataframe(crescimento_tabela, use_container_width=True, hide_index=True)
    
    # ===== EVOLUÇÃO POR TIPO (NOVO) =====
    if 'tipo_desenrola' in df_filtrado.columns and 'data_base' in df_filtrado.columns:
        st.markdown(f"""
        <div class="section-header">
            <h2>📊 Evolução por Tipo do Programa</h2>
            <span class="section-badge">Comparativo</span>
        </div>
        """, unsafe_allow_html=True)
        
        evolucao_tipo = df_filtrado.groupby(['data_base', 'tipo_desenrola'])['volume_operacoes'].sum().reset_index()
        evolucao_tipo = evolucao_tipo.sort_values('data_base')
        
        fig_evolucao_tipo = px.line(evolucao_tipo, x='data_base', y='volume_operacoes', 
                                     color='tipo_desenrola',
                                     title='Evolução do Volume por Tipo (R$)',
                                     labels={'tipo_desenrola': 'Tipo do Programa', 'volume_operacoes': 'Volume (R$)'},
                                     color_discrete_map={1: COR_PRIMARIA, 2: COR_SECUNDARIA, 3: COR_DESTAQUE})
        fig_evolucao_tipo.update_layout(template="plotly_white", height=450,
                                        xaxis_title="Mês", yaxis_title="Volume (R$)",
                                        paper_bgcolor=COR_CARD, font=dict(color=COR_TEXTO))
        st.plotly_chart(fig_evolucao_tipo, use_container_width=True)
    
    # ===== RANKING ESTADOS =====
    if 'unidade_federacao' in df_filtrado.columns:
        st.markdown(f"""
        <div class="section-header">
            <h2>🗺️ Ranking por Unidade da Federação</h2>
            <span class="section-badge">Top 10</span>
        </div>
        """, unsafe_allow_html=True)
        
        uf_data = df_filtrado.groupby('unidade_federacao').agg({
            'numero_operacoes': 'sum',
            'volume_operacoes': 'sum'
        }).reset_index()
        uf_data['ticket_medio'] = uf_data['volume_operacoes'] / uf_data['numero_operacoes']
        uf_data = uf_data.sort_values('numero_operacoes', ascending=False).head(10)
        
        fig_uf = px.bar(uf_data, x='unidade_federacao', y='numero_operacoes', 
                        color='numero_operacoes',
                        title='Operações por UF (Top 10)', 
                        color_continuous_scale='Blues')
        fig_uf.update_layout(template="plotly_white", height=450, paper_bgcolor=COR_CARD, font=dict(color=COR_TEXTO))
        st.plotly_chart(fig_uf, use_container_width=True)
    
    # ===== RANKING BANCOS =====
    if 'nome_conglomerado_financeiro' in df_filtrado.columns:
        st.markdown(f"""
        <div class="section-header">
            <h2>🏦 Top 10 Instituições Financeiras</h2>
            <span class="section-badge">Ranking</span>
        </div>
        """, unsafe_allow_html=True)
        
        banco_data = df_filtrado.groupby('nome_conglomerado_financeiro')['numero_operacoes'].sum().sort_values(ascending=False).head(10).reset_index()
        banco_data.columns = ['Instituição', 'Operações']
        
        fig_banco = px.bar(banco_data, x='Operações', y='Instituição', orientation='h',
                          title='Top 10 Instituições por Renegociações',
                          color='Operações',
                          color_continuous_scale='Viridis',
                          text='Operações')
        fig_banco.update_layout(template="plotly_white", height=500, paper_bgcolor=COR_CARD, font=dict(color=COR_TEXTO))
        fig_banco.update_traces(texttemplate='%{text:,}', textposition='outside')
        st.plotly_chart(fig_banco, use_container_width=True)
    
    # ===== DISPERSÃO =====
    if 'nome_conglomerado_financeiro' in df_filtrado.columns:
        st.markdown(f"""
        <div class="section-header">
            <h2>📊 Dispersão: Operações vs Ticket Médio</h2>
            <span class="section-badge">Portfólio</span>
        </div>
        """, unsafe_allow_html=True)
        
        dispersao_data = df_filtrado.groupby('nome_conglomerado_financeiro').agg({
            'numero_operacoes': 'sum',
            'volume_operacoes': 'sum'
        }).reset_index()
        dispersao_data['ticket_medio'] = dispersao_data['volume_operacoes'] / dispersao_data['numero_operacoes']
        dispersao_data = dispersao_data[dispersao_data['numero_operacoes'] > 1000]
        
        if len(dispersao_data) > 1:
            dispersao_data['tipo'] = dispersao_data['nome_conglomerado_financeiro'].apply(classificar_banco)
            
            fig_dispersao = px.scatter(dispersao_data, x='numero_operacoes', y='ticket_medio', 
                                       color='tipo', size='numero_operacoes', hover_name='nome_conglomerado_financeiro',
                                       title='Relação entre Volume de Operações e Ticket Médio',
                                       labels={'numero_operacoes': 'Operações', 'ticket_medio': 'Ticket Médio (R$)'},
                                       color_discrete_map={'digital': COR_PRIMARIA, 'tradicional': COR_SECUNDARIA, 'outros': '#94A3B8'})
            fig_dispersao.update_layout(template="plotly_white", height=500, paper_bgcolor=COR_CARD, font=dict(color=COR_TEXTO))
            st.plotly_chart(fig_dispersao, use_container_width=True)
            
            correlacao = np.corrcoef(dispersao_data['numero_operacoes'].values, 
                                      dispersao_data['ticket_medio'].values)[0, 1]
            st.caption(f"📊 Correlação: **{correlacao:.2f}** - {'Correlação positiva' if correlacao > 0.3 else 'Correlação negativa' if correlacao < -0.3 else 'Baixa correlação'}")
        else:
            st.info("Dados insuficientes para análise de dispersão")
    
    # ===== CONCLUSÃO =====
    st.markdown(f"""
    <div class="section-header">
        <h2>📈 Conclusão</h2>
        <span class="section-badge">Resumo Executivo</span>
    </div>
    """, unsafe_allow_html=True)
    
    banco_lider = df_filtrado.groupby('nome_conglomerado_financeiro')['numero_operacoes'].sum().idxmax() if 'nome_conglomerado_financeiro' in df_filtrado.columns and len(df_filtrado['nome_conglomerado_financeiro'].unique()) > 0 else "N/A"
    uf_lider = df_filtrado.groupby('unidade_federacao')['volume_operacoes'].sum().idxmax() if 'unidade_federacao' in df_filtrado.columns and len(df_filtrado['unidade_federacao'].unique()) > 0 else "N/A"
    
    st.markdown(f"""
    <div class="insight-card" style="background: linear-gradient(135deg, {COR_PRIMARIA}, {COR_SECUNDARIA}); color: white; border: none;">
        <div class="insight-text" style="color: rgba(255,255,255,0.95);">
        O <strong>Programa Desenrola Brasil</strong> já renegociou <strong>{fmt_brl(total_volume)}</strong> em dívidas, 
        com <strong>{fmt_num(total_operacoes)} operações</strong>. <strong>{banco_lider}</strong> lidera as renegociações,
        e <strong>{uf_lider}</strong> concentra a maior parte do volume financeiro.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # ===== EXPORTAÇÃO =====
    st.markdown("---")
    st.markdown("### 📥 Exportar Dados")
    
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        relatorio = f"""RELATÓRIO DESENROLA BRASIL
Data: {datetime.now().strftime('%d/%m/%Y %H:%M')}

VISÃO GERAL:
- Total Renegociado: {fmt_brl(total_volume)}
- Total de Operações: {fmt_num(total_operacoes)}
- Ticket Médio: {fmt_brl(ticket_medio)}
- Banco Líder: {banco_lider}
- Estado Líder: {uf_lider}

Fonte: Banco Central do Brasil (SCR/Desenrola)
"""
        st.download_button("📝 Baixar Relatório (TXT)", relatorio, f"relatorio_desenrola_{datetime.now().strftime('%Y%m%d')}.txt", "text/plain")
    
    with col2:
        csv = df_filtrado.to_csv(index=False).encode('utf-8')
        st.download_button("📊 Baixar Dados (CSV)", csv, f"dados_desenrola_{datetime.now().strftime('%Y%m%d')}.csv", "text/csv")
    
    # ===== FOOTER =====
    st.markdown(f"""
    <div class="footer">
        🏦 Desenrola Brasil · Fonte: Banco Central do Brasil (SCR)<br>
        Dashboard desenvolvido para portfólio de Análise de Dados
        <div class="footer-disclaimer">
            ⚠️ Dados de domínio público. Dashboard para fins de portfólio.
            <br>🔗 Fonte original: www.bcb.gov.br/estatisticas/scr
        </div>
    </div>
    """, unsafe_allow_html=True)

else:
    st.warning("⚠️ Nenhum dado encontrado com os filtros selecionados. Ajuste os filtros para visualizar os dados.")
