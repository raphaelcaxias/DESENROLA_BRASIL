import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import unicodedata

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
else:
    COR_PRIMARIA = "#00C878"
    COR_SECUNDARIA = "#3B82F6"
    COR_DESTAQUE = "#FBBF24"
    COR_ALERTA = "#FF8C42"
    COR_FUNDO = "#0F172A"
    COR_CARD = "#1E293B"
    COR_TEXTO = "#F1F5F9"
    COR_SUBTEXTO = "#94A3B8"
    COR_BORDA = "#334155"

# ============================================================
# CSS
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
    padding: 1.8rem 2rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}}

.main-header h1 {{
    font-size: 2rem;
    font-weight: 800;
    color: white;
    margin: 0;
}}

.main-header p {{
    color: rgba(255,255,255,0.9);
    font-size: 0.9rem;
    margin-top: 0.5rem;
}}

.header-badges {{
    display: flex;
    gap: 0.75rem;
    margin-top: 1rem;
    flex-wrap: wrap;
}}

.header-badge {{
    background: rgba(255,255,255,0.15);
    backdrop-filter: blur(10px);
    padding: 0.25rem 0.75rem;
    border-radius: 20px;
    font-size: 0.7rem;
    font-weight: 500;
    color: white;
}}

/* KPIs */
.kpi-grid {{ display: flex; gap: 1.25rem; margin-bottom: 2rem; flex-wrap: wrap; }}
.kpi-card {{
    flex: 1; min-width: 180px; background: {COR_CARD}; border-radius: 16px;
    padding: 1.25rem; border: 1px solid {COR_BORDA};
    border-bottom: 3px solid {COR_PRIMARIA};
    transition: transform 0.2s;
}}
.kpi-card:hover {{ transform: translateY(-3px); }}
.kpi-icon {{ font-size: 1.8rem; margin-bottom: 0.5rem; }}
.kpi-label {{ font-size: 0.7rem; text-transform: uppercase; color: {COR_SUBTEXTO}; font-weight: 600; }}
.kpi-value {{ font-size: 1.8rem; font-weight: 800; color: {COR_TEXTO}; margin: 0.5rem 0; }}
.kpi-sub {{ font-size: 0.65rem; color: {COR_SUBTEXTO}; }}

/* Seções */
.section-header {{
    display: flex; align-items: center; justify-content: space-between;
    margin: 2rem 0 1rem 0; border-bottom: 2px solid {COR_BORDA}; padding-bottom: 0.5rem;
}}
.section-header h2 {{ font-size: 1.2rem; font-weight: 700; color: {COR_TEXTO}; margin: 0; }}
.section-badge {{
    background: {COR_PRIMARIA}; color: white; padding: 0.2rem 0.7rem;
    border-radius: 20px; font-size: 0.65rem; font-weight: 600;
}}

/* Insight cards */
.insight-card {{
    background: {COR_CARD}; border-radius: 16px; padding: 1.25rem;
    margin-bottom: 1rem; border-left: 4px solid {COR_DESTAQUE};
}}
.insight-title {{ font-size: 0.65rem; font-weight: 700; text-transform: uppercase; color: {COR_SECUNDARIA}; }}
.insight-value {{ font-size: 1.3rem; font-weight: 800; color: {COR_TEXTO}; margin: 0.5rem 0; }}
.insight-text {{ font-size: 0.75rem; color: {COR_SUBTEXTO}; }}

/* Sidebar */
section[data-testid="stSidebar"] {{
    background: {COR_CARD} !important;
    border-right: 1px solid {COR_BORDA} !important;
}}

/* DataFrame */
[data-testid="stDataFrame"] th {{
    background: {COR_SECUNDARIA} !important;
    color: white !important;
    font-weight: 600 !important;
    padding: 10px !important;
}}

[data-testid="stDataFrame"] td {{
    background: {COR_CARD} !important;
    color: {COR_TEXTO} !important;
    border-bottom: 1px solid {COR_BORDA} !important;
    padding: 8px !important;
}}

[data-testid="stDataFrame"] tr:hover td {{
    background: {COR_BORDA} !important;
}}

/* Footer */
.footer {{
    text-align: center; padding: 2rem 0 1rem; margin-top: 2rem;
    border-top: 1px solid {COR_BORDA}; font-size: 0.7rem; color: {COR_SUBTEXTO};
}}
.footer-disclaimer {{ font-size: 0.65rem; margin-top: 0.5rem; }}
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

def normalizar_texto(texto):
    if pd.isna(texto):
        return texto
    texto = str(texto)
    texto = unicodedata.normalize('NFKD', texto).encode('ASCII', 'ignore').decode('ASCII')
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
            return df, encoding
        except:
            continue
    return None, None

# ============================================================
# CARREGAR DADOS PRIMEIRO
# ============================================================
df, encoding = carregar_dados()

# ============================================================
# SIDEBAR (com toggle de tema)
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
    
    # Filtros - só depois que os dados estão carregados
    if df is not None:
        st.markdown("### 🎯 Filtros")
        
        if 'tipo_desenrola' in df.columns:
            tipos = sorted(df['tipo_desenrola'].dropna().unique())
            tipo_filtro = st.multiselect("Tipo do Programa", tipos, default=tipos)
            df_filtrado = df[df['tipo_desenrola'].isin(tipo_filtro)] if tipo_filtro else df.copy()
        else:
            df_filtrado = df.copy()
        
        if 'unidade_federacao' in df.columns:
            ufs = sorted(df['unidade_federacao'].dropna().unique())
            uf_filtro = st.multiselect("UF", ufs, default=[])
            if uf_filtro:
                df_filtrado = df_filtrado[df_filtrado['unidade_federacao'].isin(uf_filtro)]
        
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
        st.markdown("🔵 **Tipo 1:** Faixa 1 (Pessoa Física)")
        st.markdown("🟢 **Tipo 2:** Faixa 2 (Pessoa Física)")
        st.markdown("🟡 **Tipo 3:** Pequenos Negócios")
    else:
        df_filtrado = None
        st.error("❌ Erro ao carregar dados")

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
        <span class="header-badge">📋 Lei nº 14.690/2023</span>
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
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-icon">💰</div>
            <div class="kpi-label">VOLUME RENEGOCIADO</div>
            <div class="kpi-value">{fmt_brl(total_volume)}</div>
            <div class="kpi-sub">total de dívidas</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-icon">📋</div>
            <div class="kpi-label">OPERAÇÕES</div>
            <div class="kpi-value">{fmt_num(total_operacoes)}</div>
            <div class="kpi-sub">renegociações</div>
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
            <div class="kpi-sub">bancos</div>
        </div>
        """, unsafe_allow_html=True)
    
    # ===== HHI =====
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
                <div class="insight-text">Índice Herfindahl-Hirschman</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="insight-card">
                <div class="insight-title">📈 CLASSIFICAÇÃO</div>
                <div class="insight-value">{cor_icon} {classificacao}</div>
                <div class="insight-text">Mercado de renegociação</div>
            </div>
            """, unsafe_allow_html=True)
    
    # ===== LIDERANÇA REGIONAL (CORRIGIDA) =====
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
                st.info(f"🏆 **{lideranca.index[0]}** lidera em {lideranca.iloc[0]} estados")
            
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
    
    # ===== EVOLUÇÃO MENSAL =====
    if 'data_base' in df_filtrado.columns and len(df_filtrado['data_base'].dropna()) > 1:
        st.markdown(f"""
        <div class="section-header">
            <h2>📈 Evolução Mensal</h2>
            <span class="section-badge">Temporal</span>
        </div>
        """, unsafe_allow_html=True)
        
        evolucao = df_filtrado.groupby('data_base')['volume_operacoes'].sum().reset_index().sort_values('data_base')
        
        fig_evolucao = px.line(evolucao, x='data_base', y='volume_operacoes', 
                               title='Volume Renegociado por Mês',
                               markers=True, line_shape='linear')
        fig_evolucao.update_layout(template="plotly_white", height=400,
                                   paper_bgcolor=COR_CARD, font=dict(color=COR_TEXTO))
        fig_evolucao.update_traces(line=dict(width=3, color=COR_PRIMARIA), marker=dict(size=6))
        st.plotly_chart(fig_evolucao, use_container_width=True)
    
    # ===== CONCLUSÃO =====
    st.markdown(f"""
    <div class="section-header">
        <h2>📈 Conclusão</h2>
        <span class="section-badge">Resumo</span>
    </div>
    """, unsafe_allow_html=True)
    
    banco_lider = df_filtrado.groupby('nome_conglomerado_financeiro')['numero_operacoes'].sum().idxmax() if 'nome_conglomerado_financeiro' in df_filtrado.columns else "N/A"
    uf_lider = df_filtrado.groupby('unidade_federacao')['volume_operacoes'].sum().idxmax() if 'unidade_federacao' in df_filtrado.columns else "N/A"
    
    st.markdown(f"""
    <div class="insight-card" style="background: linear-gradient(135deg, {COR_PRIMARIA}, {COR_SECUNDARIA}); color: white; border: none;">
        <div class="insight-text" style="color: rgba(255,255,255,0.95);">
        O Programa já renegociou <strong>{fmt_brl(total_volume)}</strong> em dívidas, 
        com <strong>{fmt_num(total_operacoes)} operações</strong>. <strong>{banco_lider}</strong> lidera as renegociações.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # ===== FOOTER =====
    st.markdown(f"""
    <div class="footer">
        🏦 Desenrola Brasil · Fonte: Banco Central do Brasil (SCR)<br>
        Dashboard para portfólio de Análise de Dados
        <div class="footer-disclaimer">⚠️ Dados de domínio público. Dashboard para fins de portfólio.</div>
    </div>
    """, unsafe_allow_html=True)

else:
    st.error("❌ Erro ao carregar os dados. Verifique o arquivo 'dados_desenrola.csv'")
