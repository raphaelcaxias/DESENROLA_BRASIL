import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# ============================================================
# CONFIGURAÇÃO
# ============================================================
st.set_page_config(
    page_title="Desenrola Brasil - Análise de Renegociação de Dívidas",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CORES BANCÁRIAS
# ============================================================
COR_PRIMARIA = "#0A2540"
COR_SECUNDARIA = "#0066CC"
COR_DESTAQUE = "#00A86B"
COR_ALERTA = "#FF6B35"
COR_FUNDO = "#F8FAFC"
COR_TEXTO = "#1E293B"
COR_SUBTEXTO = "#64748B"

# Mapeamento de bancos digitais vs tradicionais
BANCOS_DIGITAIS = ['NUBANK', 'INTER', 'C6 BANK', 'NUBANK - PRUDENCIAL', 'INTER - PRUDENCIAL', 'C6 BANK - PRUDENCIAL']
BANCOS_TRADICIONAIS = ['ITAU', 'BRADESCO', 'SANTANDER', 'BB', 'CAIXA ECONÔMICA FEDERAL', 'BANCO DO BRASIL', 'BRADESCO - PRUDENCIAL', 'SANTANDER - PRUDENCIAL', 'ITAU - PRUDENCIAL', 'CAIXA ECONÔMICA FEDERAL - PRUDENCIAL', 'BB - PRUDENCIAL']

# ============================================================
# CSS
# ============================================================
st.markdown(f"""
<style>
.stApp {{ background: {COR_FUNDO} !important; }}
.block-container {{ padding: 2rem 2.5rem !important; max-width: 1400px !important; }}

.page-header {{
    background: linear-gradient(135deg, {COR_PRIMARIA}, {COR_SECUNDARIA});
    padding: 1.8rem 2rem;
    border-radius: 16px;
    margin-bottom: 24px;
}}
.page-header h1 {{ font-size: 32px; font-weight: 700; color: white; margin: 0 0 8px 0; }}
.page-header p {{ color: rgba(255,255,255,0.85); font-size: 14px; margin: 0; }}
.header-badges {{ display: flex; gap: 12px; margin-top: 16px; flex-wrap: wrap; }}
.header-badge {{ background: rgba(255,255,255,0.15); padding: 4px 12px; border-radius: 20px; font-size: 12px; color: white; font-weight: 500; }}

.kpi-grid {{ display: flex; gap: 20px; margin-bottom: 32px; flex-wrap: wrap; }}
.kpi-card {{
    flex: 1; min-width: 180px; background: white; border-radius: 12px;
    padding: 24px 20px; text-align: center; border: 1px solid #E2E8F0;
    border-bottom: 3px solid {COR_SECUNDARIA};
    transition: transform 0.2s;
}}
.kpi-card:hover {{ transform: translateY(-3px); box-shadow: 0 4px 12px rgba(0,0,0,0.1); }}
.kpi-label {{ font-size: 12px; text-transform: uppercase; color: {COR_SUBTEXTO}; font-weight: 600; }}
.kpi-value {{ font-size: 32px; font-weight: 800; color: {COR_PRIMARIA}; margin: 8px 0; }}
.kpi-sub {{ font-size: 11px; color: {COR_SUBTEXTO}; }}

.info-card {{
    background: white; border-radius: 12px; padding: 16px; border: 1px solid #E2E8F0; border-left: 4px solid {COR_DESTAQUE};
}}
.insight-card {{ background: white; border-radius: 12px; padding: 20px; margin-bottom: 20px; border: 1px solid #E2E8F0; border-left: 4px solid {COR_DESTAQUE}; }}
.insight-title {{ font-size: 13px; font-weight: 700; color: {COR_SECUNDARIA}; text-transform: uppercase; }}
.insight-value {{ font-size: 24px; font-weight: 800; color: {COR_PRIMARIA}; margin: 10px 0; }}
.insight-text {{ font-size: 13px; color: {COR_SUBTEXTO}; }}

.section-header {{ display: flex; align-items: center; justify-content: space-between; margin: 32px 0 20px 0; border-bottom: 2px solid #E2E8F0; padding-bottom: 12px; }}
.section-header h2 {{ font-size: 20px; font-weight: 700; color: {COR_PRIMARIA}; margin: 0; }}
.section-tag {{ background: #E8F4FD; padding: 4px 12px; border-radius: 20px; font-size: 11px; color: {COR_SECUNDARIA}; font-weight: 600; }}

.footer {{ text-align: center; padding: 30px 0 20px; margin-top: 40px; border-top: 1px solid #E2E8F0; color: {COR_SUBTEXTO}; font-size: 12px; }}
.footer-disclaimer {{ font-size: 11px; color: #94A3B8; margin-top: 8px; }}
</style>
""", unsafe_allow_html=True)

# ============================================================
# HEADER
# ============================================================
st.markdown(f"""
<div class="page-header">
    <h1>🏦 Desenrola Brasil</h1>
    <p>Programa de Renegociação de Dívidas - Análise de dados oficiais do Banco Central do Brasil</p>
    <div class="header-badges">
        <span class="header-badge">🏛️ Fonte Oficial</span>
        <span class="header-badge">📊 Dados Abertos</span>
        <span class="header-badge">✅ SCR - Sistema de Informações de Crédito</span>
        <span class="header-badge">📋 Lei nº 14.690/2023</span>
    </div>
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

def calcular_hhi(data):
    """Calcula o Índice Herfindahl-Hirschman para concentração de mercado"""
    total = data['numero_operacoes'].sum()
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
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("### 🎯 Filtros")
    
    df, encoding = carregar_dados()
    
    if df is not None:
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
        st.markdown("🔵 Tipo 1: Faixa 1 (PF)\n🟢 Tipo 2: Faixa 2 (PF)\n🟡 Tipo 3: Pequenos Negócios")

# ============================================================
# MAIN
# ============================================================
if df_filtrado is not None and len(df_filtrado) > 0:
    st.success(f"✅ {len(df_filtrado):,} registros processados")
    
    # ===== KPIs =====
    total_operacoes = df_filtrado['numero_operacoes'].sum()
    total_volume = df_filtrado['volume_operacoes'].sum()
    ticket_medio = total_volume / total_operacoes if total_operacoes > 0 else 0
    
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
        num_bancos = df_filtrado['nome_conglomerado_financeiro'].nunique() if 'nome_conglomerado_financeiro' in df_filtrado.columns else 0
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">🏛️ INSTITUIÇÕES</div>
            <div class="kpi-value">{fmt_num(num_bancos)}</div>
            <div class="kpi-sub">bancos participantes</div>
        </div>
        """, unsafe_allow_html=True)
    
    # ===== ANÁLISE 1: HHI =====
    st.markdown("""
    <div class="section-header">
        <h2>📊 Análise de Concentração de Mercado (HHI)</h2>
        <span class="section-tag">Risco e Concorrência</span>
    </div>
    """, unsafe_allow_html=True)
    
    if 'nome_conglomerado_financeiro' in df_filtrado.columns:
        hhi_valor = calcular_hhi(df_filtrado)
        classificacao, cor = classificar_hhi(hhi_valor)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="info-card">
                <div class="insight-title">🏦 ÍNDICE HHI</div>
                <div class="insight-value">{hhi_valor:.0f}</div>
                <div class="insight-text">Índice Herfindahl-Hirschman - Mede concentração de mercado</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="info-card">
                <div class="insight-title">📈 CLASSIFICAÇÃO</div>
                <div class="insight-value">{cor} {classificacao}</div>
                <div class="insight-text">{'Mercado competitivo' if hhi_valor < 1500 else 'Alto risco de concentração' if hhi_valor > 2500 else 'Concentração moderada'}</div>
            </div>
            """, unsafe_allow_html=True)
    
    # ===== ANÁLISE 2: Digitais vs Tradicionais =====
    if 'tipo_banco' in df_filtrado.columns:
        st.markdown("""
        <div class="section-header">
            <h2>📱 Bancos Digitais vs Tradicionais</h2>
            <span class="section-tag">Segmentação de Mercado</span>
        </div>
        """, unsafe_allow_html=True)
        
        tipo_banco_data = df_filtrado.groupby('tipo_banco').agg({
            'numero_operacoes': 'sum',
            'volume_operacoes': 'sum'
        }).reset_index()
        tipo_banco_data['ticket_medio'] = tipo_banco_data['volume_operacoes'] / tipo_banco_data['numero_operacoes']
        
        col1, col2 = st.columns(2)
        with col1:
            fig_tipo_bar = px.bar(tipo_banco_data, x='tipo_banco', y='numero_operacoes',
                                  title='Operações por Tipo de Banco',
                                  color='tipo_banco', color_discrete_map={'digital': '#00A86B', 'tradicional': '#0066CC', 'outros': '#94A3B8'},
                                  text_auto='.0f')
            fig_tipo_bar.update_layout(template="plotly_white", height=400)
            st.plotly_chart(fig_tipo_bar, use_container_width=True)
        with col2:
            fig_tipo_ticket = px.bar(tipo_banco_data, x='tipo_banco', y='ticket_medio',
                                     title='Ticket Médio por Tipo de Banco (R$)',
                                     color='tipo_banco', color_discrete_map={'digital': '#00A86B', 'tradicional': '#0066CC', 'outros': '#94A3B8'},
                                     text_auto='.2s')
            fig_tipo_ticket.update_layout(template="plotly_white", height=400)
            st.plotly_chart(fig_tipo_ticket, use_container_width=True)
    
    # ===== ANÁLISE 3: Top 3 Bancos por Estado =====
    if 'unidade_federacao' in df_filtrado.columns and 'nome_conglomerado_financeiro' in df_filtrado.columns:
        st.markdown("""
        <div class="section-header">
            <h2>🗺️ Liderança Regional por Estado</h2>
            <span class="section-tag">Top 3 Bancos por UF</span>
        </div>
        """, unsafe_allow_html=True)
        
        banco_por_uf = df_filtrado.groupby(['unidade_federacao', 'nome_conglomerado_financeiro'])['numero_operacoes'].sum().reset_index()
        banco_por_uf = banco_por_uf.sort_values(['unidade_federacao', 'numero_operacoes'], ascending=[True, False])
        top3_por_uf = banco_por_uf.groupby('unidade_federacao').head(3).reset_index(drop=True)
        top3_por_uf['ranking'] = top3_por_uf.groupby('unidade_federacao').cumcount() + 1
        top3_por_uf['exibicao'] = top3_por_uf.apply(lambda x: f"{x['ranking']}º - {x['nome_conglomerado_financeiro']} ({fmt_num(x['numero_operacoes'])})", axis=1)
        
        lideranca = top3_por_uf[top3_por_uf['ranking'] == 1].groupby('nome_conglomerado_financeiro').size().sort_values(ascending=False)
        st.info(f"🏆 **{lideranca.index[0]}** lidera em {lideranca.iloc[0]} estados, seguido por **{lideranca.index[1] if len(lideranca) > 1 else 'N/A'}** com {lideranca.iloc[1] if len(lideranca) > 1 else 0} estados")
        
        tabela_lideranca = top3_por_uf.pivot_table(index='unidade_federacao', columns='ranking', values='exibicao', aggfunc='first').reset_index()
        st.dataframe(tabela_lideranca, use_container_width=True, hide_index=True)
    
    # ===== ANÁLISE 4: Evolução com Marco Regulatório =====
    if 'data_base' in df_filtrado.columns:
        st.markdown("""
        <div class="section-header">
            <h2>📈 Evolução Mensal do Programa</h2>
            <span class="section-tag">Série Temporal</span>
        </div>
        """, unsafe_allow_html=True)
        
        evolucao = df_filtrado.groupby('data_base').agg({
            'numero_operacoes': 'sum',
            'volume_operacoes': 'sum'
        }).reset_index().sort_values('data_base')
        
        fig_evolucao = go.Figure()
        fig_evolucao.add_trace(go.Scatter(x=evolucao['data_base'], y=evolucao['volume_operacoes'],
                                          mode='lines+markers', name='Volume',
                                          line=dict(width=3, color=COR_SECUNDARIA),
                                          marker=dict(size=6, color=COR_PRIMARIA)))
        fig_evolucao.add_vline(x=pd.Timestamp('2024-04-01'), line_dash="dash", line_color=COR_ALERTA,
                               annotation_text="MP 1.213/2024 - Inclusão Pequenos Negócios", annotation_position="top")
        fig_evolucao.update_layout(template="plotly_white", height=450,
                                   xaxis_title="Mês", yaxis_title="Volume (R$)",
                                   title="Volume Renegociado por Mês com Marco Regulatório")
        st.plotly_chart(fig_evolucao, use_container_width=True)
    
    # ===== ANÁLISE 5: Gráfico de Dispersão =====
    if 'nome_conglomerado_financeiro' in df_filtrado.columns:
        st.markdown("""
        <div class="section-header">
            <h2>📊 Dispersão: Operações vs Ticket Médio</h2>
            <span class="section-tag">Análise de Portfólio</span>
        </div>
        """, unsafe_allow_html=True)
        
        dispersao_data = df_filtrado.groupby('nome_conglomerado_financeiro').agg({
            'numero_operacoes': 'sum',
            'volume_operacoes': 'sum'
        }).reset_index()
        dispersao_data['ticket_medio'] = dispersao_data['volume_operacoes'] / dispersao_data['numero_operacoes']
        dispersao_data = dispersao_data[dispersao_data['numero_operacoes'] > 1000]
        dispersao_data['tipo'] = dispersao_data['nome_conglomerado_financeiro'].apply(classificar_banco)
        
        fig_dispersao = px.scatter(dispersao_data, x='numero_operacoes', y='ticket_medio', 
                                   color='tipo', size='numero_operacoes', hover_name='nome_conglomerado_financeiro',
                                   title='Relação entre Volume de Operações e Ticket Médio',
                                   labels={'numero_operacoes': 'Operações', 'ticket_medio': 'Ticket Médio (R$)'},
                                   color_discrete_map={'digital': COR_DESTAQUE, 'tradicional': COR_SECUNDARIA, 'outros': '#94A3B8'})
        fig_dispersao.update_layout(template="plotly_white", height=500)
        st.plotly_chart(fig_dispersao, use_container_width=True)
        
        if len(dispersao_data) > 1:
            x = dispersao_data['numero_operacoes'].values
            y = dispersao_data['ticket_medio'].values
            correlacao = np.corrcoef(x, y)[0, 1] if len(x) > 1 else 0
            st.caption(f"📊 Correlação entre Operações e Ticket Médio: **{correlacao:.2f}** - {'Correlação positiva' if correlacao > 0.3 else 'Correlação negativa' if correlacao < -0.3 else 'Baixa correlação'}")
    
    # ===== GRÁFICO: Por Tipo =====
    if 'tipo_desenrola' in df_filtrado.columns:
        st.markdown("""
        <div class="section-header">
            <h2>📊 Análise por Tipo do Programa</h2>
            <span class="section-tag">Distribuição</span>
        </div>
        """, unsafe_allow_html=True)
        
        tipo_data = df_filtrado.groupby('tipo_desenrola').agg({'numero_operacoes': 'sum', 'volume_operacoes': 'sum'}).reset_index()
        tipo_data['ticket_medio'] = tipo_data['volume_operacoes'] / tipo_data['numero_operacoes']
        
        col1, col2 = st.columns(2)
        with col1:
            fig_pie = px.pie(tipo_data, names='tipo_desenrola', values='numero_operacoes', hole=0.4,
                            title='Distribuição de Operações', color_discrete_sequence=[COR_PRIMARIA, COR_SECUNDARIA, COR_DESTAQUE])
            fig_pie.update_layout(template="plotly_white", height=420)
            st.plotly_chart(fig_pie, use_container_width=True)
        with col2:
            fig_ticket = px.bar(tipo_data, x='tipo_desenrola', y='ticket_medio',
                               title='Ticket Médio por Tipo (R$)', color='tipo_desenrola',
                               color_discrete_sequence=[COR_PRIMARIA, COR_SECUNDARIA, COR_DESTAQUE], text_auto='.2s')
            fig_ticket.update_layout(template="plotly_white", height=420)
            st.plotly_chart(fig_ticket, use_container_width=True)
    
    # ===== GRÁFICO: Top Estados =====
    if 'unidade_federacao' in df_filtrado.columns:
        st.markdown("""
        <div class="section-header">
            <h2>🗺️ Ranking por Unidade da Federação</h2>
            <span class="section-tag">Top 10</span>
        </div>
        """, unsafe_allow_html=True)
        
        uf_data = df_filtrado.groupby('unidade_federacao').agg({'numero_operacoes': 'sum', 'volume_operacoes': 'sum'}).reset_index()
        uf_data['ticket_medio'] = uf_data['volume_operacoes'] / uf_data['numero_operacoes']
        uf_data = uf_data.sort_values('numero_operacoes', ascending=False).head(10)
        
        fig_uf = px.bar(uf_data, x='unidade_federacao', y='numero_operacoes', color='numero_operacoes',
                       title='Operações por UF', color_continuous_scale='Blues')
        fig_uf.update_layout(template="plotly_white", height=450)
        st.plotly_chart(fig_uf, use_container_width=True)
    
    # ===== GRÁFICO: Top Bancos =====
    if 'nome_conglomerado_financeiro' in df_filtrado.columns:
        st.markdown("""
        <div class="section-header">
            <h2>🏦 Top 10 Instituições Financeiras</h2>
            <span class="section-tag">Ranking</span>
        </div>
        """, unsafe_allow_html=True)
        
        banco_data = df_filtrado.groupby('nome_conglomerado_financeiro')['numero_operacoes'].sum().sort_values(ascending=False).head(10).reset_index()
        banco_data.columns = ['Instituição', 'Operações']
        fig_banco = px.bar(banco_data, x='Operações', y='Instituição', orientation='h',
                          title='Top 10 Instituições por Renegociações', color='Operações',
                          color_continuous_scale='Viridis', text='Operações')
        fig_banco.update_layout(template="plotly_white", height=500)
        fig_banco.update_traces(texttemplate='%{text:,}', textposition='outside')
        st.plotly_chart(fig_banco, use_container_width=True)
    
    # ===== CONCLUSÃO =====
    st.markdown("""
    <div class="section-header">
        <h2>📈 Conclusão</h2>
        <span class="section-tag">Resumo Executivo</span>
    </div>
    """, unsafe_allow_html=True)
    
    banco_lider = df_filtrado.groupby('nome_conglomerado_financeiro')['numero_operacoes'].sum().idxmax() if 'nome_conglomerado_financeiro' in df_filtrado.columns else "N/A"
    uf_lider = df_filtrado.groupby('unidade_federacao')['volume_operacoes'].sum().idxmax() if 'unidade_federacao' in df_filtrado.columns else "N/A"
    
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
    
    relatorio = f"""RELATÓRIO DESENROLA BRASIL
Data: {datetime.now().strftime('%d/%m/%Y %H:%M')}

VISÃO GERAL:
- Total Renegociado: {fmt_brl(total_volume)}
- Total de Operações: {fmt_num(total_operacoes)}
- Ticket Médio: {fmt_brl(ticket_medio)}
- Banco Líder: {banco_lider}
- Estado Líder: {uf_lider}

ANÁLISE DE CONCENTRAÇÃO (HHI):
- Índice HHI: {hhi_valor:.0f}
- Classificação: {classificacao}

Fonte: Banco Central do Brasil (SCR/Desenrola)
"""
    
    st.download_button("📝 Baixar Relatório", relatorio, f"relatorio_desenrola_{datetime.now().strftime('%Y%m%d')}.txt", "text/plain")
    
    st.markdown(f"""
    <div class="footer">
        🏦 Desenrola Brasil · Fonte: Banco Central do Brasil (SCR)<br>
        Dashboard desenvolvido para portfólio de Análise de Dados
        <div class="footer-disclaimer">⚠️ Dados de domínio público. Dashboard para fins de portfólio.</div>
    </div>
    """, unsafe_allow_html=True)

else:
    st.error("❌ Erro ao carregar os dados. Verifique se o arquivo 'dados_desenrola.csv' está no repositório.")
