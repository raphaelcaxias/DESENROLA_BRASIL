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
    page_title="Desenrola Brasil - Análise de Renegociação",
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

# ============================================================
# CSS
# ============================================================
st.markdown(f"""
<style>
.stApp {{ background: {COR_FUNDO} !important; }}
.block-container {{ padding: 2rem 2.5rem !important; max-width: 1400px !important; }}

.page-header {{
    background: linear-gradient(135deg, {COR_PRIMARIA}, {COR_SECUNDARIA});
    padding: 1.5rem 2rem;
    border-radius: 12px;
    margin-bottom: 28px;
}}
.page-header h1 {{ font-size: 28px; font-weight: 700; color: white; margin: 0 0 8px 0; }}
.page-header p {{ color: rgba(255,255,255,0.85); font-size: 14px; }}
.update-badge {{
    background: {COR_DESTAQUE}; color: white; padding: 4px 12px;
    border-radius: 20px; font-size: 12px; font-weight: 600; display: inline-block; margin-top: 12px;
}}

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

.insight-card {{
    background: white; border-radius: 12px; padding: 20px; margin-bottom: 20px;
    border: 1px solid #E2E8F0; border-left: 4px solid {COR_DESTAQUE};
}}
.insight-title {{ font-size: 13px; font-weight: 700; color: {COR_SECUNDARIA}; text-transform: uppercase; }}
.insight-value {{ font-size: 24px; font-weight: 800; color: {COR_PRIMARIA}; margin: 10px 0; }}
.insight-text {{ font-size: 13px; color: {COR_SUBTEXTO}; }}

.section-header {{
    display: flex; align-items: center; justify-content: space-between;
    margin: 32px 0 20px 0; border-bottom: 2px solid #E2E8F0; padding-bottom: 12px;
}}
.section-header h2 {{ font-size: 20px; font-weight: 700; color: {COR_PRIMARIA}; margin: 0; }}
.section-tag {{
    background: #E8F4FD; padding: 4px 12px; border-radius: 20px;
    font-size: 11px; color: {COR_SECUNDARIA}; font-weight: 600;
}}

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
            
            df = df.dropna(subset=['numero_operacoes', 'volume_operacoes'])
            return df, encoding
        except:
            continue
    return None, None

# ============================================================
# SIDEBAR COM FILTROS
# ============================================================
with st.sidebar:
    st.markdown("### 🎯 Filtros Interativos")
    
    df, encoding = carregar_dados()
    
    if df is not None:
        # Filtro por Tipo
        if 'tipo_desenrola' in df.columns:
            tipos = sorted(df['tipo_desenrola'].dropna().unique())
            tipo_filtro = st.multiselect("Tipo do Programa", tipos, default=tipos)
            if tipo_filtro:
                df_filtrado = df[df['tipo_desenrola'].isin(tipo_filtro)]
            else:
                df_filtrado = df.copy()
        else:
            df_filtrado = df.copy()
        
        # Filtro por UF
        if 'unidade_federacao' in df.columns:
            ufs = sorted(df['unidade_federacao'].dropna().unique())
            uf_filtro = st.multiselect("Unidade da Federação", ufs, default=[])
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
        st.markdown("### 📊 Informações")
        st.markdown("""
        **Fonte:** Banco Central do Brasil (SCR)
        **Período:** set/2023 - mar/2026
        
        **Tipos do programa:**
        - 🔵 Tipo 1: Faixa 1 (PF)
        - 🟢 Tipo 2: Faixa 2 (PF)
        - 🟡 Tipo 3: Pequenos Negócios
        """)
    else:
        df_filtrado = None

# ============================================================
# MAIN
# ============================================================
if df_filtrado is not None and len(df_filtrado) > 0:
    st.success(f"✅ {len(df_filtrado):,} registros processados (após filtros)")
    
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
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">🏛️ INSTITUIÇÕES</div>
            <div class="kpi-value">{df_filtrado['nome_conglomerado_financeiro'].nunique() if 'nome_conglomerado_financeiro' in df_filtrado.columns else 'N/A'}</div>
            <div class="kpi-sub">bancos participantes</div>
        </div>
        """, unsafe_allow_html=True)
    
    # ===== GRÁFICO 1: EVOLUÇÃO MENSAL (NOVO) =====
    if 'data_base' in df_filtrado.columns and not df_filtrado['data_base'].isna().all():
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
        
        fig_evolucao = px.line(evolucao, x='data_base', y='volume_operacoes', 
                                title='Volume Renegociado por Mês',
                                markers=True, line_shape='spline')
        fig_evolucao.update_layout(template="plotly_white", height=400, 
                                    xaxis_title="Mês", yaxis_title="Volume (R$)")
        fig_evolucao.update_traces(line=dict(width=3, color=COR_SECUNDARIA), marker=dict(size=6))
        st.plotly_chart(fig_evolucao, use_container_width=True)
    
    # ===== GRÁFICO 2: POR TIPO =====
    if 'tipo_desenrola' in df_filtrado.columns:
        st.markdown("""
        <div class="section-header">
            <h2>📊 Análise por Tipo do Programa</h2>
            <span class="section-tag">Distribuição</span>
        </div>
        """, unsafe_allow_html=True)
        
        tipo_data = df_filtrado.groupby('tipo_desenrola').agg({
            'numero_operacoes': 'sum',
            'volume_operacoes': 'sum'
        }).reset_index()
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_pie = px.pie(tipo_data, names='tipo_desenrola', values='numero_operacoes', hole=0.4,
                            title='Distribuição de Operações',
                            color_discrete_sequence=[COR_PRIMARIA, COR_SECUNDARIA, COR_DESTAQUE])
            fig_pie.update_layout(template="plotly_white", height=420)
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # Ticket médio por tipo
            tipo_data['ticket_medio'] = tipo_data['volume_operacoes'] / tipo_data['numero_operacoes']
            fig_bar = px.bar(tipo_data, x='tipo_desenrola', y='ticket_medio',
                            title='Ticket Médio por Tipo (R$)',
                            color='tipo_desenrola',
                            color_discrete_sequence=[COR_PRIMARIA, COR_SECUNDARIA, COR_DESTAQUE],
                            text_auto='.2s')
            fig_bar.update_layout(template="plotly_white", height=420)
            st.plotly_chart(fig_bar, use_container_width=True)
    
    # ===== GRÁFICO 3: TOP ESTADOS =====
    if 'unidade_federacao' in df_filtrado.columns:
        st.markdown("""
        <div class="section-header">
            <h2>🗺️ Ranking por Unidade da Federação</h2>
            <span class="section-tag">Top 10</span>
        </div>
        """, unsafe_allow_html=True)
        
        uf_data = df_filtrado.groupby('unidade_federacao').agg({
            'numero_operacoes': 'sum',
            'volume_operacoes': 'sum'
        }).reset_index()
        uf_data['ticket_medio'] = uf_data['volume_operacoes'] / uf_data['numero_operacoes']
        uf_data = uf_data.sort_values('numero_operacoes', ascending=False).head(10)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_uf_oper = px.bar(uf_data, x='unidade_federacao', y='numero_operacoes',
                                title='Operações por UF',
                                color='numero_operacoes', color_continuous_scale='Blues')
            fig_uf_oper.update_layout(template="plotly_white", height=420)
            st.plotly_chart(fig_uf_oper, use_container_width=True)
        
        with col2:
            fig_uf_ticket = px.bar(uf_data, x='unidade_federacao', y='ticket_medio',
                                  title='Ticket Médio por UF (R$)',
                                  color='ticket_medio', color_continuous_scale='Greens')
            fig_uf_ticket.update_layout(template="plotly_white", height=420)
            st.plotly_chart(fig_uf_ticket, use_container_width=True)
    
    # ===== GRÁFICO 4: TOP BANCOS =====
    if 'nome_conglomerado_financeiro' in df_filtrado.columns:
        st.markdown("""
        <div class="section-header">
            <h2>🏦 Ranking de Instituições Financeiras</h2>
            <span class="section-tag">Top 15</span>
        </div>
        """, unsafe_allow_html=True)
        
        banco_data = df_filtrado.groupby('nome_conglomerado_financeiro')['numero_operacoes'].sum().sort_values(ascending=False).head(15).reset_index()
        banco_data.columns = ['Instituição', 'Operações']
        
        # Calcular participação percentual
        total_oper = banco_data['Operações'].sum()
        banco_data['Participação'] = (banco_data['Operações'] / total_oper * 100).round(1).astype(str) + '%'
        
        fig_banco = px.bar(banco_data, x='Operações', y='Instituição', orientation='h',
                          title='Top 15 Instituições por Renegociações',
                          color='Operações', color_continuous_scale='Viridis',
                          text='Operações')
        fig_banco.update_layout(template="plotly_white", height=550)
        fig_banco.update_traces(texttemplate='%{text:,}', textposition='outside')
        st.plotly_chart(fig_banco, use_container_width=True)
        
        # Tabela de participação
        with st.expander("📋 Ver tabela de participação dos bancos"):
            st.dataframe(banco_data, use_container_width=True, hide_index=True)
    
    # ===== GRÁFICO 5: MAPA DE CALOR (NOVO) =====
    if 'unidade_federacao' in df_filtrado.columns and 'nome_conglomerado_financeiro' in df_filtrado.columns:
        st.markdown("""
        <div class="section-header">
            <h2>🔥 Mapa de Calor: Bancos x Estados</h2>
            <span class="section-tag">Top 5 Bancos</span>
        </div>
        """, unsafe_allow_html=True)
        
        top_bancos = df_filtrado.groupby('nome_conglomerado_financeiro')['numero_operacoes'].sum().nlargest(5).index.tolist()
        heatmap_data = df_filtrado[df_filtrado['nome_conglomerado_financeiro'].isin(top_bancos)]
        heatmap_data = heatmap_data.groupby(['nome_conglomerado_financeiro', 'unidade_federacao'])['numero_operacoes'].sum().reset_index()
        
        fig_heatmap = px.density_heatmap(heatmap_data, x='unidade_federacao', y='nome_conglomerado_financeiro', 
                                         z='numero_operacoes', title='Concentração de Operações por Banco/UF',
                                         color_continuous_scale='Blues')
        fig_heatmap.update_layout(template="plotly_white", height=500)
        st.plotly_chart(fig_heatmap, use_container_width=True)
    
    # ===== CONCLUSÃO =====
    st.markdown("""
    <div class="section-header">
        <h2>📈 Conclusão</h2>
        <span class="section-tag">Resumo Executivo</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Calcular principais insights
    banco_lider = df_filtrado.groupby('nome_conglomerado_financeiro')['numero_operacoes'].sum().idxmax() if 'nome_conglomerado_financeiro' in df_filtrado.columns else "N/A"
    uf_lider = df_filtrado.groupby('unidade_federacao')['volume_operacoes'].sum().idxmax() if 'unidade_federacao' in df_filtrado.columns else "N/A"
    
    st.markdown(f"""
    <div class="insight-card" style="background: linear-gradient(135deg, {COR_PRIMARIA}, {COR_SECUNDARIA}); color: white; border: none;">
        <div class="insight-text" style="color: rgba(255,255,255,0.95);">
        O Programa <strong>Desenrola Brasil</strong> já renegociou <strong>{fmt_brl(total_volume)}</strong> em dívidas, 
        com <strong>{fmt_num(total_operacoes)} operações</strong> realizadas. O <strong>{banco_lider}</strong> lidera as renegociações,
        enquanto <strong>{uf_lider}</strong> concentra a maior parte do volume financeiro.
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
    st.error("❌ Erro ao carregar os dados. Verifique se o arquivo 'dados_desenrola.csv' está no repositório.")
