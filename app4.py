import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
from datetime import datetime

# ============================================================
# CONFIGURAÇÃO
# ============================================================
st.set_page_config(
    page_title="Desenrola Brasil - Análise de Renegociação",
    page_icon="🇧🇷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CORES
# ============================================================
COR_PRIMARIA = "#003087"
COR_SECUNDARIA = "#008000"
COR_DESTAQUE = "#FFCC00"
COR_FUNDO = "#F5F7FA"
COR_TEXTO = "#1A2B4C"

# ============================================================
# CSS
# ============================================================
st.markdown(f"""
<style>
.stApp {{ background: {COR_FUNDO} !important; }}
html, body, .stApp {{ color: {COR_TEXTO} !important; font-family: 'Inter', sans-serif !important; }}
#MainMenu, footer, header {{ visibility: hidden; }}
.block-container {{ padding: 2rem 2.5rem !important; max-width: 1400px !important; }}

.page-header {{
    background: linear-gradient(135deg, {COR_PRIMARIA}, {COR_SECUNDARIA});
    padding: 1.5rem 2rem;
    border-radius: 16px;
    margin-bottom: 28px;
}}
.page-header h1 {{
    font-size: 28px !important;
    font-weight: 700 !important;
    color: white !important;
    margin: 0 0 8px 0 !important;
}}
.page-header p {{
    color: rgba(255,255,255,0.9);
    font-size: 14px;
    margin: 0;
}}
.update-badge {{
    background: {COR_DESTAQUE};
    color: {COR_PRIMARIA};
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 600;
    display: inline-block;
    margin-top: 12px;
}}

.kpi-grid {{ display: flex; gap: 20px; margin-bottom: 32px; flex-wrap: wrap; }}
.kpi-card {{
    flex: 1; min-width: 180px; background: white; border-radius: 16px;
    padding: 24px 20px; text-align: center; border: 1px solid #E8ECF0;
    border-top: 4px solid {COR_PRIMARIA};
}}
.kpi-label {{ font-size: 12px; text-transform: uppercase; color: #6B7A8F; margin-bottom: 8px; font-weight: 600; }}
.kpi-value {{ font-size: 32px; font-weight: 700; color: {COR_TEXTO}; margin: 8px 0; }}
.kpi-sub {{ font-size: 11px; color: #6B7A8F; }}

.insight-card {{
    background: linear-gradient(135deg, #E8F4FD, #FFFFFF);
    border-left: 4px solid {COR_DESTAQUE};
    padding: 16px 20px;
    border-radius: 12px;
    margin-bottom: 20px;
}}
.insight-title {{ font-size: 13px; font-weight: 700; color: {COR_PRIMARIA}; margin-bottom: 8px; }}
.insight-text {{ font-size: 14px; color: {COR_TEXTO}; }}

.section-header {{
    display: flex; align-items: center; justify-content: space-between;
    margin: 32px 0 16px 0; border-bottom: 1px solid #E8ECF0;
    padding-bottom: 10px;
}}
.section-header h2 {{ font-size: 20px; font-weight: 700; color: {COR_TEXTO}; margin: 0; }}
.section-tag {{
    background: #E8F4FD; padding: 4px 12px; border-radius: 20px;
    font-size: 11px; color: {COR_PRIMARIA};
}}

.footer {{
    text-align: center; padding: 30px 0 20px; margin-top: 40px;
    border-top: 1px solid #E8ECF0; color: #6B7A8F; font-size: 12px;
}}
</style>
""", unsafe_allow_html=True)

# ============================================================
# HEADER
# ============================================================
st.markdown(f"""
<div class="page-header">
    <h1>🇧🇷 Desenrola Brasil</h1>
    <p>Análise de Renegociação de Dívidas | Dados Oficiais do Banco Central (set/2023 - mar/2026)</p>
    <div class="update-badge">📅 Última atualização: {datetime.now().strftime('%d/%m/%Y')}</div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("### 🎯 Filtros")
    uploaded_file = st.file_uploader("📄 Upload do CSV (dados_desenrola.csv)", type=["csv"])
    
    st.markdown("---")
    st.markdown("### 📊 Sobre os dados")
    st.markdown("""
    - **Fonte:** Banco Central do Brasil (SCR)
    - **Período:** set/2023 - mar/2026
    - **Total de operações:** 2,75 milhões
    - **Volume financeiro:** R$ 7,62 bilhões
    """)

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
def carregar_dados(uploaded_file):
    """Carrega CSV com deteccao automatica de encoding"""
    # Lista de encodings para tentar
    encodings = ['latin1', 'ISO-8859-1', 'cp1252', 'utf-8', 'WIN1252']
    
    for encoding in encodings:
        try:
            uploaded_file.seek(0)  # Volta ao inicio do arquivo
            df = pd.read_csv(uploaded_file, delimiter=';', encoding=encoding, low_memory=False)
            
            # Padroniza nomes das colunas
            df.columns = df.columns.str.lower().str.strip()
            
            # Converte colunas numéricas
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
            
            # Remove linhas sem dados
            df = df.dropna(subset=['numero_operacoes', 'volume_operacoes'])
            
            st.success(f"✅ Leitura bem-sucedida! Encoding: {encoding}")
            return df
            
        except Exception as e:
            continue
    
    st.error("❌ Nenhum encoding funcionou. Verifique o arquivo CSV.")
    return None

# ============================================================
# MAIN
# ============================================================
if uploaded_file is not None:
    with st.spinner("🔄 Processando dados..."):
        df = carregar_dados(uploaded_file)
    
    if df is not None and len(df) > 0:
        st.success(f"✅ {len(df):,} registros carregados")
        
        # Verifica se os dados foram carregados corretamente
        st.subheader("🔍 Preview dos dados carregados")
        st.dataframe(df.head(10), use_container_width=True)
        
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
                <div class="kpi-label">📅 TOTAL REGISTROS</div>
                <div class="kpi-value">{fmt_num(len(df))}</div>
                <div class="kpi-sub">linhas processadas</div>
            </div>
            """, unsafe_allow_html=True)
        
        # ===== GRÁFICO 1: POR TIPO DESENROLA =====
        if 'tipo_desenrola' in df.columns:
            st.markdown("""
            <div class="section-header">
                <h2>📊 Renegociações por Tipo do Programa</h2>
                <span class="section-tag">Faixas</span>
            </div>
            """, unsafe_allow_html=True)
            
            tipo_data = df.groupby('tipo_desenrola').agg({
                'numero_operacoes': 'sum',
                'volume_operacoes': 'sum'
            }).reset_index()
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig1 = px.pie(tipo_data, names='tipo_desenrola', values='numero_operacoes', hole=0.35,
                             title='Distribuição de Operações por Tipo',
                             color_discrete_sequence=[COR_PRIMARIA, COR_SECUNDARIA, COR_DESTAQUE])
                fig1.update_layout(template="plotly_white", height=450)
                fig1.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig1, use_container_width=True)
            
            with col2:
                fig2 = px.bar(tipo_data, x='tipo_desenrola', y='volume_operacoes', color='tipo_desenrola',
                             title='Volume Financeiro por Tipo (R$)',
                             color_discrete_sequence=[COR_PRIMARIA, COR_SECUNDARIA, COR_DESTAQUE])
                fig2.update_layout(template="plotly_white", height=450)
                st.plotly_chart(fig2, use_container_width=True)
        
        # ===== GRÁFICO 2: TOP ESTADOS =====
        if 'unidade_federacao' in df.columns:
            st.markdown("""
            <div class="section-header">
                <h2>🗺️ Top Estados com Mais Renegociações</h2>
                <span class="section-tag">Ranking</span>
            </div>
            """, unsafe_allow_html=True)
            
            uf_data = df.groupby('unidade_federacao').agg({
                'numero_operacoes': 'sum',
                'volume_operacoes': 'sum'
            }).reset_index().sort_values('numero_operacoes', ascending=False).head(10)
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig3 = px.bar(uf_data, x='unidade_federacao', y='numero_operacoes', color='numero_operacoes',
                             title='Número de Operações por UF',
                             color_continuous_scale='Blues')
                fig3.update_layout(template="plotly_white", height=450)
                st.plotly_chart(fig3, use_container_width=True)
            
            with col2:
                fig4 = px.bar(uf_data, x='unidade_federacao', y='volume_operacoes', color='volume_operacoes',
                             title='Volume Financeiro por UF (R$)',
                             color_continuous_scale='Greens')
                fig4.update_layout(template="plotly_white", height=450)
                st.plotly_chart(fig4, use_container_width=True)
        
        # ===== GRÁFICO 3: TOP BANCOS =====
        if 'nome_conglomerado_financeiro' in df.columns:
            st.markdown("""
            <div class="section-header">
                <h2>🏦 Top 10 Bancos com Mais Renegociações</h2>
                <span class="section-tag">Ranking</span>
            </div>
            """, unsafe_allow_html=True)
            
            banco_data = df.groupby('nome_conglomerado_financeiro')['numero_operacoes'].sum().sort_values(ascending=False).head(10).reset_index()
            banco_data.columns = ['Banco', 'Operações']
            
            fig5 = px.bar(banco_data, x='Operações', y='Banco', orientation='h',
                         title='Top 10 Bancos por Número de Renegociações',
                         color='Operações', color_continuous_scale='Viridis',
                         text='Operações')
            fig5.update_layout(template="plotly_white", height=500)
            fig5.update_traces(texttemplate='%{text:,}', textposition='outside')
            st.plotly_chart(fig5, use_container_width=True)
        
        # ===== INSIGHTS FINAIS =====
        st.markdown("""
        <div class="section-header">
            <h2>📈 Conclusões Estratégicas</h2>
            <span class="section-tag">Análise</span>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="insight-card" style="background:white;">
                <div class="insight-title">🏆 Programa</div>
                <div class="insight-value">R$ 7,62 Bi</div>
                <div class="insight-text">Volume total renegociado desde o lançamento do programa.</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="insight-card" style="background:white;">
                <div class="insight-title">📍 Maior Estado</div>
                <div class="insight-value">SP</div>
                <div class="insight-text">Líder em número de operações e volume financeiro.</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="insight-card" style="background:white;">
                <div class="insight-title">🏦 Líder em Renegociação</div>
                <div class="insight-value">Nubank</div>
                <div class="insight-text">Banco com maior número de renegociações no programa.</div>
            </div>
            """, unsafe_allow_html=True)
        
        # ===== EXPORTAÇÃO =====
        st.markdown("---")
        st.markdown("### 📥 Exportar Relatório")
        
        relatorio = f"""RELATÓRIO DESENROLA BRASIL
Data: {datetime.now().strftime('%d/%m/%Y %H:%M')}

VISÃO GERAL:
- Total Renegociado: {fmt_brl(total_volume)}
- Total de Operações: {fmt_num(total_operacoes)}
- Ticket Médio: {fmt_brl(ticket_medio)}

Fonte: Banco Central do Brasil (SCR/Desenrola)
"""
        
        st.download_button("📝 Exportar Relatório", relatorio, f"relatorio_desenrola_{datetime.now().strftime('%Y%m%d')}.txt", "text/plain")
        
        st.markdown(f"""
        <div class="footer">
            🇧🇷 Desenrola Brasil · Fonte: Banco Central do Brasil (SCR)<br>
            📅 Relatório gerado em {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}
        </div>
        """, unsafe_allow_html=True)
    
    else:
        st.error("❌ Erro ao carregar o arquivo.")

else:
    # ===== TELA INICIAL =====
    st.info("👈 **Faça upload do arquivo CSV no menu lateral para começar**")
    
    st.markdown("""
    <div style="background: white; border-radius: 16px; padding: 24px; margin-bottom: 24px; border: 1px solid #E8ECF0;">
        <h3 style="color: #1A2B4C; margin-bottom: 12px;">📋 Sobre o Dashboard</h3>
        <p style="color: #6B7A8F;">Este dashboard analisa os dados do <strong>Programa Desenrola Brasil</strong>, iniciativa do governo federal para renegociação de dívidas.</p>
        <p style="color: #6B7A8F; margin-top: 8px;">✅ <strong>Fonte oficial:</strong> Banco Central do Brasil (Sistema de Informações de Crédito - SCR)</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div style="background: white; border-radius: 12px; padding: 20px; text-align: center; border: 1px solid #E8ECF0;">
            <div style="font-size: 28px; font-weight: 700; color: #003087;">R$ 7,6B</div>
            <div style="font-size: 13px; color: #6B7A8F;">Volume Renegociado</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: white; border-radius: 12px; padding: 20px; text-align: center; border: 1px solid #E8ECF0;">
            <div style="font-size: 28px; font-weight: 700; color: #003087;">2,75M</div>
            <div style="font-size: 13px; color: #6B7A8F;">Operações</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="background: white; border-radius: 12px; padding: 20px; text-align: center; border: 1px solid #E8ECF0;">
            <div style="font-size: 28px; font-weight: 700; color: #003087;">Nubank</div>
            <div style="font-size: 13px; color: #6B7A8F;">Banco Líder</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div style="background: white; border-radius: 12px; padding: 20px; text-align: center; border: 1px solid #E8ECF0;">
            <div style="font-size: 28px; font-weight: 700; color: #003087;">SP</div>
            <div style="font-size: 13px; color: #6B7A8F;">Estado Líder</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="footer">
        🇧🇷 Desenrola Brasil · Dashboard para portfólio<br>
        Fonte: Banco Central do Brasil (SCR)
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# FIM
# ============================================================
