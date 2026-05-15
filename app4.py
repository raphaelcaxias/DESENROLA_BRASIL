import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import re
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURAÇÃO DA PÁGINA (Identidade Corporativa)
# ============================================================
st.set_page_config(
    page_title="Corporate Analytics | Desenrola Brasil",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# PALETA DE CORES INSTITUCIONAL (Executive Dark/Light)
# ============================================================
if "tema" not in st.session_state:
    st.session_state.tema = "claro"

if st.session_state.tema == "claro":
    COR_FUNDO = "#F4F6F9"
    COR_CARD = "#FFFFFF"
    COR_TEXTO = "#1E293B"
    COR_BORDA = "#E2E8F0"
    COR_PRIMARIA = "#0F172A"
    COR_SECUNDARIA = "#2563EB"
    COR_SUCESSO = "#16A34A"
    COR_ALERTA = "#DC2626"
    COR_ATENCAO = "#D97706"
else:
    COR_FUNDO = "#0B0F19"
    COR_CARD = "#111827"
    COR_TEXTO = "#F3F4F6"
    COR_BORDA = "#1F2937"
    COR_PRIMARIA = "#38BDF8"
    COR_SECUNDARIA = "#60A5FA"
    COR_SUCESSO = "#34D399"
    COR_ALERTA = "#F87171"
    COR_ATENCAO = "#FBBF24"

# ============================================================
# CSS CUSTOMIZADO (Design de Sistema Financeiro)
# ============================================================
st.markdown(f"""
<style>
    html, body, .stApp {{
        background-color: {COR_FUNDO};
        color: {COR_TEXTO};
        font-family: 'Inter', -apple-system, sans-serif;
    }}
    .block-container {{
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }}
    .kpi-card {{
        background: {COR_CARD};
        border-left: 4px solid {COR_SECUNDARIA};
        border-top: 1px solid {COR_BORDA};
        border-right: 1px solid {COR_BORDA};
        border-bottom: 1px solid {COR_BORDA};
        border-radius: 6px;
        padding: 1.2rem;
        margin-bottom: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }}
    .kpi-title {{
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748B;
        font-weight: 600;
    }}
    .kpi-value {{
        font-size: 1.6rem;
        font-weight: 700;
        color: {COR_TEXTO};
        margin-top: 0.3rem;
    }}
    .badge {{
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.8rem;
    }}
    .badge-low {{ background-color: rgba(22, 163, 74, 0.1); color: #16A34A; }}
    .badge-mid {{ background-color: rgba(217, 119, 6, 0.1); color: #D97706; }}
    .badge-high {{ background-color: rgba(220, 38, 38, 0.1); color: #DC2626; }}
</style>
""", unsafe_allow_html=True)

# ============================================================
# FUNÇÕES DE FORMATAÇÃO E TRATAMENTO
# ============================================================
def fmt_brl(valor):
    if pd.isna(valor) or valor == 0:
        return "R$ 0,00"
    if valor >= 1_000_000_000:
        return f"R$ {valor/1_000_000_000:.2f} Bi".replace(".", ",")
    if valor >= 1_000_000:
        return f"R$ {valor/1_000_000:.2f} Mi".replace(".", ",")
    return f"R$ {valor:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

def fmt_num(valor):
    if pd.isna(valor):
        return "0"
    return f"{int(valor):,}".replace(",", ".")

def fmt_percentual(valor, total):
    if total == 0:
        return "0%"
    return f"{(valor/total*100):.1f}%"

def classificar_banco(nome):
    nome = str(nome).upper().strip()
    nome = re.sub(r'\s*-\s*PRUDENCIAL$', '', nome)
    if any(x in nome for x in ["NUBANK", "INTER", "C6", "NEON", "ORIGINAL", "BANCO C6"]):
        return "Banco Digital"
    if any(x in nome for x in ["ITAU", "BRADESCO", "SANTANDER", "CAIXA", "BANCO DO BRASIL", "BB"]):
        return "Banco Tradicional"
    if any(x in nome for x in ["BTG", "BTG PACTUAL"]):
        return "Banco de Investimento"
    return "Demais Categorias"

def agrupar_regiao(uf):
    mapa = {
        "Norte": ["AC", "AM", "AP", "PA", "RO", "RR", "TO"],
        "Nordeste": ["AL", "BA", "CE", "MA", "PB", "PE", "PI", "RN", "SE"],
        "Centro-Oeste": ["DF", "GO", "MS", "MT"],
        "Sudeste": ["ES", "MG", "RJ", "SP"],
        "Sul": ["PR", "RS", "SC"]
    }
    for regiao, estados in mapa.items():
        if uf in estados:
            return regiao
    return "Não Identificado"

def calcular_hhi(df, coluna):
    total = df[coluna].sum()
    if total == 0:
        return 0
    return ((df[coluna] / total) ** 2).sum() * 10000

def interpretar_hhi(hhi):
    if hhi < 1500:
        return "Baixa Concentração (Mercado Competitivo)", "badge-low"
    elif hhi < 2500:
        return "Concentração Moderada", "badge-mid"
    return "Altamente Concentrado (Risco Sistêmico)", "badge-high"

def calcular_pareto(df, coluna):
    df_sorted = df.sort_values(coluna, ascending=False).reset_index(drop=True)
    df_sorted['percentual_acumulado'] = (df_sorted[coluna].cumsum() / df_sorted[coluna].sum()) * 100
    return df_sorted

# ============================================================
# NOVAS FUNÇÕES ANALÍTICAS (VERSÃO 10/10)
# ============================================================

def analise_sazonalidade(df):
    """Análise de sazonalidade - identifica meses com mais renegociações"""
    df_saz = df.copy()
    df_saz['mes'] = df_saz['data_base'].dt.month
    df_saz['ano'] = df_saz['data_base'].dt.year
    
    sazonal_mes = df_saz.groupby('mes')['volume_operacoes'].agg(['sum', 'mean', 'std']).reset_index()
    nomes_meses = ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun', 'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez']
    sazonal_mes['mes_nome'] = sazonal_mes['mes'].apply(lambda x: nomes_meses[x-1])
    
    fig = px.bar(sazonal_mes, x='mes_nome', y='mean', 
                 error_y='std', title='🎯 Sazonalidade: Volume Médio por Mês',
                 labels={'mes_nome': 'Mês', 'mean': 'Volume Médio (R$)'},
                 color='mean', color_continuous_scale='Blues')
    fig.update_layout(template="plotly_white", height=400)
    return fig

def matriz_correlacao(df):
    """Matriz de correlação entre variáveis numéricas"""
    colunas_numericas = ['numero_operacoes', 'volume_operacoes']
    if 'ticket_medio' in df.columns:
        colunas_numericas.append('ticket_medio')
    
    correlacao = df[colunas_numericas].corr()
    fig = px.imshow(correlacao, text_auto=True, aspect='auto', 
                    title='📊 Matriz de Correlação entre Variáveis',
                    color_continuous_scale='RdBu', zmin=-1, zmax=1)
    fig.update_layout(height=400)
    return fig

def boxplot_outliers(df, col_banco_nome):
    """Identifica outliers por banco"""
    top10 = df.groupby(col_banco_nome)['numero_operacoes'].sum().nlargest(10).index
    df_top = df[df[col_banco_nome].isin(top10)]
    
    fig = px.box(df_top, x=col_banco_nome, y='numero_operacoes', 
                 title='⚠️ Distribuição de Operações por Banco (Detecção de Outliers)',
                 labels={col_banco_nome: 'Instituição', 'numero_operacoes': 'Operações'},
                 color=col_banco_nome)
    fig.update_layout(height=450, showlegend=False)
    return fig

def treemap_cruzado(df):
    """Treemap de ticket médio por região e tipo de banco"""
    cruzado = df.groupby(['regiao', 'tipo_banco']).agg({
        'numero_operacoes': 'sum',
        'volume_operacoes': 'sum'
    }).reset_index()
    cruzado['ticket_medio'] = cruzado['volume_operacoes'] / cruzado['numero_operacoes']
    
    fig = px.treemap(cruzado, path=['regiao', 'tipo_banco'], 
                     values='volume_operacoes', color='ticket_medio',
                     title='🗺️ Ticket Médio por Região e Tipo de Banco',
                     color_continuous_scale='Blues',
                     hover_data={'ticket_medio': ':,.2f'})
    fig.update_layout(height=450)
    return fig

def clusterizar_bancos(df, col_banco_nome):
    """Clusterização simples de bancos por comportamento (K-Means)"""
    cluster_data = df.groupby(col_banco_nome).agg({
        'numero_operacoes': 'sum',
        'volume_operacoes': 'sum'
    }).reset_index()
    
    cluster_data['ticket_medio'] = cluster_data['volume_operacoes'] / cluster_data['numero_operacoes']
    cluster_data = cluster_data[cluster_data['numero_operacoes'] > 100]
    
    if len(cluster_data) > 3:
        scaler = StandardScaler()
        features = scaler.fit_transform(cluster_data[['numero_operacoes', 'ticket_medio']])
        
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        cluster_data['cluster'] = kmeans.fit_predict(features)
        
        nomes_cluster = {0: '🏢 Alto Volume/Baixo Ticket', 
                         1: '💎 Baixo Volume/Alto Ticket', 
                         2: '⚖️ Equilibrado'}
        cluster_data['cluster_nome'] = cluster_data['cluster'].map(nomes_cluster)
        
        fig = px.scatter(cluster_data, x='numero_operacoes', y='ticket_medio', 
                         color='cluster_nome', size='volume_operacoes',
                         hover_name=col_banco_nome,
                         title='🧠 Clusterização de Bancos por Comportamento',
                         labels={'numero_operacoes': 'Operações', 'ticket_medio': 'Ticket Médio (R$)'})
        fig.update_layout(height=450)
        return fig, cluster_data
    return None, None

def analise_yoy(df):
    """Year-over-Year: Comparação do mesmo mês em anos diferentes"""
    df_yoy = df.copy()
    df_yoy['ano'] = df_yoy['data_base'].dt.year
    df_yoy['mes'] = df_yoy['data_base'].dt.month
    
    yoy_data = df_yoy.groupby(['ano', 'mes'])['volume_operacoes'].sum().reset_index()
    yoy_data = yoy_data[yoy_data['ano'] >= 2024]
    
    fig = px.line(yoy_data, x='mes', y='volume_operacoes', color='ano',
                  title='📈 Comparativo YoY (Year over Year)',
                  labels={'mes': 'Mês', 'volume_operacoes': 'Volume (R$)', 'ano': 'Ano'},
                  markers=True)
    fig.update_layout(height=400)
    return fig

def ranking_percentual(df, col_banco_nome):
    """Ranking com participação percentual detalhada"""
    ranking = df.groupby(col_banco_nome)['numero_operacoes'].sum().sort_values(ascending=False).head(15).reset_index()
    total = ranking['numero_operacoes'].sum()
    ranking['participacao'] = (ranking['numero_operacoes'] / total * 100).round(1)
    ranking['participacao_acumulada'] = ranking['participacao'].cumsum().round(1)
    ranking.columns = ['Instituição', 'Operações', '% Individual', '% Acumulado']
    return ranking

def aplicar_layout_grafico(fig):
    fig.update_layout(
        margin=dict(l=40, r=40, t=50, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=COR_TEXTO, family="Inter"),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.update_xaxes(showgrid=False, color=COR_TEXTO)
    fig.update_yaxes(showgrid=True, gridcolor=COR_BORDA, color=COR_TEXTO)
    return fig

# ============================================================
# CARREGAMENTO DOS DADOS
# ============================================================
@st.cache_data(ttl=3600)
def carregar_dados():
    encodings = ["utf-8", "latin1", "cp1252"]
    for enc in encodings:
        try:
            df = pd.read_csv("dados_desenrola.csv", sep=";", encoding=enc, low_memory=False)
            df.columns = df.columns.str.lower().str.strip()
            
            for col in ["numero_operacoes", "volume_operacoes"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(
                        df[col].astype(str).str.replace(".", "", regex=False).str.replace(",", ".", regex=False), 
                        errors="coerce"
                    )
            
            df["data_base"] = pd.to_datetime(df["data_base"].astype(str), format="%Y%m", errors="coerce")
            df["tipo_banco"] = df["nome_conglomerado_financeiro"].apply(classificar_banco)
            df["regiao"] = df["unidade_federacao"].apply(agrupar_regiao)
            return df.dropna(subset=["volume_operacoes", "numero_operacoes"])
        except Exception as e:
            continue
    return None

df = carregar_dados()
if df is None:
    st.error("Erro crítico: Base de dados 'dados_desenrola.csv' não localizada ou corrompida.")
    st.stop()

# ============================================================
# SIDEBAR CONTROL PANEL
# ============================================================
with st.sidebar:
    st.markdown("### 🎛️ Painel de Controle")
    
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        if st.button("☀️ Modo Claro", use_container_width=True):
            st.session_state.tema = "claro"
            st.rerun()
    with col_t2:
        if st.button("🌙 Modo Escuro", use_container_width=True):
            st.session_state.tema = "escuro"
            st.rerun()
    
    st.markdown("---")
    
    tipos = sorted(df["tipo_desenrola"].unique())
    tipo = st.multiselect("Faixa do Programa", tipos, default=tipos)
    
    regioes = sorted(df["regiao"].unique())
    regiao = st.multiselect("Região Geográfica", regioes, default=regioes)
    
    bancos = sorted(df["tipo_banco"].unique())
    banco = st.multiselect("Segmento Institucional", bancos, default=bancos)
    
    if st.button("🔄 Resetar Filtros", use_container_width=True):
        st.rerun()

# Filtragem
df_filtrado = df[
    (df["tipo_desenrola"].isin(tipo)) &
    (df["regiao"].isin(regiao)) &
    (df["tipo_banco"].isin(banco))
]

if df_filtrado.empty:
    st.warning("⚠️ Nenhum registro encontrado para o cruzamento de filtros selecionado.")
    st.stop()

# ============================================================
# EXECUTIVE HEADER
# ============================================================
col_header, col_date = st.columns([2, 1])
with col_header:
    st.title("🏦 Desenrola Brasil Analytics")
    st.caption("Relatório Gerencial de Monitoramento de Crédito e Renegociações — Fonte: Banco Central do Brasil")
with col_date:
    st.markdown(f"<div style='text-align: right; color: gray; font-size: 0.85rem; padding-top: 25px;'>Atualizado em: {datetime.now().strftime('%d/%m/%Y')}</div>", unsafe_allow_html=True)

st.markdown("---")

# ============================================================
# KPIs PRINCIPAIS
# ============================================================
total_volume = df_filtrado["volume_operacoes"].sum()
total_operacoes = df_filtrado["numero_operacoes"].sum()
ticket_medio = total_volume / total_operacoes if total_operacoes > 0 else 0
num_bancos = df_filtrado["nome_conglomerado_financeiro"].nunique()

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
with kpi1:
    st.markdown(f'<div class="kpi-card"><div class="kpi-title">💰 Volume Carteira Ativa</div><div class="kpi-value">{fmt_brl(total_volume)}</div></div>', unsafe_allow_html=True)
with kpi2:
    st.markdown(f'<div class="kpi-card"><div class="kpi-title">📋 Total de Contratos</div><div class="kpi-value">{fmt_num(total_operacoes)}</div></div>', unsafe_allow_html=True)
with kpi3:
    st.markdown(f'<div class="kpi-card"><div class="kpi-title">🎫 Ticket Médio</div><div class="kpi-value">{fmt_brl(ticket_medio)}</div></div>', unsafe_allow_html=True)
with kpi4:
    st.markdown(f'<div class="kpi-card"><div class="kpi-title">🏦 Instituições</div><div class="kpi-value">{fmt_num(num_bancos)}</div></div>', unsafe_allow_html=True)

# ============================================================
# NAVEGAÇÃO EM ABAS (VERSÃO 10/10 - 6 ABAS)
# ============================================================
aba_temporal, aba_market, aba_regional, aba_analise, aba_cluster, aba_insights = st.tabs([
    "📈 Evolução & Projeções",
    "🏦 Market Share & HHI",
    "🗺️ Distribuição Regional",
    "🧬 Análise Avançada",
    "🧠 Clusterização",
    "📊 Insights & Exportação"
])

# ------------------------------------------------------------
# ABA 1: EVOLUÇÃO E PROJEÇÕES
# ------------------------------------------------------------
with aba_temporal:
    col_t_left, col_t_right = st.columns([2, 1])
    
    with col_t_left:
        st.markdown("### Histórico de Volume vs Média Móvel")
        evolucao = df_filtrado.groupby("data_base").agg({"volume_operacoes": "sum", "numero_operacoes": "sum"}).reset_index()
        evolucao["crescimento"] = evolucao["volume_operacoes"].pct_change() * 100
        evolucao["media_movel"] = evolucao["volume_operacoes"].rolling(window=3 if len(evolucao) >=3 else 1).mean()
        
        fig_ev = go.Figure()
        fig_ev.add_trace(go.Scatter(x=evolucao["data_base"], y=evolucao["volume_operacoes"], mode="lines+markers", name="Volume Período", line=dict(color=COR_SECUNDARIA, width=3)))
        fig_ev.add_trace(go.Scatter(x=evolucao["data_base"], y=evolucao["media_movel"], mode="lines", name="Média Móvel", line=dict(color=COR_ATENCAO, dash="dash")))
        st.plotly_chart(aplicar_layout_grafico(fig_ev), use_container_width=True)
    
    with col_t_right:
        st.markdown("### Desempenho Mensal")
        cresc_tabela = evolucao[["data_base", "crescimento"]].dropna().tail(5).copy()
        cresc_tabela["data_base"] = cresc_tabela["data_base"].dt.strftime("%m/%Y")
        cresc_tabela["crescimento"] = cresc_tabela["crescimento"].apply(lambda x: f"{x:+.2f}%")
        cresc_tabela.columns = ["Competência", "Variação MoM"]
        st.dataframe(cresc_tabela, use_container_width=True, hide_index=True)
    
    # Projeção
    if len(evolucao) > 3:
        st.markdown("---")
        st.markdown("### Modelo Preditivo Linear (Próximo Trimestre)")
        modelo = evolucao.dropna().copy()
        modelo["indice"] = np.arange(len(modelo))
        
        lr = LinearRegression()
        lr.fit(modelo[["indice"]], modelo["volume_operacoes"])
        
        futuro = pd.DataFrame({"indice": np.arange(len(modelo), len(modelo) + 3)})
        previsao = lr.predict(futuro)
        datas_futuras = pd.date_range(evolucao["data_base"].max(), periods=4, freq="MS")[1:]
        
        fig_prev = go.Figure()
        fig_prev.add_trace(go.Scatter(x=evolucao["data_base"], y=evolucao["volume_operacoes"], name="Histórico", line=dict(color=COR_SECUNDARIA, width=2)))
        fig_prev.add_trace(go.Scatter(x=datas_futuras, y=previsao, name="Projeção", line=dict(color=COR_ATENCAO, dash="dot", width=2)))
        st.plotly_chart(aplicar_layout_grafico(fig_prev), use_container_width=True)
    
    # Sazonalidade (NOVO)
    st.markdown("---")
    st.markdown("### Análise de Sazonalidade")
    fig_saz = analise_sazonalidade(df_filtrado)
    st.plotly_chart(aplicar_layout_grafico(fig_saz), use_container_width=True)
    
    # YoY (NOVO)
    st.markdown("### Comparativo Anual (YoY)")
    fig_yoy = analise_yoy(df_filtrado)
    st.plotly_chart(aplicar_layout_grafico(fig_yoy), use_container_width=True)

# ------------------------------------------------------------
# ABA 2: MARKET SHARE E HHI
# ------------------------------------------------------------
with aba_market:
    col_banco_nome = "nome_conglomerado_financeiro"
    market = df_filtrado.groupby(col_banco_nome)["numero_operacoes"].sum().sort_values(ascending=False).reset_index()
    
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.markdown("### Índice de Concentração (HHI)")
        hhi = calcular_hhi(market, "numero_operacoes")
        classificacao, classe_css = interpretar_hhi(hhi)
        st.markdown(f"""
        <div style="background:{COR_CARD}; border:1px solid {COR_BORDA}; padding: 20px; border-radius:6px;">
            <p style="margin:0; font-size:0.9rem; color:gray;">Índice Herfindahl-Hirschman</p>
            <h2 style="margin:5px 0 15px 0; font-size:2.5rem; font-weight:700;">{hhi:.0f}</h2>
            <span class="badge {classe_css}">{classificacao}</span>
        </div>
        """, unsafe_allow_html=True)
    
    with col_m2:
        st.markdown("### Estrutura de Pareto (80/20)")
        pareto_data = calcular_pareto(market.head(10), "numero_operacoes")
        fig_pareto = go.Figure()
        fig_pareto.add_trace(go.Bar(x=pareto_data[col_banco_nome], y=pareto_data["numero_operacoes"], name="Contratos", marker_color=COR_SECUNDARIA))
        fig_pareto.add_trace(go.Scatter(x=pareto_data[col_banco_nome], y=pareto_data["percentual_acumulado"], name="% Acumulado", yaxis="y2", mode="lines+markers", line=dict(color=COR_ALERTA, width=2)))
        fig_pareto.update_layout(yaxis2=dict(overlaying="y", side="right", range=[0, 105], showgrid=False))
        st.plotly_chart(aplicar_layout_grafico(fig_pareto), use_container_width=True)
    
    st.markdown("---")
    st.markdown("### Ranking Detalhado com Participação Percentual")
    ranking_detalhado = ranking_percentual(df_filtrado, col_banco_nome)
    st.dataframe(ranking_detalhado, use_container_width=True, hide_index=True)
    
    # Matriz de Correlação (NOVO)
    st.markdown("---")
    st.markdown("### Matriz de Correlação")
    fig_corr = matriz_correlacao(df_filtrado)
    st.plotly_chart(aplicar_layout_grafico(fig_corr), use_container_width=True)

# ------------------------------------------------------------
# ABA 3: DISTRIBUIÇÃO REGIONAL
# ------------------------------------------------------------
with aba_regional:
    col_r1, col_r2 = st.columns(2)
    with col_r1:
        st.markdown("### Volume Financeiro por Região")
        regiao_data = df_filtrado.groupby("regiao")["volume_operacoes"].sum().reset_index()
        fig_reg = px.pie(regiao_data, names="regiao", values="volume_operacoes", hole=0.6, 
                         color_discrete_sequence=[COR_SECUNDARIA, COR_ATENCAO, COR_SUCESSO, COR_ALERTA, "#64748B"])
        st.plotly_chart(aplicar_layout_grafico(fig_reg), use_container_width=True)
    
    with col_r2:
        st.markdown("### Heatmap Regional (Milhões R$)")
        heat = df_filtrado.groupby(["regiao", df_filtrado["data_base"].dt.strftime("%m/%Y")])["volume_operacoes"].sum().reset_index()
        pivot = heat.pivot(index="regiao", columns="data_base", values="volume_operacoes") / 1_000_000
        fig_heat = px.imshow(pivot, aspect="auto", text_auto=".1f", color_continuous_scale="Blues")
        st.plotly_chart(aplicar_layout_grafico(fig_heat), use_container_width=True)
    
    st.markdown("---")
    st.markdown("### Top 3 Instituições por Estado")
    banco_por_uf = df_filtrado.groupby(["unidade_federacao", col_banco_nome])["numero_operacoes"].sum().reset_index()
    banco_por_uf = banco_por_uf.sort_values(["unidade_federacao", "numero_operacoes"], ascending=[True, False])
    top3_por_uf = banco_por_uf.groupby("unidade_federacao").head(3).reset_index(drop=True)
    top3_por_uf["ranking"] = top3_por_uf.groupby("unidade_federacao").cumcount() + 1
    top3_por_uf["exibicao"] = top3_por_uf.apply(lambda x: f"{x[col_banco_nome]} ({fmt_num(x['numero_operacoes'])})", axis=1)
    
    tabela_lideranca = top3_por_uf.pivot_table(index="unidade_federacao", columns="ranking", values="exibicao", aggfunc="first").reset_index()
    tabela_lideranca.columns = ["UF", "🥇 Líder", "🥈 Segundo", "🥉 Terceiro"]
    st.dataframe(tabela_lideranca, use_container_width=True, hide_index=True)
    
    # Boxplot de Outliers (NOVO)
    st.markdown("---")
    st.markdown("### Detecção de Outliers por Instituição")
    fig_box = boxplot_outliers(df_filtrado, col_banco_nome)
    st.plotly_chart(aplicar_layout_grafico(fig_box), use_container_width=True)
    
    # Treemap Cruzado (NOVO)
    st.markdown("### Ticket Médio por Região e Tipo de Banco")
    fig_treemap = treemap_cruzado(df_filtrado)
    st.plotly_chart(aplicar_layout_grafico(fig_treemap), use_container_width=True)

# ------------------------------------------------------------
# ABA 4: ANÁLISE AVANÇADA
# ------------------------------------------------------------
with aba_analise:
    st.markdown("### Análise de Dispersão por Segmento")
    dispersao = df_filtrado.groupby(col_banco_nome).agg({
        "numero_operacoes": "sum",
        "volume_operacoes": "sum",
        "tipo_banco": "first"
    }).reset_index()
    dispersao["ticket_medio"] = dispersao["volume_operacoes"] / dispersao["numero_operacoes"]
    dispersao = dispersao[dispersao["numero_operacoes"] > 100]
    
    fig_disp = px.scatter(dispersao, x="numero_operacoes", y="ticket_medio", 
                          color="tipo_banco", size="volume_operacoes",
                          hover_name=col_banco_nome,
                          labels={"numero_operacoes": "Operações", "ticket_medio": "Ticket Médio (R$)"},
                          color_discrete_map={"Banco Digital": "#10B981", "Banco Tradicional": "#2563EB", 
                                             "Banco de Investimento": "#D97706", "Demais Categorias": "#64748B"})
    fig_disp.update_layout(height=500)
    st.plotly_chart(aplicar_layout_grafico(fig_disp), use_container_width=True)
    
    # Ticket por tipo de banco
    st.markdown("### Comparativo: Banco Digital vs Tradicional")
    comparativo = df_filtrado.groupby("tipo_banco").agg({
        "numero_operacoes": "sum",
        "volume_operacoes": "sum"
    }).reset_index()
    comparativo["ticket_medio"] = comparativo["volume_operacoes"] / comparativo["numero_operacoes"]
    comparativo["participacao"] = (comparativo["numero_operacoes"] / comparativo["numero_operacoes"].sum() * 100).round(1)
    
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        fig_comp = px.bar(comparativo, x="tipo_banco", y="numero_operacoes", 
                          title="Operações por Tipo", color="tipo_banco", text_auto=".0f")
        st.plotly_chart(aplicar_layout_grafico(fig_comp), use_container_width=True)
    with col_c2:
        fig_ticket = px.bar(comparativo, x="tipo_banco", y="ticket_medio", 
                           title="Ticket Médio por Tipo", color="tipo_banco", text_auto=".2s")
        st.plotly_chart(aplicar_layout_grafico(fig_ticket), use_container_width=True)

# ------------------------------------------------------------
# ABA 5: CLUSTERIZAÇÃO
# ------------------------------------------------------------
with aba_cluster:
    st.markdown("### Clusterização de Bancos por Comportamento (K-Means)")
    st.caption("Agrupamento baseado em volume de operações e ticket médio")
    
    fig_cluster, cluster_data = clusterizar_bancos(df_filtrado, col_banco_nome)
    if fig_cluster:
        st.plotly_chart(aplicar_layout_grafico(fig_cluster), use_container_width=True)
        
        if cluster_data is not None:
            with st.expander("📋 Detalhamento dos Clusters"):
                st.dataframe(cluster_data[['nome_conglomerado_financeiro', 'numero_operacoes', 'ticket_medio', 'cluster_nome']].head(20), 
                            use_container_width=True, hide_index=True)
    else:
        st.info("Dados insuficientes para clusterização (necessário mais de 3 bancos com operações relevantes)")

# ------------------------------------------------------------
# ABA 6: INSIGHTS E EXPORTAÇÃO
# ------------------------------------------------------------
with aba_insights:
    st.markdown("### 🧠 Sumário Executivo Automatizado")
    
    crescimento_medio = evolucao["crescimento"].mean() if 'evolucao' in locals() else 0
    maior_regiao = regiao_data.sort_values("volume_operacoes", ascending=False).iloc[0]["regiao"] if 'regiao_data' in locals() else "N/A"
    banco_lider = market.iloc[0][col_banco_nome] if not market.empty else "N/A"
    participacao_lider = (market.iloc[0]["numero_operacoes"] / total_operacoes * 100) if not market.empty else 0
    
    col_ins1, col_ins2 = st.columns(2)
    with col_ins1:
        if crescimento_medio > 0:
            st.success(f"📈 **Aceleração de Carteira:** Expansão média de **+{crescimento_medio:.2f}%** ao mês.")
        else:
            st.warning(f"📉 **Fase de Desaceleração:** Retração média de **{crescimento_medio:.2f}%** ao mês.")
        
        st.markdown(f"📍 **Concentração Geográfica:** Região **{maior_regiao}** lidera em volume.")
        
    with col_ins2:
        st.markdown(f"🏆 **Liderança de Mercado:** **{banco_lider}** detém **{participacao_lider:.1f}%** do mercado.")
        if hhi > 2500:
            st.error("⚠️ **Alerta de Risco:** Mercado altamente concentrado.")
        else:
            st.success("✅ **Mercado Competitivo:** Baixo risco de concentração.")
    
    st.markdown("---")
    st.markdown("### Insights Avançados")
    
    # Insight de sazonalidade
    df_saz = df_filtrado.copy()
    df_saz['mes'] = df_filtrado['data_base'].dt.month
    mes_pico = df_saz.groupby('mes')['volume_operacoes'].sum().idxmax()
    nomes_meses = ['Janeiro', 'Fevereiro', 'Março', 'Abril', 'Maio', 'Junho', 
                   'Julho', 'Agosto', 'Setembro', 'Outubro', 'Novembro', 'Dezembro']
    st.info(f"📅 **Sazonalidade:** O mês de **{nomes_meses[mes_pico-1]}** concentra o maior volume de renegociações.")
    
    # Insight de correlação
    correlacao = df_filtrado[['numero_operacoes', 'volume_operacoes']].corr().iloc[0,1]
    st.info(f"📊 **Correlação:** Operações e volume têm correlação de **{correlacao:.3f}** - {'forte' if correlacao > 0.7 else 'moderada' if correlacao > 0.4 else 'fraca'}.")
    
    st.markdown("---")
    st.markdown("### 📥 Exportação de Dados")
    
    csv = df_filtrado.to_csv(index=False).encode("utf-8")
    col_exp1, col_exp2 = st.columns(2)
    with col_exp1:
        st.download_button(
            label="📥 Baixar Dataset Filtrado (CSV)",
            data=csv,
            file_name=f"BACEN_Desenrola_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    with col_exp2:
        relatorio = f"""RELATÓRIO DESENROLA BRASIL
Data: {datetime.now().strftime('%d/%m/%Y %H:%M')}

INDICADORES PRINCIPAIS:
- Volume Total: {fmt_brl(total_volume)}
- Operações: {fmt_num(total_operacoes)}
- Ticket Médio: {fmt_brl(ticket_medio)}
- Instituições: {fmt_num(num_bancos)}

CONCENTRAÇÃO:
- HHI: {hhi:.0f}
- Classificação: {classificacao}

LIDERANÇA:
- Banco Líder: {banco_lider}
- Região Líder: {maior_regiao}

Fonte: Banco Central do Brasil (SCR)
"""
        st.download_button("📝 Baixar Relatório Executivo", relatorio, f"relatorio_{datetime.now().strftime('%Y%m%d')}.txt", "text/plain")

# ============================================================
# COMPLIANCE FOOTER
# ============================================================
st.markdown("---")
st.markdown("""
<p style='text-align: center; color: #64748B; font-size: 0.75rem;'>
    📊 Desenrola Brasil Analytics | Fonte: Banco Central do Brasil (SCR)<br>
    Dashboard desenvolvido para portfólio de Análise de Dados | Dados de domínio público
</p>
""", unsafe_allow_html=True)
