import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import re
import warnings

# Apenas suprimir avisos de futuras alterações (não todos)
warnings.filterwarnings("ignore", category=FutureWarning)

# ============================================================
# CONFIGURAÇÃO DA PÁGINA
# ============================================================
st.set_page_config(
    page_title="Desenrola Brasil - Painel Executivo",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# PALETA DE CORES INSTITUCIONAL (Claro/Escuro)
# ============================================================
if "tema" not in st.session_state:
    st.session_state.tema = "claro"

# Atualiza template do Plotly conforme o tema
if st.session_state.tema == "claro":
    COR_FUNDO = "#F8FAFC"
    COR_CARD = "#FFFFFF"
    COR_TEXTO = "#1E293B"
    COR_BORDA = "#E2E8F0"
    COR_PRIMARIA = "#0F172A"
    COR_SECUNDARIA = "#2563EB"
    COR_SUCESSO = "#16A34A"
    COR_ALERTA = "#DC2626"
    COR_ATENCAO = "#D97706"
    PLOTLY_TEMPLATE = "plotly_white"
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
    PLOTLY_TEMPLATE = "plotly_dark"

# ============================================================
# CSS PERSONALIZADO
# ============================================================
st.markdown(f"""
<style>
    html, body, .stApp {{
        background-color: {COR_FUNDO};
        color: {COR_TEXTO};
        font-family: 'Inter', sans-serif;
    }}
    .block-container {{
        padding: 1rem 1.5rem;
    }}
    .kpi-card {{
        background: {COR_CARD};
        border-left: 4px solid {COR_SECUNDARIA};
        border-radius: 12px;
        padding: 0.8rem 1rem;
        margin-bottom: 0.5rem;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    }}
    .kpi-title {{
        font-size: 0.7rem;
        text-transform: uppercase;
        color: #64748B;
        font-weight: 600;
    }}
    .kpi-value {{
        font-size: 1.5rem;
        font-weight: 700;
        color: {COR_TEXTO};
        margin-top: 0.2rem;
    }}
    .badge {{
        padding: 2px 8px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.7rem;
        display: inline-block;
    }}
    .badge-low {{ background-color: rgba(22, 163, 74, 0.1); color: #16A34A; }}
    .badge-mid {{ background-color: rgba(217, 119, 6, 0.1); color: #D97706; }}
    .badge-high {{ background-color: rgba(220, 38, 38, 0.1); color: #DC2626; }}
</style>
""", unsafe_allow_html=True)

# ============================================================
# FUNÇÕES UTILITÁRIAS
# ============================================================
def fmt_brl(valor):
    if pd.isna(valor) or valor == 0:
        return "R$ 0"
    if valor >= 1_000_000_000:
        return f"R$ {valor/1_000_000_000:.1f}B".replace(".", ",")
    if valor >= 1_000_000:
        return f"R$ {valor/1_000_000:.1f}M".replace(".", ",")
    return f"R$ {valor:,.0f}".replace(",", ".")

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
    if any(x in nome for x in ["NUBANK", "INTER", "C6", "NEON", "ORIGINAL"]):
        return "Banco Digital"
    if any(x in nome for x in ["ITAU", "BRADESCO", "SANTANDER", "CAIXA", "BANCO DO BRASIL", "BB"]):
        return "Banco Tradicional"
    if any(x in nome for x in ["BTG"]):
        return "Banco de Investimento"
    return "Outras Instituições"

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
    total = df_sorted[coluna].sum()
    df_sorted['percentual_acumulado'] = (df_sorted[coluna].cumsum() / total) * 100
    return df_sorted

def aplicar_layout_padrao(fig, height=450):
    """Aplica layout padronizado a todos os gráficos"""
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=height,
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
# FUNÇÕES ANALÍTICAS (COM CACHE)
# ============================================================
@st.cache_data
def analise_sazonalidade(df):
    df_saz = df.copy()
    df_saz['mes'] = df_saz['data_base'].dt.month
    sazonal_mes = df_saz.groupby('mes')['volume_operacoes'].agg(['sum', 'mean', 'std']).reset_index()
    nomes_meses = ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun', 'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez']
    sazonal_mes['mes_nome'] = sazonal_mes['mes'].apply(lambda x: nomes_meses[x-1])
    fig = px.bar(sazonal_mes, x='mes_nome', y='mean', error_y='std',
                 title='Média de Volume por Mês (com variação)',
                 labels={'mes_nome': 'Mês', 'mean': 'Volume Médio (R$)'},
                 color='mean', color_continuous_scale='Blues')
    return aplicar_layout_padrao(fig, height=400)

@st.cache_data
def matriz_correlacao(df):
    cols = ['numero_operacoes', 'volume_operacoes']
    if 'ticket_medio' in df.columns:
        cols.append('ticket_medio')
    corr = df[cols].corr()
    fig = px.imshow(corr, text_auto=True, aspect='auto', zmin=-1, zmax=1,
                    title='Matriz de Correlação entre Indicadores',
                    color_continuous_scale='RdBu')
    return aplicar_layout_padrao(fig, height=400)

@st.cache_data
def boxplot_outliers(df, col_banco):
    top10 = df.groupby(col_banco)['numero_operacoes'].sum().nlargest(10).index
    df_top = df[df[col_banco].isin(top10)]
    fig = px.box(df_top, x=col_banco, y='numero_operacoes',
                 title='Distribuição de Operações por Instituição (Valores Atípicos)',
                 labels={col_banco: 'Instituição', 'numero_operacoes': 'Operações'},
                 color=col_banco)
    fig.update_layout(showlegend=False)
    return aplicar_layout_padrao(fig, height=450)

@st.cache_data
def treemap_cruzado(df):
    cruzado = df.groupby(['regiao', 'tipo_banco']).agg({
        'numero_operacoes': 'sum',
        'volume_operacoes': 'sum'
    }).reset_index()
    cruzado['ticket_medio'] = cruzado['volume_operacoes'] / cruzado['numero_operacoes']
    fig = px.treemap(cruzado, path=['regiao', 'tipo_banco'],
                     values='volume_operacoes', color='ticket_medio',
                     title='Ticket Médio por Região e Tipo de Banco',
                     color_continuous_scale='Blues',
                     hover_data={'ticket_medio': ':,.2f'})
    return aplicar_layout_padrao(fig, height=500)

@st.cache_data
def clusterizar_bancos(df, col_banco):
    cluster_data = df.groupby(col_banco).agg({
        'numero_operacoes': 'sum',
        'volume_operacoes': 'sum'
    }).reset_index()
    cluster_data['ticket_medio'] = cluster_data['volume_operacoes'] / cluster_data['numero_operacoes']
    cluster_data = cluster_data[cluster_data['numero_operacoes'] > 100]
    if len(cluster_data) < 3:
        return None, None
    scaler = StandardScaler()
    features = scaler.fit_transform(cluster_data[['numero_operacoes', 'ticket_medio']])
    n_clusters = min(3, len(cluster_data))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_data['cluster'] = kmeans.fit_predict(features)
    nomes_cluster = {0: 'Alto Volume / Baixo Ticket',
                     1: 'Baixo Volume / Alto Ticket',
                     2: 'Perfil Equilibrado'}
    cluster_data['cluster_nome'] = cluster_data['cluster'].map(nomes_cluster)
    fig = px.scatter(cluster_data, x='numero_operacoes', y='ticket_medio',
                     color='cluster_nome', size='volume_operacoes',
                     hover_name=col_banco,
                     title='Agrupamento de Instituições por Comportamento (K-Means)',
                     labels={'numero_operacoes': 'Operações', 'ticket_medio': 'Ticket Médio (R$)'})
    return aplicar_layout_padrao(fig, height=500), cluster_data

@st.cache_data
def comparativo_anual(df):
    df_yoy = df.copy()
    df_yoy['ano'] = df_yoy['data_base'].dt.year
    df_yoy['mes'] = df_yoy['data_base'].dt.month
    yoy_data = df_yoy.groupby(['ano', 'mes'])['volume_operacoes'].sum().reset_index()
    yoy_data = yoy_data[yoy_data['ano'] >= 2024]
    fig = px.line(yoy_data, x='mes', y='volume_operacoes', color='ano',
                  title='Comparativo Ano a Ano (YoY)',
                  labels={'mes': 'Mês', 'volume_operacoes': 'Volume (R$)', 'ano': 'Ano'},
                  markers=True)
    return aplicar_layout_padrao(fig, height=400)

# ============================================================
# CARREGAMENTO DE DADOS (COM TRATAMENTO DE EXCEÇÕES ESPECÍFICAS)
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
            df = df.dropna(subset=["volume_operacoes", "numero_operacoes"])
            return df
        except (ValueError, KeyError, TypeError) as e:
            continue
    return None

df = carregar_dados()
if df is None:
    st.error("Erro ao carregar os dados. Verifique se o arquivo 'dados_desenrola.csv' está presente e no formato correto.")
    st.stop()

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("### Painel de Controle")
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        if st.button("☀️ Claro", use_container_width=True):
            st.session_state.tema = "claro"
            st.rerun()
    with col_t2:
        if st.button("🌙 Escuro", use_container_width=True):
            st.session_state.tema = "escuro"
            st.rerun()
    st.markdown("---")
    tipos = sorted(df["tipo_desenrola"].unique())
    tipo = st.multiselect("Faixa do Programa", tipos, default=tipos)
    regioes = sorted(df["regiao"].unique())
    regiao = st.multiselect("Região", regioes, default=regioes)
    bancos = sorted(df["tipo_banco"].unique())
    banco = st.multiselect("Segmento Institucional", bancos, default=bancos)
    if st.button("🔄 Limpar Filtros", use_container_width=True):
        st.rerun()

df_filtrado = df[
    (df["tipo_desenrola"].isin(tipo)) &
    (df["regiao"].isin(regiao)) &
    (df["tipo_banco"].isin(banco))
]

if df_filtrado.empty:
    st.warning("Nenhum dado encontrado para a seleção atual. Ajuste os filtros.")
    st.stop()

# ============================================================
# CABEÇALHO EXECUTIVO
# ============================================================
col_header, col_date = st.columns([2, 1])
with col_header:
    st.title("Desenrola Brasil - Painel Executivo")
    st.caption("Monitoramento de renegociação de dívidas - Fonte: Banco Central do Brasil")
with col_date:
    st.markdown(f"<div style='text-align: right; color: gray; font-size: 0.75rem;'>Atualizado em {datetime.now().strftime('%d/%m/%Y')}</div>", unsafe_allow_html=True)

st.markdown("---")

# ============================================================
# INDICADORES PRINCIPAIS (KPIs)
# ============================================================
total_volume = df_filtrado["volume_operacoes"].sum()
total_operacoes = df_filtrado["numero_operacoes"].sum()
ticket_medio = total_volume / total_operacoes if total_operacoes > 0 else 0
num_bancos = df_filtrado["nome_conglomerado_financeiro"].nunique()

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f'<div class="kpi-card"><div class="kpi-title">Volume de Renegociação</div><div class="kpi-value">{fmt_brl(total_volume)}</div></div>', unsafe_allow_html=True)
with col2:
    st.markdown(f'<div class="kpi-card"><div class="kpi-title">Total de Contratos</div><div class="kpi-value">{fmt_num(total_operacoes)}</div></div>', unsafe_allow_html=True)
with col3:
    st.markdown(f'<div class="kpi-card"><div class="kpi-title">Ticket Médio</div><div class="kpi-value">{fmt_brl(ticket_medio)}</div></div>', unsafe_allow_html=True)
with col4:
    st.markdown(f'<div class="kpi-card"><div class="kpi-title">Instituições Atuantes</div><div class="kpi-value">{fmt_num(num_bancos)}</div></div>', unsafe_allow_html=True)

# ============================================================
# ABAS DE NAVEGAÇÃO (USANDO COLUNAS PARA MELHOR ESPAÇAMENTO)
# ============================================================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈 Evolução e Projeções",
    "🏦 Participação de Mercado",
    "🗺️ Distribuição Regional",
    "🧬 Análise Avançada",
    "🧠 Agrupamento de Instituições",
    "📊 Conclusões e Exportação"
])

# -------------------- TAB 1 --------------------
with tab1:
    evolucao = df_filtrado.groupby("data_base").agg({
        "volume_operacoes": "sum",
        "numero_operacoes": "sum"
    }).reset_index()
    evolucao["crescimento"] = evolucao["volume_operacoes"].pct_change() * 100
    evolucao["media_movel"] = evolucao["volume_operacoes"].rolling(window=3, min_periods=1).mean()

    col_esq, col_dir = st.columns([2, 1])
    with col_esq:
        st.markdown("### Histórico e Tendência")
        fig_ev = go.Figure()
        fig_ev.add_trace(go.Scatter(x=evolucao["data_base"], y=evolucao["volume_operacoes"], mode="lines+markers", name="Volume mensal", line=dict(color=COR_SECUNDARIA, width=2)))
        fig_ev.add_trace(go.Scatter(x=evolucao["data_base"], y=evolucao["media_movel"], mode="lines", name="Média móvel (3 meses)", line=dict(color=COR_ATENCAO, dash="dash")))
        st.plotly_chart(aplicar_layout_padrao(fig_ev, height=400), use_container_width=True)
    with col_dir:
        st.markdown("### Variação Mensal")
        cresc_tabela = evolucao[["data_base", "crescimento"]].dropna().tail(5).copy()
        cresc_tabela["data_base"] = cresc_tabela["data_base"].dt.strftime("%m/%Y")
        cresc_tabela["crescimento"] = cresc_tabela["crescimento"].apply(lambda x: f"{x:+.2f}%")
        cresc_tabela.columns = ["Mês", "Variação"]
        st.dataframe(cresc_tabela, use_container_width=True, hide_index=True)

    if len(evolucao) > 3:
        st.markdown("### Projeção Linear (Próximos 3 Meses)")
        modelo = evolucao.dropna().copy()
        modelo["indice"] = np.arange(len(modelo))
        lr = LinearRegression()
        lr.fit(modelo[["indice"]], modelo["volume_operacoes"])
        futuro = pd.DataFrame({"indice": np.arange(len(modelo), len(modelo) + 3)})
        previsao = lr.predict(futuro)
        datas_futuras = pd.date_range(evolucao["data_base"].max(), periods=4, freq="MS")[1:]
        fig_prev = go.Figure()
        fig_prev.add_trace(go.Scatter(x=evolucao["data_base"], y=evolucao["volume_operacoes"], name="Realizado", line=dict(color=COR_SECUNDARIA, width=2)))
        fig_prev.add_trace(go.Scatter(x=datas_futuras, y=previsao, name="Projeção", line=dict(color=COR_ATENCAO, dash="dot")))
        st.plotly_chart(aplicar_layout_padrao(fig_prev, height=400), use_container_width=True)

    st.markdown("### Sazonalidade")
    fig_saz = analise_sazonalidade(df_filtrado)
    st.plotly_chart(fig_saz, use_container_width=True)

    st.markdown("### Comparativo Ano a Ano")
    fig_yoy = comparativo_anual(df_filtrado)
    st.plotly_chart(fig_yoy, use_container_width=True)

# -------------------- TAB 2 --------------------
with tab2:
    col_banco = "nome_conglomerado_financeiro"
    market = df_filtrado.groupby(col_banco)["numero_operacoes"].sum().sort_values(ascending=False).reset_index()

    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.markdown("### Índice de Concentração (HHI)")
        hhi = calcular_hhi(market, "numero_operacoes")
        classificacao, classe_css = interpretar_hhi(hhi)
        st.markdown(f"""
        <div style="background:{COR_CARD}; border:1px solid {COR_BORDA}; border-radius:12px; padding:1.2rem;">
            <p style="margin:0; color:gray;">Índice Herfindahl-Hirschman</p>
            <h2 style="margin:5px 0 10px; font-size:2rem;">{hhi:.0f}</h2>
            <span class="badge {classe_css}">{classificacao}</span>
        </div>
        """, unsafe_allow_html=True)
    with col_m2:
        st.markdown("### Análise de Pareto (80/20)")
        pareto_data = calcular_pareto(market.head(10), "numero_operacoes")
        fig_pareto = go.Figure()
        fig_pareto.add_trace(go.Bar(x=pareto_data[col_banco], y=pareto_data["numero_operacoes"], name="Contratos", marker_color=COR_SECUNDARIA))
        fig_pareto.add_trace(go.Scatter(x=pareto_data[col_banco], y=pareto_data["percentual_acumulado"], name="% Acumulado", yaxis="y2", mode="lines+markers", line=dict(color=COR_ALERTA, width=2)))
        fig_pareto.update_layout(yaxis2=dict(overlaying="y", side="right", range=[0, 105], showgrid=False))
        st.plotly_chart(aplicar_layout_padrao(fig_pareto, height=400), use_container_width=True)

    st.markdown("### Ranking com Participação")
    ranking = market.head(15).copy()
    total = ranking["numero_operacoes"].sum()
    ranking["Participação (%)"] = (ranking["numero_operacoes"] / total * 100).round(1)
    ranking["% Acumulado"] = ranking["Participação (%)"].cumsum().round(1)
    ranking.columns = ["Instituição", "Operações", "% Individual", "% Acumulado"]
    st.dataframe(ranking, use_container_width=True, hide_index=True)

    st.markdown("### Matriz de Correlação")
    fig_corr = matriz_correlacao(df_filtrado)
    st.plotly_chart(fig_corr, use_container_width=True)

# -------------------- TAB 3 --------------------
with tab3:
    col_r1, col_r2 = st.columns(2)
    with col_r1:
        st.markdown("### Volume por Região")
        reg_data = df_filtrado.groupby("regiao")["volume_operacoes"].sum().reset_index()
        fig_pie = px.pie(reg_data, names="regiao", values="volume_operacoes", hole=0.5,
                         title="Distribuição Regional do Volume",
                         color_discrete_sequence=[COR_SECUNDARIA, COR_ATENCAO, COR_SUCESSO, COR_ALERTA, "#64748B"])
        st.plotly_chart(aplicar_layout_padrao(fig_pie, height=400), use_container_width=True)
    with col_r2:
        st.markdown("### Evolução Regional (Heatmap)")
        heat = df_filtrado.groupby(["regiao", df_filtrado["data_base"].dt.strftime("%m/%Y")])["volume_operacoes"].sum().reset_index()
        pivot = heat.pivot(index="regiao", columns="data_base", values="volume_operacoes") / 1_000_000
        fig_heat = px.imshow(pivot, aspect="auto", text_auto=".1f", color_continuous_scale="Blues")
        fig_heat.update_layout(height=400)
        st.plotly_chart(aplicar_layout_padrao(fig_heat, height=400), use_container_width=True)

    st.markdown("### Top 3 Instituições por Estado")
    uf_banco = df_filtrado.groupby(["unidade_federacao", col_banco])["numero_operacoes"].sum().reset_index()
    uf_banco = uf_banco.sort_values(["unidade_federacao", "numero_operacoes"], ascending=[True, False])
    top3_uf = uf_banco.groupby("unidade_federacao").head(3).reset_index(drop=True)
    top3_uf["rank"] = top3_uf.groupby("unidade_federacao").cumcount() + 1
    top3_uf["display"] = top3_uf.apply(lambda x: f"{x[col_banco]} ({fmt_num(x['numero_operacoes'])})", axis=1)
    pivot_uf = top3_uf.pivot_table(index="unidade_federacao", columns="rank", values="display", aggfunc="first").reset_index()
    pivot_uf.columns = ["UF", "1º Lugar", "2º Lugar", "3º Lugar"]
    st.dataframe(pivot_uf, use_container_width=True, hide_index=True)

    st.markdown("### Ticket Médio Cruzado")
    fig_tree = treemap_cruzado(df_filtrado)
    st.plotly_chart(fig_tree, use_container_width=True)

    st.markdown("### Detecção de Valores Atípicos")
    fig_out = boxplot_outliers(df_filtrado, col_banco)
    st.plotly_chart(fig_out, use_container_width=True)

# -------------------- TAB 4 --------------------
with tab4:
    st.markdown("### Dispersão por Segmento")
    dispersao = df_filtrado.groupby(col_banco).agg({
        "numero_operacoes": "sum",
        "volume_operacoes": "sum",
        "tipo_banco": "first"
    }).reset_index()
    dispersao["ticket_medio"] = dispersao["volume_operacoes"] / dispersao["numero_operacoes"]
    dispersao = dispersao[dispersao["numero_operacoes"] > 100]
    fig_disp = px.scatter(dispersao, x="numero_operacoes", y="ticket_medio",
                          color="tipo_banco", size="volume_operacoes", hover_name=col_banco,
                          labels={"numero_operacoes": "Operações", "ticket_medio": "Ticket Médio (R$)"},
                          color_discrete_map={"Banco Digital": "#10B981", "Banco Tradicional": "#2563EB",
                                              "Banco de Investimento": "#D97706", "Outras Instituições": "#64748B"})
    st.plotly_chart(aplicar_layout_padrao(fig_disp, height=500), use_container_width=True)

    st.markdown("### Comparativo: Digital vs Tradicional")
    comp = df_filtrado.groupby("tipo_banco").agg({
        "numero_operacoes": "sum",
        "volume_operacoes": "sum"
    }).reset_index()
    comp["ticket_medio"] = comp["volume_operacoes"] / comp["numero_operacoes"]
    comp["participacao"] = comp["numero_operacoes"] / comp["numero_operacoes"].sum() * 100

    col_c1, col_c2 = st.columns(2)
    with col_c1:
        fig_bar = px.bar(comp, x="tipo_banco", y="numero_operacoes", title="Operações por Segmento", color="tipo_banco", text_auto=".0f")
        st.plotly_chart(aplicar_layout_padrao(fig_bar, height=400), use_container_width=True)
    with col_c2:
        fig_tick = px.bar(comp, x="tipo_banco", y="ticket_medio", title="Ticket Médio por Segmento", color="tipo_banco", text_auto=".2s")
        st.plotly_chart(aplicar_layout_padrao(fig_tick, height=400), use_container_width=True)

# -------------------- TAB 5 --------------------
with tab5:
    st.markdown("### Agrupamento de Instituições por Comportamento")
    st.caption("Algoritmo K-Means agrupa bancos com perfil semelhante (volume x ticket médio)")
    fig_cluster, cluster_data = clusterizar_bancos(df_filtrado, col_banco)
    if fig_cluster:
        st.plotly_chart(fig_cluster, use_container_width=True)
        if cluster_data is not None:
            with st.expander("Detalhes dos Grupos"):
                st.dataframe(cluster_data[[col_banco, 'numero_operacoes', 'ticket_medio', 'cluster_nome']].head(20),
                            use_container_width=True, hide_index=True)
    else:
        st.info("Dados insuficientes para realizar o agrupamento (necessário mais de 2 instituições com operações relevantes).")

# -------------------- TAB 6 --------------------
with tab6:
    st.markdown("### Principais Conclusões")
    crescimento_medio = evolucao["crescimento"].mean() if 'evolucao' in locals() else 0
    regiao_lider = reg_data.sort_values("volume_operacoes", ascending=False).iloc[0]["regiao"] if 'reg_data' in locals() else "N/A"
    banco_lider = market.iloc[0][col_banco] if not market.empty else "N/A"
    participacao_lider = (market.iloc[0]["numero_operacoes"] / total_operacoes * 100) if not market.empty else 0

    col_i1, col_i2 = st.columns(2)
    with col_i1:
        if crescimento_medio > 0:
            st.success(f"📈 **Crescimento Sustentado:** Média mensal de **+{crescimento_medio:.2f}%**.")
        else:
            st.warning(f"📉 **Desaceleração:** Retração média de **{crescimento_medio:.2f}%** ao mês.")
        st.info(f"📍 **Região de Destaque:** **{regiao_lider}** concentra o maior volume.")
    with col_i2:
        st.info(f"🏆 **Liderança de Mercado:** **{banco_lider}** detém **{participacao_lider:.1f}%** dos contratos.")
        if hhi > 2500:
            st.error("⚠️ **Alerta de Concentração:** Mercado com alto risco de oligopólio.")
        else:
            st.success("✅ **Mercado Competitivo:** Baixo risco de concentração.")

    # Sazonalidade
    df_saz = df_filtrado.copy()
    df_saz['mes'] = df_filtrado['data_base'].dt.month
    mes_pico = df_saz.groupby('mes')['volume_operacoes'].sum().idxmax()
    nomes_meses = ['Janeiro', 'Fevereiro', 'Março', 'Abril', 'Maio', 'Junho',
                   'Julho', 'Agosto', 'Setembro', 'Outubro', 'Novembro', 'Dezembro']
    st.info(f"📅 **Sazonalidade:** O mês de **{nomes_meses[mes_pico-1]}** registra o maior volume de renegociações.")

    # Correlação
    corr = df_filtrado[['numero_operacoes', 'volume_operacoes']].corr().iloc[0,1]
    st.info(f"📊 **Correlação:** A relação entre operações e volume é **{corr:.3f}** – "
            f"{'forte' if corr > 0.7 else 'moderada' if corr > 0.4 else 'fraca'}.")

    st.markdown("---")
    st.markdown("### Exportação de Dados")
    csv = df_filtrado.to_csv(index=False).encode("utf-8")
    col_exp1, col_exp2 = st.columns(2)
    with col_exp1:
        st.download_button("📥 Baixar Dados Filtrados (CSV)", data=csv,
                           file_name=f"desenrola_filtrado_{datetime.now().strftime('%Y%m%d')}.csv",
                           mime="text/csv", use_container_width=True)
    with col_exp2:
        relatorio = f"""RELATÓRIO DESENROLA BRASIL - {datetime.now().strftime('%d/%m/%Y %H:%M')}
Volume Total: {fmt_brl(total_volume)}
Operações: {fmt_num(total_operacoes)}
Ticket Médio: {fmt_brl(ticket_medio)}
Instituições: {fmt_num(num_bancos)}
HHI: {hhi:.0f} - {classificacao}
Banco Líder: {banco_lider} ({participacao_lider:.1f}%)
Região Líder: {regiao_lider}
Crescimento Mensal Médio: {crescimento_medio:+.2f}%
Sazonalidade: Pico em {nomes_meses[mes_pico-1]}
Fonte: Banco Central do Brasil (SCR)
"""
        st.download_button("📝 Baixar Relatório Executivo (TXT)", data=relatorio,
                           file_name=f"relatorio_desenrola_{datetime.now().strftime('%Y%m%d')}.txt",
                           mime="text/plain", use_container_width=True)

# ============================================================
# RODAPÉ
# ============================================================
st.markdown("---")
st.markdown("""
<p style='text-align: center; color: #64748B; font-size: 0.7rem;'>
    Dashboard desenvolvido para portfólio de Análise de Dados.<br>
    Fonte: Banco Central do Brasil – Sistema de Informações de Crédito (SCR).
</p>
""", unsafe_allow_html=True)
