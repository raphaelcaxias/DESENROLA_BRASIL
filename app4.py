import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from datetime import datetime
import re

# ============================================================
# CONFIGURAÇÃO DA PÁGINA
# ============================================================

st.set_page_config(
    page_title="Desenrola Brasil Analytics",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# TEMA
# ============================================================

if "tema" not in st.session_state:
    st.session_state.tema = "claro"

if st.session_state.tema == "claro":
    COR_FUNDO = "#F8FAFC"
    COR_CARD = "#FFFFFF"
    COR_TEXTO = "#0F172A"
    COR_BORDA = "#E2E8F0"
    COR_PRIMARIA = "#10B981"
    COR_SECUNDARIA = "#2563EB"
    COR_ALERTA = "#F59E0B"
else:
    COR_FUNDO = "#0F172A"
    COR_CARD = "#1E293B"
    COR_TEXTO = "#F8FAFC"
    COR_BORDA = "#334155"
    COR_PRIMARIA = "#34D399"
    COR_SECUNDARIA = "#60A5FA"
    COR_ALERTA = "#FBBF24"

# ============================================================
# CSS
# ============================================================

st.markdown(f"""
<style>
html, body, .stApp {{
    background-color: {COR_FUNDO};
    color: {COR_TEXTO};
}}

.block-container {{
    padding-top: 1rem;
}}

.kpi-card {{
    background: {COR_CARD};
    border: 1px solid {COR_BORDA};
    border-radius: 16px;
    padding: 1rem;
    margin-bottom: 1rem;
}}

.kpi-title {{
    font-size: 0.8rem;
    color: gray;
}}

.kpi-value {{
    font-size: 1.8rem;
    font-weight: bold;
}}

.section {{
    background: {COR_CARD};
    padding: 1rem;
    border-radius: 16px;
    border: 1px solid {COR_BORDA};
    margin-bottom: 1rem;
}}
</style>
""", unsafe_allow_html=True)

# ============================================================
# FUNÇÕES
# ============================================================

def fmt_brl(valor):
    if pd.isna(valor) or valor == 0:
        return "R$ 0"
    if valor >= 1_000_000_000:
        return f"R$ {valor/1_000_000_000:.2f}B".replace(".", ",")
    if valor >= 1_000_000:
        return f"R$ {valor/1_000_000:.2f}M".replace(".", ",")
    return f"R$ {valor:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

def fmt_num(valor):
    if pd.isna(valor):
        return "0"
    return f"{int(valor):,}".replace(",", ".")

def classificar_banco(nome):
    """Classificação melhorada - corrige PRUDENCIAL e detecta mais bancos"""
    nome = str(nome).upper().strip()
    # Remove sufixo PRUDENCIAL para agrupar corretamente
    nome = re.sub(r'\s*-\s*PRUDENCIAL$', '', nome)
    
    if any(x in nome for x in ["NUBANK", "INTER", "C6"]):
        return "Banco Digital"
    if any(x in nome for x in ["ITAU", "BRADESCO", "SANTANDER", "CAIXA", "BANCO DO BRASIL", "BB"]):
        return "Banco Tradicional"
    if any(x in nome for x in ["BTG", "BMG", "BANRISUL", "VOTORANTIM"]):
        return "Outros"
    return "Outros"

def agrupar_regiao(uf):
    mapa = {
        "NORTE": ["AC", "AM", "AP", "PA", "RO", "RR", "TO"],
        "NORDESTE": ["AL", "BA", "CE", "MA", "PB", "PE", "PI", "RN", "SE"],
        "CENTRO-OESTE": ["DF", "GO", "MS", "MT"],
        "SUDESTE": ["ES", "MG", "RJ", "SP"],
        "SUL": ["PR", "RS", "SC"]
    }
    for regiao, estados in mapa.items():
        if uf in estados:
            return regiao
    return "Outros"

def calcular_hhi(df, coluna):
    total = df[coluna].sum()
    if total == 0:
        return 0
    shares = (df[coluna] / total) ** 2
    return shares.sum() * 10000

def interpretar_hhi(hhi):
    if hhi < 1500:
        return "Baixa concentração", "🟢"
    elif hhi < 2500:
        return "Concentração moderada", "🟡"
    return "Alta concentração", "🔴"

def calcular_pareto(df, coluna, percentual=80):
    """Calcula o percentual acumulado para análise de Pareto"""
    df_sorted = df.sort_values(coluna, ascending=False).reset_index(drop=True)
    total = df_sorted[coluna].sum()
    df_sorted['percentual_acumulado'] = (df_sorted[coluna].cumsum() / total) * 100
    return df_sorted

# ============================================================
# CARREGAMENTO
# ============================================================

@st.cache_data(ttl=3600)
def carregar_dados():
    encodings = ["utf-8", "latin1", "cp1252"]
    for enc in encodings:
        try:
            df = pd.read_csv(
                "dados_desenrola.csv",
                sep=";",
                encoding=enc,
                low_memory=False
            )
            df.columns = df.columns.str.lower().str.strip()
            
            df["numero_operacoes"] = pd.to_numeric(
                df["numero_operacoes"]
                .astype(str)
                .str.replace(".", "", regex=False)
                .str.replace(",", ".", regex=False),
                errors="coerce"
            )
            
            df["volume_operacoes"] = pd.to_numeric(
                df["volume_operacoes"]
                .astype(str)
                .str.replace(".", "", regex=False)
                .str.replace(",", ".", regex=False),
                errors="coerce"
            )
            
            df["data_base"] = pd.to_datetime(
                df["data_base"].astype(str),
                format="%Y%m",
                errors="coerce"
            )
            
            df["tipo_banco"] = df["nome_conglomerado_financeiro"].apply(classificar_banco)
            df["regiao"] = df["unidade_federacao"].apply(agrupar_regiao)
            df = df.dropna()
            return df
        except Exception as e:
            continue
    return None

df = carregar_dados()

if df is None:
    st.error("❌ Erro ao carregar o arquivo CSV. Verifique se o arquivo existe.")
    st.stop()

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.title("🎛️ Filtros")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("☀️ Tema Claro", use_container_width=True):
            st.session_state.tema = "claro"
            st.rerun()
    with col2:
        if st.button("🌙 Tema Escuro", use_container_width=True):
            st.session_state.tema = "escuro"
            st.rerun()
    
    st.markdown("---")
    
    tipos = sorted(df["tipo_desenrola"].unique())
    tipo = st.multiselect("Tipo do Programa", tipos, default=tipos)
    
    regioes = sorted(df["regiao"].unique())
    regiao = st.multiselect("Região", regioes, default=regioes)
    
    bancos = sorted(df["tipo_banco"].unique())
    banco = st.multiselect("Tipo de Banco", bancos, default=bancos)
    
    if st.button("🔄 Resetar Filtros", use_container_width=True):
        st.rerun()

# ============================================================
# FILTROS
# ============================================================

df_filtrado = df[
    (df["tipo_desenrola"].isin(tipo)) &
    (df["regiao"].isin(regiao)) &
    (df["tipo_banco"].isin(banco))
]

if len(df_filtrado) == 0:
    st.warning("⚠️ Nenhum dado encontrado com os filtros selecionados. Ajuste os filtros.")
    st.stop()

# ============================================================
# HEADER
# ============================================================

st.title("🏦 Desenrola Brasil Analytics")
st.caption("Dashboard avançado de renegociação de dívidas - Dados do Banco Central")

# ============================================================
# KPIs
# ============================================================

total_volume = df_filtrado["volume_operacoes"].sum()
total_operacoes = df_filtrado["numero_operacoes"].sum()
ticket_medio = total_volume / total_operacoes if total_operacoes > 0 else 0
num_bancos = df_filtrado["nome_conglomerado_financeiro"].nunique()

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">💰 Volume Total</div>
        <div class="kpi-value">{fmt_brl(total_volume)}</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">📋 Operações</div>
        <div class="kpi-value">{fmt_num(total_operacoes)}</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">🎫 Ticket Médio</div>
        <div class="kpi-value">{fmt_brl(ticket_medio)}</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">🏦 Instituições</div>
        <div class="kpi-value">{fmt_num(num_bancos)}</div>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# EVOLUÇÃO TEMPORAL
# ============================================================

st.subheader("📈 Evolução Temporal")

evolucao = (
    df_filtrado.groupby("data_base")
    .agg({
        "volume_operacoes": "sum",
        "numero_operacoes": "sum"
    })
    .reset_index()
)

evolucao["crescimento"] = evolucao["volume_operacoes"].pct_change() * 100
evolucao["media_movel"] = evolucao["volume_operacoes"].rolling(3).mean()

fig = go.Figure()
fig.add_trace(go.Scatter(x=evolucao["data_base"], y=evolucao["volume_operacoes"], mode="lines+markers", name="Volume"))
fig.add_trace(go.Scatter(x=evolucao["data_base"], y=evolucao["media_movel"], mode="lines", name="Média móvel (3 meses)"))
fig.update_layout(height=450, paper_bgcolor=COR_CARD, plot_bgcolor=COR_CARD, font=dict(color=COR_TEXTO), hovermode="x unified")
st.plotly_chart(fig, use_container_width=True)

# Tabela de crescimento mensal
st.markdown("#### 📊 Crescimento Mensal (%)")
cresc_tabela = evolucao[["data_base", "crescimento"]].dropna().tail(6).copy()
cresc_tabela["data_base"] = cresc_tabela["data_base"].dt.strftime("%b/%Y")
cresc_tabela["crescimento"] = cresc_tabela["crescimento"].apply(lambda x: f"{x:+.1f}%")
cresc_tabela.columns = ["Mês", "Variação"]
st.dataframe(cresc_tabela, use_container_width=True, hide_index=True)

# ============================================================
# EVOLUÇÃO POR TIPO (NOVO)
# ============================================================

st.subheader("📊 Evolução por Tipo do Programa")

evolucao_tipo = df_filtrado.groupby(["data_base", "tipo_desenrola"])["volume_operacoes"].sum().reset_index()
evolucao_tipo = evolucao_tipo.sort_values("data_base")

fig_tipo = go.Figure()
cores_tipo = {1: COR_PRIMARIA, 2: COR_SECUNDARIA, 3: COR_ALERTA}
for tipo_valor in evolucao_tipo["tipo_desenrola"].unique():
    dados = evolucao_tipo[evolucao_tipo["tipo_desenrola"] == tipo_valor]
    nome_tipo = {1: "Tipo 1 (Faixa 1 PF)", 2: "Tipo 2 (Faixa 2 PF)", 3: "Tipo 3 (Pequenos Negócios)"}.get(tipo_valor, f"Tipo {tipo_valor}")
    fig_tipo.add_trace(go.Scatter(x=dados["data_base"], y=dados["volume_operacoes"], mode="lines+markers", name=nome_tipo, line=dict(color=cores_tipo.get(tipo_valor, COR_SECUNDARIA))))

fig_tipo.update_layout(height=450, paper_bgcolor=COR_CARD, plot_bgcolor=COR_CARD, font=dict(color=COR_TEXTO), hovermode="x unified")
st.plotly_chart(fig_tipo, use_container_width=True)

# ============================================================
# PROJEÇÃO
# ============================================================

st.subheader("🔮 Projeção (Próximos 3 Meses)")

if len(evolucao) > 3:
    modelo = evolucao.dropna().copy()
    modelo["indice"] = np.arange(len(modelo))
    X = modelo[["indice"]]
    y = modelo["volume_operacoes"]
    
    lr = LinearRegression()
    lr.fit(X, y)
    
    futuro = pd.DataFrame({"indice": np.arange(len(modelo), len(modelo) + 3)})
    previsao = lr.predict(futuro)
    datas_futuras = pd.date_range(evolucao["data_base"].max(), periods=4, freq="MS")[1:]
    
    fig_prev = go.Figure()
    fig_prev.add_trace(go.Scatter(x=evolucao["data_base"], y=evolucao["volume_operacoes"], mode="lines+markers", name="Histórico"))
    fig_prev.add_trace(go.Scatter(x=datas_futuras, y=previsao, mode="lines+markers", name="Previsão", line=dict(dash="dot")))
    fig_prev.update_layout(height=450, paper_bgcolor=COR_CARD, plot_bgcolor=COR_CARD, font=dict(color=COR_TEXTO))
    st.plotly_chart(fig_prev, use_container_width=True)

# ============================================================
# TOP 3 BANCOS POR ESTADO (NOVO)
# ============================================================

st.subheader("🗺️ Top 3 Bancos por Estado")

banco_por_uf = df_filtrado.groupby(["unidade_federacao", "nome_conglomerado_financeiro"])["numero_operacoes"].sum().reset_index()
banco_por_uf = banco_por_uf.sort_values(["unidade_federacao", "numero_operacoes"], ascending=[True, False])
top3_por_uf = banco_por_uf.groupby("unidade_federacao").head(3).reset_index(drop=True)
top3_por_uf["ranking"] = top3_por_uf.groupby("unidade_federacao").cumcount() + 1
top3_por_uf["exibicao"] = top3_por_uf.apply(lambda x: f"{x['ranking']}º - {x['nome_conglomerado_financeiro']} ({fmt_num(x['numero_operacoes'])})", axis=1)

tabela_lideranca = top3_por_uf.pivot_table(index="unidade_federacao", columns="ranking", values="exibicao", aggfunc="first").reset_index()
tabela_lideranca.columns = ["UF", "🥇 1º Lugar", "🥈 2º Lugar", "🥉 3º Lugar"]
st.dataframe(tabela_lideranca, use_container_width=True, hide_index=True)

# ============================================================
# DISPERSÃO (OPERAÇÕES x TICKET MÉDIO) (NOVO)
# ============================================================

st.subheader("📊 Dispersão: Operações vs Ticket Médio por Banco")

dispersao = df_filtrado.groupby("nome_conglomerado_financeiro").agg({
    "numero_operacoes": "sum",
    "volume_operacoes": "sum"
}).reset_index()
dispersao["ticket_medio"] = dispersao["volume_operacoes"] / dispersao["numero_operacoes"]
dispersao = dispersao[dispersao["numero_operacoes"] > 1000]
dispersao["tipo_banco"] = dispersao["nome_conglomerado_financeiro"].apply(classificar_banco)

if len(dispersao) > 1:
    fig_disp = px.scatter(dispersao, x="numero_operacoes", y="ticket_medio", 
                          color="tipo_banco", size="numero_operacoes", hover_name="nome_conglomerado_financeiro",
                          title="Relação entre Volume de Operações e Ticket Médio",
                          labels={"numero_operacoes": "Operações", "ticket_medio": "Ticket Médio (R$)"},
                          color_discrete_map={"Banco Digital": COR_PRIMARIA, "Banco Tradicional": COR_SECUNDARIA, "Outros": COR_ALERTA})
    fig_disp.update_layout(height=450, paper_bgcolor=COR_CARD, plot_bgcolor=COR_CARD, font=dict(color=COR_TEXTO))
    st.plotly_chart(fig_disp, use_container_width=True)
    
    correlacao = np.corrcoef(dispersao["numero_operacoes"].values, dispersao["ticket_medio"].values)[0, 1]
    st.caption(f"📊 Correlação entre Operações e Ticket Médio: **{correlacao:.2f}** - {'Correlação positiva' if correlacao > 0.3 else 'Correlação negativa' if correlacao < -0.3 else 'Baixa correlação'}")
else:
    st.info("Dados insuficientes para análise de dispersão (menos de 2 bancos com mais de 1000 operações)")

# ============================================================
# MARKET SHARE
# ============================================================

st.subheader("🏦 Market Share")

market = df_filtrado.groupby("nome_conglomerado_financeiro")["numero_operacoes"].sum().sort_values(ascending=False).head(10).reset_index()
fig_market = px.bar(market, x="numero_operacoes", y="nome_conglomerado_financeiro", orientation="h", color="numero_operacoes")
fig_market.update_layout(height=500, paper_bgcolor=COR_CARD, plot_bgcolor=COR_CARD, font=dict(color=COR_TEXTO))
st.plotly_chart(fig_market, use_container_width=True)

# ============================================================
# ANÁLISE DE PARETO (80/20) (NOVO)
# ============================================================

st.subheader("📊 Análise de Pareto (80/20)")

pareto_data = calcular_pareto(market, "numero_operacoes")
pareto_data["percentual_acumulado"] = pareto_data["percentual_acumulado"].round(1)

fig_pareto = go.Figure()
fig_pareto.add_trace(go.Bar(x=pareto_data["nome_conglomerado_financeiro"], y=pareto_data["numero_operacoes"], name="Operações", marker_color=COR_PRIMARIA))
fig_pareto.add_trace(go.Scatter(x=pareto_data["nome_conglomerado_financeiro"], y=pareto_data["percentual_acumulado"], name="% Acumulado", yaxis="y2", mode="lines+markers", line=dict(color=COR_ALERTA, width=3)))
fig_pareto.update_layout(height=450, yaxis2=dict(title="Percentual Acumulado (%)", overlaying="y", side="right"), paper_bgcolor=COR_CARD, plot_bgcolor=COR_CARD, font=dict(color=COR_TEXTO))
st.plotly_chart(fig_pareto, use_container_width=True)

# ============================================================
# HHI
# ============================================================

st.subheader("📊 Índice HHI (Concentração de Mercado)")

hhi = calcular_hhi(market, "numero_operacoes")
classificacao, cor = interpretar_hhi(hhi)

col1, col2 = st.columns(2)
with col1:
    st.metric("HHI", f"{hhi:.0f}")
with col2:
    st.metric("Classificação", f"{cor} {classificacao}")

# ============================================================
# REGIÕES
# ============================================================

st.subheader("🗺️ Volume por Região")

regiao_data = df_filtrado.groupby("regiao")["volume_operacoes"].sum().reset_index()
fig_regiao = px.pie(regiao_data, names="regiao", values="volume_operacoes", hole=0.5)
fig_regiao.update_layout(height=450, paper_bgcolor=COR_CARD, font=dict(color=COR_TEXTO))
st.plotly_chart(fig_regiao, use_container_width=True)

# ============================================================
# TICKET POR ESTADO
# ============================================================

st.subheader("🎫 Ticket Médio por Estado (Top 10)")

ticket = df_filtrado.groupby("unidade_federacao").agg({"volume_operacoes": "sum", "numero_operacoes": "sum"}).reset_index()
ticket["ticket_medio"] = ticket["volume_operacoes"] / ticket["numero_operacoes"]
ticket = ticket.sort_values("ticket_medio", ascending=False).head(10)

fig_ticket = px.bar(ticket, x="unidade_federacao", y="ticket_medio", color="ticket_medio", text_auto=".2s")
fig_ticket.update_layout(height=450, paper_bgcolor=COR_CARD, plot_bgcolor=COR_CARD, font=dict(color=COR_TEXTO))
st.plotly_chart(fig_ticket, use_container_width=True)

# ============================================================
# COMPARATIVO DIGITAL vs TRADICIONAL (NOVO)
# ============================================================

st.subheader("📱 Banco Digital vs Banco Tradicional")

comparativo = df_filtrado.groupby("tipo_banco").agg({
    "numero_operacoes": "sum",
    "volume_operacoes": "sum"
}).reset_index()
comparativo["ticket_medio"] = comparativo["volume_operacoes"] / comparativo["numero_operacoes"]

col1, col2 = st.columns(2)
with col1:
    fig_comp_bar = px.bar(comparativo, x="tipo_banco", y="numero_operacoes", title="Operações por Tipo de Banco", color="tipo_banco", text_auto=".0f")
    fig_comp_bar.update_layout(height=400, paper_bgcolor=COR_CARD, plot_bgcolor=COR_CARD, font=dict(color=COR_TEXTO))
    st.plotly_chart(fig_comp_bar, use_container_width=True)
with col2:
    fig_comp_ticket = px.bar(comparativo, x="tipo_banco", y="ticket_medio", title="Ticket Médio por Tipo de Banco", color="tipo_banco", text_auto=".2s")
    fig_comp_ticket.update_layout(height=400, paper_bgcolor=COR_CARD, plot_bgcolor=COR_CARD, font=dict(color=COR_TEXTO))
    st.plotly_chart(fig_comp_ticket, use_container_width=True)

# ============================================================
# HEATMAP
# ============================================================

st.subheader("🔥 Heatmap Regional Temporal")

heat = df_filtrado.groupby(["regiao", df_filtrado["data_base"].dt.strftime("%Y-%m")])["volume_operacoes"].sum().reset_index()
pivot = heat.pivot(index="regiao", columns="data_base", values="volume_operacoes")

fig_heat = px.imshow(pivot, aspect="auto", text_auto=True, color_continuous_scale="Blues")
fig_heat.update_layout(height=450, paper_bgcolor=COR_CARD, font=dict(color=COR_TEXTO))
st.plotly_chart(fig_heat, use_container_width=True)

# ============================================================
# INSIGHTS AUTOMÁTICOS
# ============================================================

st.subheader("🧠 Insights Automáticos")

crescimento_medio = evolucao["crescimento"].mean()
maior_regiao = regiao_data.sort_values("volume_operacoes", ascending=False).iloc[0]["regiao"]
banco_lider = market.iloc[0]["nome_conglomerado_financeiro"]
participacao_lider = (market.iloc[0]["numero_operacoes"] / total_operacoes * 100)

if crescimento_medio > 0:
    st.success(f"📈 O programa apresenta **crescimento médio de {crescimento_medio:.2f}% ao mês**.")
else:
    st.warning(f"📉 O programa apresenta **retração média de {abs(crescimento_medio):.2f}% ao mês**.")

st.info(f"📍 A **região {maior_regiao}** lidera em volume de renegociação.")
st.info(f"🏆 A instituição **{banco_lider}** lidera o mercado com **{participacao_lider:.1f}%** de participação.")

if hhi > 2500:
    st.error("⚠️ **Mercado altamente concentrado** - alto risco de concentração.")

# ============================================================
# EXPORTAÇÃO
# ============================================================

st.subheader("📥 Exportação")

csv = df_filtrado.to_csv(index=False).encode("utf-8")
st.download_button("📄 Baixar CSV", csv, file_name=f"desenrola_{datetime.now().strftime('%Y%m%d')}.csv", mime="text/csv")

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")
st.caption("Dashboard analítico do programa Desenrola Brasil. Fonte: Banco Central do Brasil. Projeto para portfólio de Data Analytics.")
