import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import re
import warnings

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURAÇÃO DA PÁGINA
# ============================================================
st.set_page_config(
    page_title="Desenrola Brasil – Painel Executivo",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# TEMA CORPORATIVO PERSONALIZADO (PALETA SLATE & INDIGO)
# ============================================================
if "tema" not in st.session_state:
    st.session_state.tema = "claro"

T = st.session_state.tema
if T == "claro":
    COR_FUNDO       = "#F8FAFC"  # Slate 50
    COR_CARD        = "#FFFFFF"
    COR_TEXTO       = "#0F172A"  # Slate 900
    COR_BORDA       = "#E2E8F0"  # Slate 200
    COR_PRIMARIA    = "#4F46E5"  # Indigo 600
    COR_SECUNDARIA  = "#0EA5E9"  # Sky 500
    COR_SUCESSO     = "#10B981"  # Emerald 500
    COR_ALERTA      = "#EF4444"  # Red 500
    COR_ATENCAO     = "#F59E0B"  # Amber 500
    PLOTLY_TEMPLATE = "plotly_white"
    COR_GRID        = "rgba(15,23,42,0.06)"
else:
    COR_FUNDO       = "#0F172A"  # Slate 900
    COR_CARD        = "#1E293B"  # Slate 800
    COR_TEXTO       = "#F8FAFC"  # Slate 50
    COR_BORDA       = "#334155"  # Slate 700
    COR_PRIMARIA    = "#818CF8"  # Indigo 400
    COR_SECUNDARIA  = "#38BDF8"  # Sky 400
    COR_SUCESSO     = "#34D399"  # Emerald 400
    COR_ALERTA      = "#F87171"  # Red 400
    COR_ATENCAO     = "#FBBF24"  # Amber 400
    PLOTLY_TEMPLATE = "plotly_dark"
    COR_GRID        = "rgba(248,250,252,0.06)"

# ============================================================
# ESTILIZAÇÃO CSS
# ============================================================
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');
html, body, .stApp {{
    background-color: {COR_FUNDO};
    color: {COR_TEXTO};
    font-family: 'Inter', sans-serif;
}}
.block-container {{ padding: 1.5rem 2rem; }}
.kpi-card {{
    background: {COR_CARD};
    border-top: 4px solid {COR_PRIMARIA};
    border-radius: 12px;
    padding: 1.2rem;
    margin-bottom: 1rem;
    box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05), 0 2px 4px -1px rgba(0,0,0,0.03);
}}
.kpi-title {{ font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; color: #64748B; font-weight: 600; }}
.kpi-value {{ font-size: 1.8rem; font-weight: 700; color: {COR_TEXTO}; margin-top: 0.25rem; font-family: 'JetBrains Mono', monospace; }}
.kpi-sub   {{ font-size: 0.75rem; color: #94A3B8; margin-top: 0.25rem; }}
.insight-box {{
    background: {COR_CARD};
    border: 1px solid {COR_BORDA};
    border-radius: 12px;
    padding: 1.2rem;
    margin-bottom: 0.75rem;
    box-shadow: 0 1px 2px rgba(0,0,0,0.02);
}}
.insight-label {{ font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.05em; color: {COR_PRIMARIA}; font-weight: 700; margin-bottom: 0.4rem; }}
.insight-text  {{ font-size: 0.92rem; color: {COR_TEXTO}; line-height: 1.6; }}
.badge {{ padding: 4px 10px; border-radius: 6px; font-weight: 600; font-size: 0.75rem; display: inline-block; }}
.badge-low  {{ background: rgba(16,185,129,0.12); color: {COR_SUCESSO}; }}
.badge-mid  {{ background: rgba(245,158,11,0.12); color: {COR_ATENCAO}; }}
.badge-high {{ background: rgba(239,68,68,0.12);  color: {COR_ALERTA}; }}
.dq-card {{ background: {COR_CARD}; border: 1px solid {COR_BORDA}; border-radius: 8px; padding: 1rem; font-size: 0.85rem; line-height: 1.6; }}
.mono {{ font-family: 'JetBrains Mono', monospace; font-weight: 600; }}
</style>
""", unsafe_allow_html=True)

# ============================================================
# FORMATADORES E AUXILIARES
# ============================================================
def fmt_brl(v):
    if pd.isna(v) or v == 0: return "R$ 0"
    if v >= 1e9:  return f"R$ {v/1e9:.2f}B".replace(".", ",")
    if v >= 1e6:  return f"R$ {v/1e6:.2f}M".replace(".", ",")
    return f"R$ {v:,.0f}".replace(",", ".")

def fmt_num(v):
    if pd.isna(v): return "0"
    return f"{int(v):,}".replace(",", ".")

def classificar_banco(nome):
    nome = re.sub(r'\s*-\s*PRUDENCIAL$', '', str(nome).upper().strip())
    if any(x in nome for x in ["NUBANK","INTER","C6","NEON","ORIGINAL"]): return "Banco Digital"
    if any(x in nome for x in ["ITAU","BRADESCO","SANTANDER","CAIXA","BANCO DO BRASIL","BB"]): return "Banco Tradicional"
    if "BTG" in nome: return "Banco de Investimento"
    return "Outras Instituições"

def agrupar_regiao(uf):
    mapa = {
        "Norte":        ["AC","AM","AP","PA","RO","RR","TO"],
        "Nordeste":     ["AL","BA","CE","MA","PB","PE","PI","RN","SE"],
        "Centro-Oeste": ["DF","GO","MS","MT"],
        "Sudeste":      ["ES","MG","RJ","SP"],
        "Sul":          ["PR","RS","SC"]
    }
    for r, ests in mapa.items():
        if uf in ests: return r
    return "Não Identificado"

def layout_base(fig, height=400, showlegend=True):
    fig.update_layout(
        template=PLOTLY_TEMPLATE, height=height,
        margin=dict(l=40, r=40, t=50, b=40),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=COR_TEXTO, family="Inter", size=12),
        hovermode="x unified", showlegend=showlegend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.update_xaxes(showgrid=False, color=COR_TEXTO, linecolor=COR_BORDA)
    fig.update_yaxes(showgrid=True, gridcolor=COR_GRID, color=COR_TEXTO)
    return fig

@st.cache_data
def calcular_hhi(df, col):
    # Proteção extra para garantir soma numérica no HHI
    valores_numericos = pd.to_numeric(df[col], errors="coerce").fillna(0)
    total = valores_numericos.sum()
    return 0 if total == 0 else ((valores_numericos/total)**2).sum()*10000

@st.cache_data
def interpretar_hhi(hhi):
    if hhi < 1500: return "Mercado Altamente Competitivo", "badge-low", "Baixa concentração bancária."
    if hhi < 2500: return "Concentração Moderada", "badge-mid", "Mercado saudável, porém monitorável."
    return "Altamente Concentrado (Oligopólio)", "badge-high", "Alto risco de concentração de crédito."

# ============================================================
# ENGINE DE MODELAGEM E PROJEÇÃO (HOLT-WINTERS)
# ============================================================
@st.cache_data
def projetar_holt_winters(series_volume, datas, periodos=3):
    # Força os dados da série a serem estritamente numéricos float
    series_limpa = pd.to_numeric(series_volume, errors="coerce").fillna(0)
    if len(series_limpa) < 4: return None, None, None, None
    try:
        modelo = ExponentialSmoothing(series_limpa.values, trend="add", seasonal=None).fit(optimized=True)
        previsao = modelo.forecast(periodos)
        datas_futuras = pd.date_range(datas.max(), periods=periodos+1, freq="MS")[1:]
        sigma = np.std(modelo.resid) if len(modelo.resid) > 0 else 0
        return datas_futuras, previsao, previsao - 1.96*sigma, previsao + 1.96*sigma
    except:
        return None, None, None, None

# ============================================================
# PIPELINE DE TRATAMENTO DE DADOS
# ============================================================
@st.cache_data(ttl=3600)
def carregar_dados():
    try:
        df = pd.read_csv("dados_desenrola.csv", sep=";", encoding="utf-8")
    except Exception:
        # Fallback de dados fictícios para fins de teste se o arquivo não estiver presente
        datas = pd.date_range(start="2024-01-01", periods=12, freq="MS").strftime("%Y%m").astype(int)
        bancos_mock = ["Banco Itaú", "Banco do Brasil", "Bradesco", "Caixa Econômica", "Nubank", "Banco Inter"]
        ufs = ["SP", "RJ", "MG", "BA", "PR", "RS", "PE", "CE", "DF", "AM"]
        faixas = ["Faixa 1", "Faixa 2"]
        
        np.random.seed(42)
        rows = []
        for d in datas:
            for b in bancos_mock:
                for uf in ufs:
                    for f in faixas:
                        ops = np.random.randint(500, 15000)
                        vol = ops * np.random.uniform(1200, 4500)
                        rows.append([d, b, uf, f, ops, vol])
        df = pd.DataFrame(rows, columns=["data_base", "nome_conglomerado_financeiro", "unidade_federacao", "tipo_desenrola", "numero_operacoes", "volume_operacoes"])
    
    df.columns = df.columns.str.lower().str.strip()
    
    # Tratamento agressivo de strings numéricas (Remove pontos de milhar e substitui vírgulas decimais)
    for col in ["volume_operacoes", "numero_operacoes"]:
        if col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].astype(str).str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df["data_base"] = pd.to_datetime(df["data_base"].astype(str), format="%Y%m", errors="coerce")
    df["tipo_banco"] = df["nome_conglomerado_financeiro"].apply(classificar_banco)
    df["regiao"]     = df["unidade_federacao"].apply(agrupar_regiao)
    return df, df.dropna(subset=["volume_operacoes","numero_operacoes"])

df_raw, df = carregar_dados()

# ============================================================
# SIDEBAR DE ADM & FILTROS
# ============================================================
with st.sidebar:
    st.markdown("### ⚙️ Interface & Layout")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("☀️ Claro", use_container_width=True): st.session_state.tema = "claro"; st.rerun()
    with c2:
        if st.button("🌙 Escuro", use_container_width=True): st.session_state.tema = "escuro"; st.rerun()

    st.markdown("---")
    st.markdown("### 🔍 Filtros Estratégicos")
    tipos   = sorted(df["tipo_desenrola"].unique())
    tipo    = st.multiselect("Faixa do Programa", tipos, default=tipos)
    regioes = sorted(df["regiao"].unique())
    regiao  = st.multiselect("Região Demográfica", regioes, default=regioes)
    bancos  = sorted(df["tipo_banco"].unique())
    banco   = st.multiselect("Segmento Institucional", bancos, default=bancos)

# Aplicação de Filtros Dinâmicos
df_f = df[df["tipo_desenrola"].isin(tipo) & df["regiao"].isin(regiao) & df["tipo_banco"].isin(banco)]

if df_f.empty:
    st.warning("Nenhum registro encontrado para a combinação de filtros selecionada.")
    st.stop()

# ============================================================
# CÁLCULO DE KPIS DO TOPO (BLINDAGEM TYPE_ERROR)
# ============================================================
total_volume = float(df_f["volume_operacoes"].sum())
total_ops    = float(df_f["numero_operacoes"].sum())
ticket_medio = float(total_volume / total_ops) if total_ops > 0 else 0.0
num_inst     = int(df_f["nome_conglomerated_financeiro"].nunique()) if "nome_conglomerated_financeiro" in df_f.columns else int(df_f["nome_conglomerado_financeiro"].nunique())

# ============================================================
# CABEÇALHO & VISÃO GERAL
# ============================================================
st.title("🏦 Desenrola Brasil – Painel Executivo")
st.caption("Monitoramento estratégico das operações de renegociação de dívidas ativas")

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
with kpi1: st.markdown(f'<div class="kpi-card"><div class="kpi-title">💵 Volume Total</div><div class="kpi-value">{fmt_brl(total_volume)}</div><div class="kpi-sub">Total renegociado</div></div>', unsafe_allow_html=True)
with kpi2: st.markdown(f'<div class="kpi-card"><div class="kpi-title">📄 Total de Contratos</div><div class="kpi-value">{fmt_num(total_ops)}</div><div class="kpi-sub">CPFs/Dívidas liquidadas</div></div>', unsafe_allow_html=True)
with kpi3: st.markdown(f'<div class="kpi-card"><div class="kpi-title">🎫 Ticket Médio</div><div class="kpi-value">{fmt_brl(ticket_medio)}</div><div class="kpi-sub">Valor médio por contrato</div></div>', unsafe_allow_html=True)
with kpi4: st.markdown(f'<div class="kpi-card"><div class="kpi-title">🏛️ Players Ativos</div><div class="kpi-value">{num_inst}</div><div class="kpi-sub">Instituições financeiras</div></div>', unsafe_allow_html=True)

# CÁLCULO DE INSIGHTS AUTOMATIZADOS PARA TEXTO
market_share = df_f.groupby("nome_conglomerado_financeiro")["numero_operacoes"].sum().reset_index()
hhi_val = calcular_hhi(market_share, "numero_operacoes")
lbl_hhi, badge_hhi, desc_hhi = interpretar_hhi(hhi_val)

reg_share = df_f.groupby("regiao")["volume_operacoes"].sum().reset_index()
lider_reg = reg_share.loc[reg_share["volume_operacoes"].idxmax()] if not reg_share.empty else {"regiao": "N/A", "volume_operacoes": 0}

# ============================================================
# ESTAÇÃO DE ABAS ANALÍTICAS
# ============================================================
aba_temporal, aba_bancos, aba_geografica = st.tabs(["📈 Evolução & Tendências", "🏛️ Análise Bancária & Market Share", "🗺️ Distribuição Geográfica"])

# --- ABA 1: EVOLUÇÃO TEMPORAL ---
with aba_temporal:
    st.subheader("Análise Temporal e Projeção de Volume")
    evolucao_mensal = df_f.groupby("data_base")["volume_operacoes"].sum().reset_index()
    
    fig_temp = go.Figure()
    fig_temp.add_trace(go.Scatter(x=evolucao_mensal["data_base"], y=evolucao_mensal["volume_operacoes"], mode="lines+markers", name="Volume Realizado", line=dict(color=COR_PRIMARIA, width=3)))
    
    if len(evolucao_mensal) >= 3:
        evolucao_mensal["ma"] = evolucao_mensal["volume_operacoes"].rolling(3).mean()
        fig_temp.add_trace(go.Scatter(x=evolucao_mensal["data_base"], y=evolucao_mensal["ma"], mode="lines", name="Média Móvel (3 Meses)", line=dict(color=COR_ATENCAO, dash="dash")))

    df_futuras, prev, low, upp = projetar_holt_winters(evolucao_mensal["volume_operacoes"], evolucao_mensal["data_base"])
    if df_futuras is not None:
        fig_temp.add_trace(go.Scatter(x=df_futuras, y=prev, mode="lines+markers", name="Projeção (Holt-Winters)", line=dict(color=COR_SECUNDARIA, dash="dot")))
        fig_temp.add_trace(go.Scatter(x=list(df_futuras)+list(df_futuras)[::-1], y=list(upp)+list(low)[::-1], fill='toself', fillcolor='rgba(14,165,233,0.1)', line=dict(color='rgba(255,255,255,0)'), name="Intervalo de Confiança (95%)"))

    layout_base(fig_temp)
    st.plotly_chart(fig_temp, use_container_width=True)

# --- ABA 2: ANÁLISE BANCÁRIA ---
with aba_bancos:
    st.subheader("Concentração e Dominância de Mercado")
    
    c_banco_1, c_banco_2 = st.columns([1, 2])
    with c_banco_1:
        st.markdown(f"""
        <div class="insight-box">
            <div class="insight-label">Índice Herfindahl-Hirschman (HHI)</div>
            <span class="badge {badge_hhi}">{lbl_hhi}</span>
            <div class="insight-text" style="margin-top:0.5rem;"><b>Score: {hhi_val:.0f}</b>. {desc_hhi}</div>
        </div>
        """, unsafe_allow_html=True)
        
        banco_seg = df_f.groupby("tipo_banco")["volume_operacoes"].sum().reset_index()
        fig_pie = px.pie(banco_seg, values="volume_operacoes", names="tipo_banco", color_discrete_sequence=[COR_PRIMARIA, COR_SECUNDARIA, COR_SUCESSO, COR_ATENCAO])
        layout_base(fig_pie, height=280, showlegend=True)
        st.plotly_chart(fig_pie, use_container_width=True)

    with c_banco_2:
        top_bancos = df_f.groupby("nome_conglomerado_financeiro")["volume_operacoes"].sum().reset_index().sort_values("volume_operacoes", ascending=False).head(10)
        fig_bar = px.bar(top_bancos, x="volume_operacoes", y="nome_conglomerado_financeiro", orientation="h", title="Top 10 Instituições por Volume Financeiro", color_discrete_sequence=[COR_PRIMARIA])
        layout_base(fig_bar, height=420)
        fig_bar.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_bar, use_container_width=True)

# --- ABA 3: GEOGRAFIA ---
with aba_geografica:
    st.subheader("Distribuição do Impacto Regional")
    
    c_geo_1, c_geo_2 = st.columns(2)
    with c_geo_1:
        reg_data = df_f.groupby("regiao")["volume_operacoes"].sum().reset_index().sort_values("volume_operacoes", ascending=False)
        fig_reg = px.bar(reg_data, x="regiao", y="volume_operacoes", title="Volume Renegociado por Região", color="volume_operacoes", color_continuous_scale=[COR_SECUNDARIA, COR_PRIMARIA])
        layout_base(fig_reg)
        fig_reg.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig_reg, use_container_width=True)

    with c_geo_2:
        uf_data = df_f.groupby("unidade_federacao")["numero_operacoes"].sum().reset_index().sort_values("numero_operacoes", ascending=False).head(15)
        fig_uf = px.bar(uf_data, x="numero_operacoes", y="unidade_federacao", title="Top 15 Estados (Número de Operações)", orientation="h", color_discrete_sequence=[COR_SUCESSO])
        layout_base(fig_uf)
        fig_uf.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_uf, use_container_width=True)
