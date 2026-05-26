import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from datetime import datetime
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
# PALETAS DE CORES APRIMORADAS
# ============================================================
if "tema" not in st.session_state:
    st.session_state.tema = "claro"

T = st.session_state.tema

# Paleta Claro: tons de pedra/nude com acento verde-esmeralda + âmbar
# Paleta Escuro: grafite profundo com acento teal + laranja queimado
if T == "claro":
    COR_FUNDO        = "#F5F4F0"          # off-white creme
    COR_CARD         = "#FFFFFF"
    COR_CARD_ALT     = "#EEF2EF"          # verde muito claro p/ destaque
    COR_TEXTO        = "#1C2321"          # preto-esverdeado
    COR_TEXTO_SUB    = "#5C6B5E"          # cinza-verde
    COR_BORDA        = "#D6D9D2"
    COR_PRIMARIA     = "#2D6A4F"          # verde-esmeralda profundo
    COR_SECUNDARIA   = "#40916C"          # verde médio
    COR_ACENTO       = "#52B788"          # verde claro
    COR_SUCESSO      = "#1B7A52"
    COR_ALERTA       = "#C1440E"          # terracota
    COR_ATENCAO      = "#E09F3E"          # âmbar
    COR_INFO         = "#2196A6"          # teal
    COR_SIDEBAR      = "#2D6A4F"
    COR_SIDEBAR_TEXT = "#FFFFFF"
    PLOTLY_TEMPLATE  = "plotly_white"
    COR_GRID         = "rgba(45,106,79,0.08)"
    # Paleta sequencial para charts
    PALETTE_MAIN     = ["#2D6A4F","#40916C","#52B788","#74C69D","#95D5B2","#B7E4C7","#D8F3DC"]
    PALETTE_DIV      = ["#C1440E","#E09F3E","#F2E8C6","#B7E4C7","#52B788","#2D6A4F","#1B4332"]
    PALETTE_QUAL     = ["#2D6A4F","#E09F3E","#C1440E","#2196A6","#7B2D8B","#5C6B5E"]
else:
    COR_FUNDO        = "#0D1117"          # grafite quase-preto
    COR_CARD         = "#161B22"          # card escuro
    COR_CARD_ALT     = "#1A2332"          # card com toque azul
    COR_TEXTO        = "#E6EDF3"          # branco-gelo
    COR_TEXTO_SUB    = "#8B949E"          # cinza médio
    COR_BORDA        = "#30363D"
    COR_PRIMARIA     = "#3FB68C"          # teal-verde
    COR_SECUNDARIA   = "#56D6A8"          # verde-água claro
    COR_ACENTO       = "#79E4C0"          # menta
    COR_SUCESSO      = "#3FB68C"
    COR_ALERTA       = "#FF6B47"          # laranja queimado
    COR_ATENCAO      = "#F4A535"          # âmbar
    COR_INFO         = "#58A6FF"          # azul claro
    COR_SIDEBAR      = "#0D1117"
    COR_SIDEBAR_TEXT = "#E6EDF3"
    PLOTLY_TEMPLATE  = "plotly_dark"
    COR_GRID         = "rgba(63,182,140,0.08)"
    PALETTE_MAIN     = ["#3FB68C","#56D6A8","#79E4C0","#A0EDD3","#C5F4E6","#E0FAF2","#F0FDF9"]
    PALETTE_DIV      = ["#FF6B47","#F4A535","#3A3A2A","#1A3A2A","#3FB68C","#56D6A8","#79E4C0"]
    PALETTE_QUAL     = ["#3FB68C","#F4A535","#FF6B47","#58A6FF","#C377E0","#8B949E"]

# ============================================================
# CSS APRIMORADO
# ============================================================
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&family=Playfair+Display:wght@600;700&display=swap');

html, body, .stApp {{
    background-color: {COR_FUNDO};
    color: {COR_TEXTO};
    font-family: 'DM Sans', sans-serif;
}}

.block-container {{ padding: 1.2rem 1.8rem; max-width: 1600px; }}

/* ── Sidebar ── */
[data-testid="stSidebar"] {{
    background-color: {COR_SIDEBAR} !important;
    border-right: 1px solid {COR_BORDA};
}}
[data-testid="stSidebar"] * {{ color: {COR_SIDEBAR_TEXT} !important; }}
[data-testid="stSidebar"] .stMultiSelect [data-baseweb="tag"] {{
    background-color: {COR_PRIMARIA}33 !important;
}}

/* ── KPI Cards ── */
.kpi-card {{
    background: {COR_CARD};
    border-left: 4px solid {COR_PRIMARIA};
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.6rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
    position: relative;
    overflow: hidden;
}}
.kpi-card::after {{
    content: '';
    position: absolute;
    top: 0; right: 0; bottom: 0;
    width: 40%;
    background: linear-gradient(135deg, transparent, {COR_PRIMARIA}08);
    pointer-events: none;
}}
.kpi-card:hover {{
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(0,0,0,0.10);
}}
.kpi-card.sucesso  {{ border-left-color: {COR_SUCESSO}; }}
.kpi-card.alerta   {{ border-left-color: {COR_ALERTA}; }}
.kpi-card.atencao  {{ border-left-color: {COR_ATENCAO}; }}
.kpi-card.info     {{ border-left-color: {COR_INFO}; }}

.kpi-icon  {{ font-size: 1.1rem; margin-bottom: 0.3rem; display: block; }}
.kpi-title {{ font-size: 0.62rem; text-transform: uppercase; letter-spacing: 0.1em; color: {COR_TEXTO_SUB}; font-weight: 600; }}
.kpi-value {{ font-size: 1.65rem; font-weight: 700; color: {COR_TEXTO}; margin-top: 0.2rem; font-family: 'DM Mono', monospace; line-height: 1.1; }}
.kpi-sub   {{ font-size: 0.71rem; color: {COR_TEXTO_SUB}; margin-top: 0.25rem; }}
.kpi-delta-pos {{ color: {COR_SUCESSO}; font-weight: 600; font-size: 0.75rem; }}
.kpi-delta-neg {{ color: {COR_ALERTA}; font-weight: 600; font-size: 0.75rem; }}

/* ── Insight / Conclusão ── */
.insight-box {{
    background: {COR_CARD};
    border: 1px solid {COR_BORDA};
    border-top: 3px solid {COR_PRIMARIA};
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.7rem;
    transition: border-top-color 0.3s;
}}
.insight-box.warning {{ border-top-color: {COR_ATENCAO}; }}
.insight-box.danger  {{ border-top-color: {COR_ALERTA}; }}
.insight-box.info    {{ border-top-color: {COR_INFO}; }}
.insight-label {{ font-size: 0.62rem; text-transform: uppercase; letter-spacing: 0.1em; color: {COR_TEXTO_SUB}; font-weight: 700; margin-bottom: 0.35rem; }}
.insight-text  {{ font-size: 0.87rem; color: {COR_TEXTO}; line-height: 1.6; }}

/* ── Badges ── */
.badge {{ padding: 3px 10px; border-radius: 20px; font-weight: 600; font-size: 0.68rem; display: inline-block; }}
.badge-low  {{ background: {COR_SUCESSO}20; color: {COR_SUCESSO}; border: 1px solid {COR_SUCESSO}40; }}
.badge-mid  {{ background: {COR_ATENCAO}20; color: {COR_ATENCAO}; border: 1px solid {COR_ATENCAO}40; }}
.badge-high {{ background: {COR_ALERTA}20; color: {COR_ALERTA}; border: 1px solid {COR_ALERTA}40; }}
.badge-info {{ background: {COR_INFO}20; color: {COR_INFO}; border: 1px solid {COR_INFO}40; }}

/* ── DQ Card ── */
.dq-card {{
    background: {COR_CARD};
    border: 1px solid {COR_BORDA};
    border-radius: 10px;
    padding: 0.9rem 1rem;
    font-size: 0.78rem;
    line-height: 1.7;
}}
.mono {{ font-family: 'DM Mono', monospace; font-size: 0.82rem; }}

/* ── Section header ── */
.section-header {{
    font-family: 'Playfair Display', serif;
    font-size: 1.3rem;
    font-weight: 700;
    color: {COR_TEXTO};
    margin: 1.5rem 0 0.8rem;
    padding-bottom: 0.4rem;
    border-bottom: 2px solid {COR_PRIMARIA}40;
}}

/* ── Alert strip ── */
.alert-strip {{
    padding: 0.6rem 1rem;
    border-radius: 8px;
    margin-bottom: 0.5rem;
    font-size: 0.84rem;
    display: flex;
    align-items: center;
    gap: 0.6rem;
}}
.alert-error   {{ background: {COR_ALERTA}15; border-left: 3px solid {COR_ALERTA}; color: {COR_ALERTA}; }}
.alert-warning {{ background: {COR_ATENCAO}15; border-left: 3px solid {COR_ATENCAO}; color: {COR_ATENCAO}; }}
.alert-success {{ background: {COR_SUCESSO}15; border-left: 3px solid {COR_SUCESSO}; color: {COR_SUCESSO}; }}
.alert-info    {{ background: {COR_INFO}15; border-left: 3px solid {COR_INFO}; color: {COR_INFO}; }}

/* ── Métrica comparativa ── */
.rank-row {{
    display: flex; align-items: center; justify-content: space-between;
    padding: 0.45rem 0; border-bottom: 1px solid {COR_BORDA};
    font-size: 0.82rem;
}}
.rank-row:last-child {{ border-bottom: none; }}
.rank-bar-wrap {{ flex: 1; margin: 0 0.8rem; height: 6px; background: {COR_BORDA}; border-radius: 3px; overflow: hidden; }}
.rank-bar {{ height: 100%; border-radius: 3px; background: {COR_PRIMARIA}; }}
</style>
""", unsafe_allow_html=True)

# ============================================================
# UTILITÁRIOS
# ============================================================
def fmt_brl(v):
    if pd.isna(v) or v == 0: return "R$ 0"
    if v >= 1e9:  return f"R$ {v/1e9:.2f}B".replace(".", ",")
    if v >= 1e6:  return f"R$ {v/1e6:.1f}M".replace(".", ",")
    return f"R$ {v:,.0f}".replace(",", ".")

def fmt_num(v):
    if pd.isna(v): return "0"
    return f"{int(v):,}".replace(",", ".")

def fmt_pct(v):
    return f"{v:+.1f}%" if v > 0 else f"{v:.1f}%"

def classificar_banco(nome):
    nome = re.sub(r'\s*-\s*PRUDENCIAL$', '', str(nome).upper().strip())
    if any(x in nome for x in ["NUBANK","INTER","C6","NEON","ORIGINAL","PAN","NEXT"]): return "Banco Digital"
    if any(x in nome for x in ["ITAU","BRADESCO","SANTANDER","CAIXA","BANCO DO BRASIL","BB","HSBC"]): return "Banco Tradicional"
    if any(x in nome for x in ["BTG","XP","MODAL","GENIAL"]): return "Banco de Investimento"
    if any(x in nome for x in ["SICOOB","SICREDI","CRESOL","UNICRED"]): return "Cooperativa"
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

@st.cache_data
def calcular_hhi(df, col):
    total = df[col].sum()
    return 0 if total == 0 else ((df[col]/total)**2).sum()*10000

@st.cache_data
def interpretar_hhi(hhi):
    if hhi < 1500:
        return "Mercado Competitivo (HHI < 1.500)", "badge-low", "Baixo risco de concentração bancária – ambiente saudável para o consumidor."
    if hhi < 2500:
        return "Concentração Moderada (HHI 1.500–2.500)", "badge-mid", "Atenção: poucos bancos lideram o programa. Monitorar tendência de concentração."
    return "Altamente Concentrado (HHI > 2.500)", "badge-high", "Risco sistêmico elevado: oligopólio pode limitar acesso e reduzir competitividade."

@st.cache_data
def calcular_pareto(df, col):
    df_s = df.sort_values(col, ascending=False).reset_index(drop=True)
    total = df_s[col].sum()
    df_s["pct_acum"] = (df_s[col].cumsum() / total * 100) if total > 0 else 0
    return df_s

@st.cache_data
def calcular_gini(series):
    """Calcula o coeficiente de Gini para medir desigualdade de distribuição."""
    arr = np.sort(series.dropna().values)
    n = len(arr)
    if n == 0: return 0
    index = np.arange(1, n+1)
    return (2 * np.sum(index * arr) / (n * arr.sum()) - (n + 1) / n)

def layout_base(fig, height=440, showlegend=True):
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=height,
        margin=dict(l=50, r=40, t=60, b=50),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=COR_TEXTO, family="DM Sans", size=12),
        hovermode="x unified",
        showlegend=showlegend,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="right", x=1, font=dict(size=11),
            bgcolor="rgba(0,0,0,0)", borderwidth=0
        )
    )
    fig.update_xaxes(showgrid=False, color=COR_TEXTO, title_font_size=12, linecolor=COR_BORDA, tickfont=dict(size=11))
    fig.update_yaxes(showgrid=True, gridcolor=COR_GRID, color=COR_TEXTO, title_font_size=12, tickfont=dict(size=11))
    return fig

@st.cache_data
def projetar_holt_winters(series_volume, datas, periodos=3):
    if len(series_volume) < 4:
        return None, None, None, None
    try:
        modelo = ExponentialSmoothing(
            series_volume.values, trend="add", seasonal=None, initialization_method="estimated"
        ).fit(optimized=True)
        previsao = modelo.forecast(periodos)
        datas_futuras = pd.date_range(datas.max(), periods=periodos+1, freq="MS")[1:]
        sigma = np.std(modelo.resid)
        return datas_futuras, previsao, previsao - 1.96*sigma, previsao + 1.96*sigma
    except Exception:
        return None, None, None, None

@st.cache_data
def clusterizar_bancos(df, col_banco):
    dados = df.groupby(col_banco).agg(
        numero_operacoes=("numero_operacoes","sum"),
        volume_operacoes=("volume_operacoes","sum")
    ).reset_index()
    dados["ticket_medio"] = dados["volume_operacoes"] / dados["numero_operacoes"].replace(0, np.nan)
    dados = dados[dados["numero_operacoes"] > 100].dropna()
    if len(dados) < 3:
        return None, None
    scaler = StandardScaler()
    features = scaler.fit_transform(dados[["numero_operacoes","ticket_medio"]])
    n = min(3, len(dados))
    kmeans = KMeans(n_clusters=n, random_state=42, n_init=10)
    dados["cluster"] = kmeans.fit_predict(features)
    medias = dados.groupby("cluster")[["numero_operacoes","ticket_medio"]].mean()
    rank_vol    = medias["numero_operacoes"].rank(ascending=False).astype(int)
    rank_ticket = medias["ticket_medio"].rank(ascending=False).astype(int)
    def rotulo(c):
        alto_vol = rank_vol[c] == 1
        alto_ticket = rank_ticket[c] == 1
        if alto_vol and not alto_ticket: return "Alto Volume / Baixo Ticket"
        if not alto_vol and alto_ticket: return "Baixo Volume / Alto Ticket"
        return "Perfil Equilibrado"
    dados["cluster_nome"] = dados["cluster"].map(rotulo)
    fig = go.Figure()
    cores_cluster = {
        "Alto Volume / Baixo Ticket": COR_PRIMARIA,
        "Baixo Volume / Alto Ticket": COR_ATENCAO,
        "Perfil Equilibrado": COR_INFO
    }
    for nome, grp in dados.groupby("cluster_nome"):
        size_norm = np.log1p(grp["volume_operacoes"] / grp["volume_operacoes"].max() + 0.01) * 28 + 10
        fig.add_trace(go.Scatter(
            x=grp["numero_operacoes"], y=grp["ticket_medio"],
            mode="markers", name=nome,
            marker=dict(size=size_norm, color=cores_cluster.get(nome, COR_TEXTO_SUB),
                        opacity=0.82, line=dict(width=1.5, color=COR_BORDA)),
            hovertemplate="<b>%{customdata}</b><br>Operações: %{x:,.0f}<br>Ticket Médio: R$ %{y:,.2f}<extra></extra>",
            customdata=grp[col_banco]
        ))
    fig.update_layout(title=dict(text="Agrupamento de Instituições por Comportamento Operacional (K-Means)", font_size=14),
                      xaxis_title="Número de Operações", yaxis_title="Ticket Médio (R$)")
    layout_base(fig, height=500)
    return fig, dados

def calcular_data_quality(df_original, df_limpo):
    total_raw = len(df_original) if df_original is not None else len(df_limpo)
    total_limpo = len(df_limpo)
    completude = (df_limpo.notna().sum() / len(df_limpo)) * 100
    periodo_min = df_limpo["data_base"].min().strftime("%m/%Y") if not df_limpo["data_base"].isna().all() else "N/D"
    periodo_max = df_limpo["data_base"].max().strftime("%m/%Y") if not df_limpo["data_base"].isna().all() else "N/D"
    pct_descartados = (total_raw - total_limpo) / total_raw * 100 if total_raw > 0 else 0
    return {
        "total_registros":       total_limpo,
        "registros_descartados": total_raw - total_limpo,
        "pct_descartados":       pct_descartados,
        "completude_volume":     completude.get("volume_operacoes", 100),
        "completude_operacoes":  completude.get("numero_operacoes", 100),
        "periodo_inicio":        periodo_min,
        "periodo_fim":           periodo_max,
        "meses_cobertos":        df_limpo["data_base"].nunique(),
        "ultima_data":           periodo_max
    }

@st.cache_data
def gerar_alertas(evolucao, hhi, ticket_medio_geral, gini_val=None):
    alertas = []
    if len(evolucao) >= 2:
        cresc_ultimo = evolucao["crescimento"].dropna().iloc[-1] if len(evolucao["crescimento"].dropna()) > 0 else 0
        if cresc_ultimo < -15:
            alertas.append(("error",   "🔴 Queda Abrupta",      f"Volume caiu {cresc_ultimo:.1f}% no último mês – investigar fatores estruturais."))
        elif cresc_ultimo < -5:
            alertas.append(("warning", "🟡 Desaceleração",       f"Queda de {cresc_ultimo:.1f}% sinaliza perda de ritmo; acompanhar próximo ciclo."))
        elif cresc_ultimo > 20:
            alertas.append(("success", "🟢 Aceleração Forte",    f"Crescimento de +{cresc_ultimo:.1f}% – tendência positiva sustentada."))
        elif cresc_ultimo > 5:
            alertas.append(("success", "🟢 Crescimento Estável", f"Expansão de +{cresc_ultimo:.1f}% dentro da faixa normal."))
    if hhi > 2500:
        alertas.append(("error",   "🔴 Concentração Elevada",  "HHI > 2.500: mercado oligopolizado – risco ao consumidor."))
    elif hhi > 1500:
        alertas.append(("warning", "🟡 Concentração Moderada", "HHI 1.500–2.500: monitorar tendência e entrada de novos players."))
    else:
        alertas.append(("success", "🟢 Mercado Competitivo",   "HHI < 1.500: estrutura saudável e diversificada."))
    if gini_val is not None:
        if gini_val > 0.7:
            alertas.append(("error",   "🔴 Alta Desigualdade Regional", f"Gini = {gini_val:.2f}: distribuição muito concentrada em poucos estados."))
        elif gini_val > 0.5:
            alertas.append(("warning", "🟡 Desigualdade Regional",     f"Gini = {gini_val:.2f}: desigualdade relevante entre regiões."))
    if ticket_medio_geral > 8000:
        alertas.append(("warning", "🟡 Ticket Elevado",   f"Ticket médio de {fmt_brl(ticket_medio_geral)} pode indicar baixa adesão por devedores de menor renda."))
    elif ticket_medio_geral < 1000:
        alertas.append(("info",    "ℹ️ Ticket Reduzido",  f"Ticket médio de {fmt_brl(ticket_medio_geral)} reflete base de consumidores endividados com dívidas menores."))
    return alertas

# ============================================================
# CARREGAMENTO
# ============================================================
@st.cache_data(ttl=3600)
def carregar_dados():
    for enc in ["utf-8","latin1","cp1252"]:
        try:
            df = pd.read_csv("dados_desenrola.csv", sep=";", encoding=enc, low_memory=False)
            df.columns = df.columns.str.lower().str.strip()
            for col in ["numero_operacoes","volume_operacoes"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(
                        df[col].astype(str)
                            .str.replace(".","",regex=False)
                            .str.replace(",",".",regex=False),
                        errors="coerce"
                    )
            df["data_base"] = pd.to_datetime(df["data_base"].astype(str), format="%Y%m", errors="coerce")
            df["tipo_banco"] = df["nome_conglomerado_financeiro"].apply(classificar_banco)
            df["regiao"]     = df["unidade_federacao"].apply(agrupar_regiao)
            # Métricas derivadas
            df["ticket_estimado"] = df["volume_operacoes"] / df["numero_operacoes"].replace(0, np.nan)
            df_limpo = df.dropna(subset=["volume_operacoes","numero_operacoes"])
            return df, df_limpo
        except Exception:
            continue
    return None, None

df_raw, df = carregar_dados()
if df is None:
    st.error("❌ Erro ao carregar dados. Verifique se 'dados_desenrola.csv' está presente e bem formatado.")
    st.stop()

dq = calcular_data_quality(df_raw, df)

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown(f"""
    <div style="padding:0.6rem 0 1rem; border-bottom:1px solid {COR_BORDA}40; margin-bottom:1rem;">
        <div style="font-family:'Playfair Display',serif; font-size:1.05rem; font-weight:700; color:{COR_SIDEBAR_TEXT};">🏦 Desenrola Brasil</div>
        <div style="font-size:0.72rem; color:{COR_SIDEBAR_TEXT}90; margin-top:0.2rem;">Painel Executivo · BCB/SCR</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**⚙️ Aparência**")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("☀️ Claro", use_container_width=True):
            st.session_state.tema = "claro"; st.rerun()
    with c2:
        if st.button("🌙 Escuro", use_container_width=True):
            st.session_state.tema = "escuro"; st.rerun()

    st.markdown("---")
    st.markdown("**🔍 Filtros**")
    tipos   = sorted(df["tipo_desenrola"].unique())
    tipo    = st.multiselect("Faixa do Programa", tipos, default=tipos)
    regioes = sorted(df["regiao"].unique())
    regiao  = st.multiselect("Região", regioes, default=regioes)
    bancos  = sorted(df["tipo_banco"].unique())
    banco   = st.multiselect("Segmento", bancos, default=bancos)

    # Filtro de período
    datas_disp = sorted(df["data_base"].dropna().unique())
    if len(datas_disp) > 1:
        idx_min, idx_max = st.select_slider(
            "Período", options=list(range(len(datas_disp))),
            value=(0, len(datas_disp)-1),
            format_func=lambda i: pd.Timestamp(datas_disp[i]).strftime("%m/%Y")
        )
        data_min = datas_disp[idx_min]
        data_max = datas_disp[idx_max]
    else:
        data_min, data_max = datas_disp[0], datas_disp[-1]

    if st.button("🔄 Limpar Filtros", use_container_width=True): st.rerun()

    st.markdown("---")
    st.markdown("**📋 Qualidade dos Dados**")
    dq_color = COR_SUCESSO if dq['pct_descartados'] < 5 else (COR_ATENCAO if dq['pct_descartados'] < 15 else COR_ALERTA)
    st.markdown(f"""
    <div class="dq-card">
    <b>Registros válidos:</b> <span class="mono">{fmt_num(dq['total_registros'])}</span><br>
    <b>Descartados:</b> <span class="mono" style="color:{dq_color}">{fmt_num(dq['registros_descartados'])} ({dq['pct_descartados']:.1f}%)</span><br>
    <b>Período:</b> <span class="mono">{dq['periodo_inicio']} → {dq['periodo_fim']}</span><br>
    <b>Meses:</b> <span class="mono">{dq['meses_cobertos']}</span><br>
    <b>Completude volume:</b> <span class="mono">{dq['completude_volume']:.1f}%</span><br>
    <b>Completude ops:</b> <span class="mono">{dq['completude_operacoes']:.1f}%</span>
    </div>
    """, unsafe_allow_html=True)

# Filtro principal
df_f = df[
    df["tipo_desenrola"].isin(tipo) &
    df["regiao"].isin(regiao) &
    df["tipo_banco"].isin(banco) &
    (df["data_base"] >= pd.Timestamp(data_min)) &
    (df["data_base"] <= pd.Timestamp(data_max))
]

if df_f.empty:
    st.warning("⚠️ Nenhum dado encontrado com os filtros selecionados. Ajuste os parâmetros na barra lateral.")
    st.stop()

col_banco = "nome_conglomerado_financeiro"

# ============================================================
# CABEÇALHO
# ============================================================
st.markdown("""
<div style="margin-bottom:0.3rem;">
    <span style="font-family:'Playfair Display',serif; font-size:2rem; font-weight:700; line-height:1.1;">
        🏦 Desenrola Brasil
    </span>
    <span style="font-size:1rem; opacity:0.6; margin-left:0.6rem;">Painel Executivo</span>
</div>
""", unsafe_allow_html=True)
st.caption("Monitoramento de renegociação de dívidas · Fonte: Banco Central do Brasil (SCR) · Atualização mensal")

# ============================================================
# KPIs PRINCIPAIS
# ============================================================
total_volume  = df_f["volume_operacoes"].sum()
total_ops     = df_f["numero_operacoes"].sum()
ticket_medio  = total_volume / total_ops if total_ops > 0 else 0
num_inst      = df_f[col_banco].nunique()
num_estados   = df_f["unidade_federacao"].nunique() if "unidade_federacao" in df_f.columns else 0

# Cálculo de crescimento MoM do último mês
evolucao_tmp = df_f.groupby("data_base")["volume_operacoes"].sum().reset_index().sort_values("data_base")
if len(evolucao_tmp) >= 2:
    cresc_mom = ((evolucao_tmp["volume_operacoes"].iloc[-1] / evolucao_tmp["volume_operacoes"].iloc[-2]) - 1) * 100
    delta_html = f'<div class="kpi-delta-{"pos" if cresc_mom >= 0 else "neg"}">{fmt_pct(cresc_mom)} vs mês anterior</div>'
else:
    delta_html = ""

col_k1, col_k2, col_k3, col_k4, col_k5 = st.columns(5)
kpi_data = [
    (col_k1, "💵", "Volume Renegociado",    fmt_brl(total_volume),  delta_html,            ""),
    (col_k2, "📄", "Contratos",             fmt_num(total_ops),     "operações registradas", "info"),
    (col_k3, "🎫", "Ticket Médio",          fmt_brl(ticket_medio),  "Volume ÷ Contratos",    "atencao"),
    (col_k4, "🏛️", "Instituições",          fmt_num(num_inst),      "financeiras participantes",""),
    (col_k5, "📍", "Estados Cobertos",      fmt_num(num_estados),   "unidades da federação", "sucesso"),
]
for col, icon, title, value, sub, cls in kpi_data:
    with col:
        cls_str = f" {cls}" if cls else ""
        st.markdown(f"""
        <div class="kpi-card{cls_str}">
            <span class="kpi-icon">{icon}</span>
            <div class="kpi-title">{title}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-sub">{sub}</div>
        </div>
        """, unsafe_allow_html=True)

# ============================================================
# ALERTAS INTELIGENTES
# ============================================================
market_hhi_df = df_f.groupby(col_banco)["numero_operacoes"].sum().reset_index()
hhi_val       = calcular_hhi(market_hhi_df, "numero_operacoes")

evolucao_g = df_f.groupby("data_base")["volume_operacoes"].sum().reset_index().sort_values("data_base")
evolucao_g["crescimento"] = evolucao_g["volume_operacoes"].pct_change() * 100

vol_uf = df_f.groupby("unidade_federacao")["volume_operacoes"].sum() if "unidade_federacao" in df_f.columns else pd.Series([0])
gini_uf = calcular_gini(vol_uf)

alertas = gerar_alertas(evolucao_g, hhi_val, ticket_medio, gini_uf)

if alertas:
    st.markdown('<div class="section-header">⚡ Alertas Automáticos</div>', unsafe_allow_html=True)
    cols_alerta = st.columns(min(len(alertas), 3))
    for i, (tipo_a, titulo, msg) in enumerate(alertas):
        css_map = {"error":"alert-error","warning":"alert-warning","success":"alert-success","info":"alert-info"}
        with cols_alerta[i % len(cols_alerta)]:
            st.markdown(f'<div class="alert-strip {css_map.get(tipo_a,"")}"><b>{titulo}</b> – {msg}</div>', unsafe_allow_html=True)

# ============================================================
# RESUMO EXECUTIVO
# ============================================================
reg_data = df_f.groupby("regiao")["volume_operacoes"].sum().reset_index()
reg_data["pct"] = (reg_data["volume_operacoes"] / reg_data["volume_operacoes"].sum() * 100).round(1)
lider_regiao = reg_data.loc[reg_data["volume_operacoes"].idxmax()]
top2_regioes = reg_data.nlargest(2,"volume_operacoes")
conc_top2 = top2_regioes["pct"].sum()

total_contratos = market_hhi_df["numero_operacoes"].sum()
if total_contratos > 0:
    lider_banco  = market_hhi_df.loc[market_hhi_df["numero_operacoes"].idxmax(), col_banco]
    part_banco   = (market_hhi_df["numero_operacoes"].max() / total_contratos) * 100
    top5_part    = (market_hhi_df.nlargest(5,"numero_operacoes")["numero_operacoes"].sum() / total_contratos) * 100
else:
    lider_banco, part_banco, top5_part = "N/A", 0, 0

_, badge_hhi, expl_hhi = interpretar_hhi(hhi_val)

if len(evolucao_g) > 1:
    cresc_medio = evolucao_g["crescimento"].dropna().mean()
    tendencia_txt = (
        f"crescimento médio de <b>+{cresc_medio:.1f}%</b>/mês, demonstrando expansão sustentada"
        if cresc_medio > 0
        else f"contração média de <b>{cresc_medio:.1f}%</b>/mês, indicando perda de momentum"
    )
else:
    tendencia_txt = "dados insuficientes para calcular tendência temporal"

st.markdown('<div class="section-header">📌 Resumo Executivo</div>', unsafe_allow_html=True)
col_c1, col_c2, col_c3 = st.columns(3)

with col_c1:
    st.markdown(f"""
    <div class="insight-box">
        <div class="insight-label">🎯 Concentração Regional</div>
        <div class="insight-text">
        A região <b>{lider_regiao['regiao']}</b> lidera com <b>{lider_regiao['pct']:.1f}%</b> do volume total.
        As duas maiores regiões juntas concentram <b>{conc_top2:.1f}%</b> das renegociações,
        sugerindo oportunidade de expansão nas demais.
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_c2:
    conc_class = "danger" if part_banco > 40 else ("warning" if part_banco > 25 else "")
    st.markdown(f"""
    <div class="insight-box {conc_class}">
        <div class="insight-label">🏦 Liderança Bancária</div>
        <div class="insight-text">
        <b>{lider_banco}</b> responde por <b>{part_banco:.1f}%</b> dos contratos.
        Os 5 maiores participantes concentram <b>{top5_part:.1f}%</b> do mercado
        (HHI: <b>{hhi_val:.0f}</b>). {expl_hhi}
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_c3:
    tend_class = "danger" if cresc_medio < -5 else ("warning" if cresc_medio < 0 else "")
    st.markdown(f"""
    <div class="insight-box {tend_class}">
        <div class="insight-label">📈 Dinâmica Temporal</div>
        <div class="insight-text">
        O programa apresenta {tendencia_txt}.
        Ticket médio de <b>{fmt_brl(ticket_medio)}</b> e desigualdade regional
        (Gini = <b>{gini_uf:.2f}</b>) reforçam a necessidade de políticas regionalizadas.
        </div>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# ABA PRINCIPAL DE ANÁLISES
# ============================================================
tabs = st.tabs([
    "📈 Evolução Temporal",
    "🏛️ Concentração Bancária",
    "🗺️ Análise Regional",
    "🔬 Análise Avançada",
    "📊 Distribuição & Pareto",
])

# ─────────────────────────────────────────────────────────────
# TAB 1: EVOLUÇÃO TEMPORAL
# ─────────────────────────────────────────────────────────────
with tabs[0]:
    st.markdown('<div class="section-header">📈 Evolução Temporal do Programa</div>', unsafe_allow_html=True)

    evolucao_tipo = df_f.groupby(["data_base","tipo_desenrola"])["volume_operacoes"].sum().reset_index()

    col_e1, col_e2 = st.columns([2,1])
    with col_e1:
        fig_ev = go.Figure()
        tipos_unicos = evolucao_tipo["tipo_desenrola"].unique()
        for i, tp in enumerate(tipos_unicos):
            grp = evolucao_tipo[evolucao_tipo["tipo_desenrola"]==tp]
            fig_ev.add_trace(go.Scatter(
                x=grp["data_base"], y=grp["volume_operacoes"],
                mode="lines+markers", name=tp,
                line=dict(color=PALETTE_QUAL[i % len(PALETTE_QUAL)], width=2.5),
                marker=dict(size=6),
                fill="tonexty" if i > 0 else "none",
                fillcolor=f"{PALETTE_QUAL[i % len(PALETTE_QUAL)]}18",
                hovertemplate=f"<b>{tp}</b><br>%{{x|%b/%Y}}<br>Volume: R$ %{{y:,.0f}}<extra></extra>"
            ))
        # Projeção
        datas_fut, prev, low, upp = projetar_holt_winters(
            evolucao_g["volume_operacoes"], evolucao_g["data_base"], periodos=3
        )
        if datas_fut is not None:
            fig_ev.add_trace(go.Scatter(
                x=list(datas_fut), y=list(prev),
                mode="lines+markers", name="Projeção (HW)",
                line=dict(color=COR_ATENCAO, width=2, dash="dash"),
                marker=dict(size=7, symbol="diamond"),
                hovertemplate="<b>Projeção</b><br>%{x|%b/%Y}<br>R$ %{y:,.0f}<extra></extra>"
            ))
            fig_ev.add_trace(go.Scatter(
                x=list(datas_fut)+list(datas_fut[::-1]),
                y=list(upp)+list(low[::-1]),
                fill="toself", fillcolor=f"{COR_ATENCAO}18",
                line=dict(color="rgba(0,0,0,0)"), name="IC 95%", showlegend=False
            ))
        fig_ev.update_layout(title="Volume Renegociado por Faixa do Programa + Projeção")
        layout_base(fig_ev, height=420)
        fig_ev.update_yaxes(tickprefix="R$ ", tickformat=",.0f")
        st.plotly_chart(fig_ev, use_container_width=True)

    with col_e2:
        # Taxa de crescimento MoM
        fig_cresc = go.Figure()
        cresc_valido = evolucao_g.dropna(subset=["crescimento"])
        cores_barras = [COR_SUCESSO if v >= 0 else COR_ALERTA for v in cresc_valido["crescimento"]]
        fig_cresc.add_trace(go.Bar(
            x=cresc_valido["data_base"], y=cresc_valido["crescimento"],
            marker_color=cores_barras,
            hovertemplate="%{x|%b/%Y}<br>Δ %{y:.1f}%<extra></extra>",
            name="MoM %"
        ))
        fig_cresc.add_hline(y=0, line_color=COR_BORDA, line_width=1.5)
        fig_cresc.update_layout(title="Crescimento Mensal (MoM %)")
        layout_base(fig_cresc, height=420, showlegend=False)
        fig_cresc.update_yaxes(ticksuffix="%")
        st.plotly_chart(fig_cresc, use_container_width=True)

    # Heatmap volume por mês e faixa
    st.markdown("**Heatmap · Volume por Mês e Faixa do Programa**")
    pivot = df_f.pivot_table(index="tipo_desenrola", columns="data_base",
                              values="volume_operacoes", aggfunc="sum")
    pivot.columns = [pd.Timestamp(c).strftime("%b/%y") for c in pivot.columns]
    fig_heat = go.Figure(go.Heatmap(
        z=pivot.values, x=pivot.columns.tolist(), y=pivot.index.tolist(),
        colorscale=[[0,"#0D1117"],[0.5,COR_SECUNDARIA],[1,COR_ACENTO]] if T=="escuro" else
                   [[0,"#F5F4F0"],[0.5,COR_SECUNDARIA],[1,COR_PRIMARIA]],
        hovertemplate="Faixa: %{y}<br>Mês: %{x}<br>Volume: R$ %{z:,.0f}<extra></extra>",
        colorbar=dict(title="Volume", tickprefix="R$ ", tickformat=".2s")
    ))
    layout_base(fig_heat, height=280, showlegend=False)
    fig_heat.update_layout(title="Mapa de Calor – Volume por Faixa e Período")
    st.plotly_chart(fig_heat, use_container_width=True)

# ─────────────────────────────────────────────────────────────
# TAB 2: CONCENTRAÇÃO BANCÁRIA
# ─────────────────────────────────────────────────────────────
with tabs[1]:
    st.markdown('<div class="section-header">🏛️ Análise de Concentração Bancária</div>', unsafe_allow_html=True)

    top_n = st.slider("Top N instituições", 5, 30, 15, key="topn_banco")
    banco_agg = df_f.groupby(col_banco).agg(
        volume=("volume_operacoes","sum"),
        operacoes=("numero_operacoes","sum")
    ).reset_index()
    banco_agg["ticket"] = banco_agg["volume"] / banco_agg["operacoes"].replace(0, np.nan)
    banco_agg["segmento"] = banco_agg[col_banco].apply(classificar_banco)
    banco_agg = banco_agg.nlargest(top_n, "volume")

    col_b1, col_b2 = st.columns(2)
    with col_b1:
        fig_bar = go.Figure(go.Bar(
            x=banco_agg["volume"],
            y=banco_agg[col_banco].str[:25],
            orientation="h",
            marker=dict(
                color=banco_agg["volume"],
                colorscale=[[0, f"{COR_PRIMARIA}60"], [1, COR_PRIMARIA]],
                showscale=False,
                line=dict(width=0)
            ),
            customdata=banco_agg[["operacoes","ticket","segmento"]].values,
            hovertemplate="<b>%{y}</b><br>Volume: R$ %{x:,.0f}<br>Operações: %{customdata[0]:,.0f}<br>Ticket: R$ %{customdata[1]:,.0f}<br>Segmento: %{customdata[2]}<extra></extra>"
        ))
        fig_bar.update_layout(title=f"Volume Renegociado – Top {top_n} Instituições")
        layout_base(fig_bar, height=480, showlegend=False)
        fig_bar.update_xaxes(tickprefix="R$ ", tickformat=".2s")
        fig_bar.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_b2:
        # Treemap por segmento e banco
        fig_tree = px.treemap(
            banco_agg, path=["segmento", col_banco], values="volume",
            color="ticket", color_continuous_scale=[
                [0, f"{COR_PRIMARIA}40"], [0.5, COR_SECUNDARIA], [1, COR_ACENTO]
            ],
            custom_data=["operacoes","ticket"]
        )
        fig_tree.update_traces(
            hovertemplate="<b>%{label}</b><br>Volume: R$ %{value:,.0f}<br>Operações: %{customdata[0]:,.0f}<br>Ticket: R$ %{customdata[1]:,.0f}<extra></extra>",
            textfont=dict(family="DM Sans", size=11)
        )
        fig_tree.update_layout(
            title=f"Treemap por Segmento – Top {top_n}",
            template=PLOTLY_TEMPLATE, height=480,
            paper_bgcolor="rgba(0,0,0,0)", font=dict(color=COR_TEXTO),
            coloraxis_colorbar=dict(title="Ticket Médio", tickprefix="R$ ", tickformat=".2s")
        )
        st.plotly_chart(fig_tree, use_container_width=True)

    # Métricas de concentração
    col_hhi1, col_hhi2, col_hhi3, col_hhi4 = st.columns(4)
    label_hhi, badge_hhi, _ = interpretar_hhi(hhi_val)
    cr3 = (banco_agg.nlargest(3,"volume")["volume"].sum() / banco_agg["volume"].sum() * 100) if len(banco_agg) >= 3 else 0
    cr5 = (banco_agg.nlargest(5,"volume")["volume"].sum() / banco_agg["volume"].sum() * 100) if len(banco_agg) >= 5 else 0
    gini_bancos = calcular_gini(banco_agg["volume"])

    for col_, title_, val_, sub_ in [
        (col_hhi1, "📐 HHI", f"{hhi_val:.0f}", label_hhi),
        (col_hhi2, "🏆 CR3", f"{cr3:.1f}%", "3 maiores bancos"),
        (col_hhi3, "🏆 CR5", f"{cr5:.1f}%", "5 maiores bancos"),
        (col_hhi4, "⚖️ Gini", f"{gini_bancos:.3f}", "desigualdade bancária"),
    ]:
        with col_:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">{title_}</div>
                <div class="kpi-value">{val_}</div>
                <div class="kpi-sub">{sub_}</div>
            </div>
            """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# TAB 3: ANÁLISE REGIONAL
# ─────────────────────────────────────────────────────────────
with tabs[2]:
    st.markdown('<div class="section-header">🗺️ Análise Regional e por Estado</div>', unsafe_allow_html=True)

    col_r1, col_r2 = st.columns([1,1])
    with col_r1:
        reg_comp = df_f.groupby("regiao").agg(
            volume=("volume_operacoes","sum"),
            operacoes=("numero_operacoes","sum")
        ).reset_index()
        reg_comp["ticket"] = reg_comp["volume"] / reg_comp["operacoes"].replace(0,np.nan)
        reg_comp["pct"]    = reg_comp["volume"] / reg_comp["volume"].sum() * 100

        fig_reg = go.Figure()
        fig_reg.add_trace(go.Bar(
            name="Volume", x=reg_comp["regiao"], y=reg_comp["volume"],
            marker_color=PALETTE_QUAL[:len(reg_comp)],
            hovertemplate="<b>%{x}</b><br>Volume: R$ %{y:,.0f}<extra></extra>",
            yaxis="y"
        ))
        fig_reg.add_trace(go.Scatter(
            name="Ticket Médio", x=reg_comp["regiao"], y=reg_comp["ticket"],
            mode="lines+markers", line=dict(color=COR_ATENCAO, width=2.5),
            marker=dict(size=9, symbol="diamond"),
            hovertemplate="<b>%{x}</b><br>Ticket: R$ %{y:,.0f}<extra></extra>",
            yaxis="y2"
        ))
        fig_reg.update_layout(
            title="Volume e Ticket Médio por Região",
            yaxis=dict(title="Volume (R$)", tickprefix="R$ ", tickformat=".2s", color=COR_TEXTO),
            yaxis2=dict(title="Ticket Médio (R$)", overlaying="y", side="right",
                        tickprefix="R$ ", tickformat=",.0f", color=COR_ATENCAO),
            barmode="group"
        )
        layout_base(fig_reg, height=400)
        st.plotly_chart(fig_reg, use_container_width=True)

    with col_r2:
        # Pizza com furo (donut)
        fig_donut = go.Figure(go.Pie(
            labels=reg_comp["regiao"], values=reg_comp["volume"],
            hole=0.55, sort=True,
            marker=dict(colors=PALETTE_QUAL[:len(reg_comp)], line=dict(color=COR_FUNDO, width=2)),
            hovertemplate="<b>%{label}</b><br>R$ %{value:,.0f}<br>%{percent}<extra></extra>",
            textinfo="label+percent", textfont=dict(size=11)
        ))
        fig_donut.add_annotation(
            text=f"<b>{fmt_brl(total_volume)}</b><br><span style='font-size:10px'>Total</span>",
            x=0.5, y=0.5, showarrow=False, font=dict(size=13, color=COR_TEXTO)
        )
        fig_donut.update_layout(title="Participação Regional no Volume Total", showlegend=True)
        layout_base(fig_donut, height=400)
        st.plotly_chart(fig_donut, use_container_width=True)

    # Evolução regional ao longo do tempo
    if "unidade_federacao" in df_f.columns:
        evolucao_reg = df_f.groupby(["data_base","regiao"])["volume_operacoes"].sum().reset_index()
        fig_area = go.Figure()
        for i, reg in enumerate(evolucao_reg["regiao"].unique()):
            grp = evolucao_reg[evolucao_reg["regiao"]==reg]
            fig_area.add_trace(go.Scatter(
                x=grp["data_base"], y=grp["volume_operacoes"],
                name=reg, stackgroup="one",
                line=dict(color=PALETTE_QUAL[i % len(PALETTE_QUAL)], width=0.5),
                fillcolor=f"{PALETTE_QUAL[i % len(PALETTE_QUAL)]}AA",
                hovertemplate=f"<b>{reg}</b><br>%{{x|%b/%Y}}<br>R$ %{{y:,.0f}}<extra></extra>"
            ))
        fig_area.update_layout(title="Evolução do Volume por Região (Área Empilhada)")
        layout_base(fig_area, height=380)
        fig_area.update_yaxes(tickprefix="R$ ", tickformat=".2s")
        st.plotly_chart(fig_area, use_container_width=True)

# ─────────────────────────────────────────────────────────────
# TAB 4: ANÁLISE AVANÇADA
# ─────────────────────────────────────────────────────────────
with tabs[3]:
    st.markdown('<div class="section-header">🔬 Análise Avançada & Machine Learning</div>', unsafe_allow_html=True)

    col_av1, col_av2 = st.columns([3,2])
    with col_av1:
        fig_cluster, dados_cluster = clusterizar_bancos(df_f, col_banco)
        if fig_cluster:
            st.plotly_chart(fig_cluster, use_container_width=True)
        else:
            st.info("Dados insuficientes para clusterização (mínimo 3 instituições com >100 operações).")

    with col_av2:
        if dados_cluster is not None:
            st.markdown("**Sumário por Cluster**")
            sumario = dados_cluster.groupby("cluster_nome").agg(
                Instituições=("numero_operacoes","count"),
                Vol_Total=("volume_operacoes","sum"),
                Ops_Media=("numero_operacoes","mean"),
                Ticket_Medio=("ticket_medio","mean")
            ).reset_index()
            for _, row in sumario.iterrows():
                st.markdown(f"""
                <div class="insight-box">
                    <div class="insight-label">{row['cluster_nome']}</div>
                    <div class="insight-text">
                    <b>{int(row['Instituições'])}</b> inst. · Vol: <b>{fmt_brl(row['Vol_Total'])}</b><br>
                    Ops médias: <b>{fmt_num(row['Ops_Media'])}</b> · Ticket: <b>{fmt_brl(row['Ticket_Medio'])}</b>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # Análise de correlação: crescimento vs concentração por período
    st.markdown("**Dispersão: Ticket Médio × Participação de Mercado por Banco**")
    scatter_df = df_f.groupby(col_banco).agg(
        volume=("volume_operacoes","sum"),
        operacoes=("numero_operacoes","sum")
    ).reset_index()
    scatter_df["ticket"] = scatter_df["volume"] / scatter_df["operacoes"].replace(0,np.nan)
    scatter_df["market_share"] = scatter_df["operacoes"] / scatter_df["operacoes"].sum() * 100
    scatter_df["segmento"] = scatter_df[col_banco].apply(classificar_banco)
    scatter_df = scatter_df.dropna().query("operacoes > 50")

    cor_map = {s: PALETTE_QUAL[i % len(PALETTE_QUAL)] for i, s in enumerate(scatter_df["segmento"].unique())}
    fig_sc = go.Figure()
    for seg, grp in scatter_df.groupby("segmento"):
        fig_sc.add_trace(go.Scatter(
            x=grp["market_share"], y=grp["ticket"],
            mode="markers", name=seg,
            marker=dict(
                size=np.log1p(grp["volume"]/grp["volume"].max()+0.01)*18+8,
                color=cor_map.get(seg, "#888"),
                opacity=0.8, line=dict(width=1, color=COR_BORDA)
            ),
            hovertemplate="<b>%{customdata}</b><br>Market Share: %{x:.2f}%<br>Ticket: R$ %{y:,.0f}<extra></extra>",
            customdata=grp[col_banco]
        ))
    fig_sc.update_layout(title="Ticket Médio vs. Market Share (tamanho = volume)")
    layout_base(fig_sc, height=420)
    fig_sc.update_xaxes(title="Market Share (%)")
    fig_sc.update_yaxes(title="Ticket Médio (R$)", tickprefix="R$ ", tickformat=",")
    st.plotly_chart(fig_sc, use_container_width=True)

# ─────────────────────────────────────────────────────────────
# TAB 5: DISTRIBUIÇÃO & PARETO
# ─────────────────────────────────────────────────────────────
with tabs[4]:
    st.markdown('<div class="section-header">📊 Distribuição & Análise de Pareto</div>', unsafe_allow_html=True)

    pareto_df = calcular_pareto(
        df_f.groupby(col_banco)["volume_operacoes"].sum().reset_index(), "volume_operacoes"
    )
    pct_80 = (pareto_df["pct_acum"] <= 80).sum()

    fig_pareto = make_subplots(specs=[[{"secondary_y": True}]])
    fig_pareto.add_trace(go.Bar(
        x=pareto_df[col_banco].str[:20],
        y=pareto_df["volume_operacoes"],
        name="Volume",
        marker=dict(
            color=pareto_df["volume_operacoes"],
            colorscale=[[0,f"{COR_PRIMARIA}40"],[1,COR_PRIMARIA]],
            showscale=False
        ),
        hovertemplate="<b>%{x}</b><br>Volume: R$ %{y:,.0f}<extra></extra>"
    ), secondary_y=False)
    fig_pareto.add_trace(go.Scatter(
        x=pareto_df[col_banco].str[:20],
        y=pareto_df["pct_acum"],
        name="% Acumulado",
        line=dict(color=COR_ATENCAO, width=2.5),
        marker=dict(size=5),
        hovertemplate="%{x}<br>Acumulado: %{y:.1f}%<extra></extra>"
    ), secondary_y=True)
    fig_pareto.add_hline(y=80, line_dash="dot", line_color=COR_ALERTA, secondary_y=True,
                         annotation_text="80%", annotation_font_color=COR_ALERTA)
    fig_pareto.update_layout(title=f"Curva de Pareto – Volume por Instituição | {pct_80} bancos = 80% do volume")
    layout_base(fig_pareto, height=460)
    fig_pareto.update_yaxes(title_text="Volume (R$)", tickprefix="R$ ", tickformat=".2s", secondary_y=False)
    fig_pareto.update_yaxes(title_text="% Acumulado", ticksuffix="%", secondary_y=True, showgrid=False)
    fig_pareto.update_xaxes(tickangle=-45)
    st.plotly_chart(fig_pareto, use_container_width=True)

    st.markdown(f"""
    <div class="insight-box">
        <div class="insight-label">📐 Lei de Pareto – Interpretação</div>
        <div class="insight-text">
        <b>{pct_80} instituições</b> respondem por <b>80%</b> do volume total renegociado no período selecionado.
        Gini bancário de <b>{gini_bancos:.3f}</b> confirma estrutura {'altamente' if gini_bancos > 0.6 else 'moderadamente'} concentrada.
        Avaliar incentivos para ampliar participação das instituições menores pode diversificar o acesso ao programa.
        </div>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# RODAPÉ
# ============================================================
st.markdown("---")
st.markdown(f"""
<div style="text-align:center; font-size:0.75rem; color:{COR_TEXTO_SUB}; padding:0.5rem 0 1rem;">
    🏦 Desenrola Brasil · Painel Executivo &nbsp;|&nbsp;
    Fonte: <a href="https://www.bcb.gov.br/estatisticas/scr" target="_blank" style="color:{COR_PRIMARIA};">Banco Central do Brasil – SCR</a>
    &nbsp;|&nbsp; Desenvolvido com Streamlit + Plotly
</div>
""", unsafe_allow_html=True)
