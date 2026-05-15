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
# TEMA
# ============================================================
if "tema" not in st.session_state:
    st.session_state.tema = "claro"

T = st.session_state.tema
if T == "claro":
    COR_FUNDO       = "#F8FAFC"
    COR_CARD        = "#FFFFFF"
    COR_TEXTO       = "#1E293B"
    COR_BORDA       = "#E2E8F0"
    COR_PRIMARIA    = "#0F172A"
    COR_SECUNDARIA  = "#2563EB"
    COR_SUCESSO     = "#16A34A"
    COR_ALERTA      = "#DC2626"
    COR_ATENCAO     = "#D97706"
    PLOTLY_TEMPLATE = "plotly_white"
    COR_GRID        = "rgba(0,0,0,0.06)"
else:
    COR_FUNDO       = "#0B0F19"
    COR_CARD        = "#111827"
    COR_TEXTO       = "#F3F4F6"
    COR_BORDA       = "#1F2937"
    COR_PRIMARIA    = "#38BDF8"
    COR_SECUNDARIA  = "#60A5FA"
    COR_SUCESSO     = "#34D399"
    COR_ALERTA      = "#F87171"
    COR_ATENCAO     = "#FBBF24"
    PLOTLY_TEMPLATE = "plotly_dark"
    COR_GRID        = "rgba(255,255,255,0.06)"

# ============================================================
# CSS
# ============================================================
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;600;700&family=IBM+Plex+Mono:wght@400;600&display=swap');
html, body, .stApp {{
    background-color: {COR_FUNDO};
    color: {COR_TEXTO};
    font-family: 'IBM Plex Sans', sans-serif;
}}
.block-container {{ padding: 1rem 1.5rem; }}
.kpi-card {{
    background: {COR_CARD};
    border-left: 4px solid {COR_SECUNDARIA};
    border-radius: 10px;
    padding: 0.9rem 1.1rem;
    margin-bottom: 0.5rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.07);
}}
.kpi-title {{ font-size: 0.65rem; text-transform: uppercase; letter-spacing: 0.08em; color: #64748B; font-weight: 600; }}
.kpi-value {{ font-size: 1.6rem; font-weight: 700; color: {COR_TEXTO}; margin-top: 0.15rem; font-family: 'IBM Plex Mono', monospace; }}
.kpi-sub   {{ font-size: 0.72rem; color: #94A3B8; margin-top: 0.15rem; }}
.insight-box {{
    background: {COR_CARD};
    border: 1px solid {COR_BORDA};
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.6rem;
}}
.insight-label {{ font-size: 0.65rem; text-transform: uppercase; letter-spacing: 0.08em; color: #94A3B8; font-weight: 600; margin-bottom: 0.3rem; }}
.insight-text  {{ font-size: 0.88rem; color: {COR_TEXTO}; line-height: 1.55; }}
.badge {{ padding: 2px 8px; border-radius: 20px; font-weight: 600; font-size: 0.68rem; display: inline-block; }}
.badge-low  {{ background: rgba(22,163,74,0.12);  color: #16A34A; }}
.badge-mid  {{ background: rgba(217,119,6,0.12);  color: #D97706; }}
.badge-high {{ background: rgba(220,38,38,0.12);  color: #DC2626; }}
.dq-card {{
    background: {COR_CARD};
    border: 1px solid {COR_BORDA};
    border-radius: 10px;
    padding: 0.8rem 1rem;
    font-size: 0.8rem;
}}
.mono {{ font-family: 'IBM Plex Mono', monospace; }}
</style>
""", unsafe_allow_html=True)

# ============================================================
# UTILITÁRIOS
# ============================================================
def fmt_brl(v):
    if pd.isna(v) or v == 0: return "R$ 0"
    if v >= 1e9:  return f"R$ {v/1e9:.1f}B".replace(".", ",")
    if v >= 1e6:  return f"R$ {v/1e6:.1f}M".replace(".", ",")
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

@st.cache_data
def calcular_hhi(df, col):
    total = df[col].sum()
    return 0 if total == 0 else ((df[col]/total)**2).sum()*10000

@st.cache_data
def interpretar_hhi(hhi):
    if hhi < 1500:
        return "Mercado Competitivo (HHI < 1.500)", "badge-low", "Baixo risco de concentração bancária – saudável para o consumidor."
    if hhi < 2500:
        return "Concentração Moderada (HHI 1.500–2.500)", "badge-mid", "Atenção: poucos bancos lideram o programa. Monitorar tendência."
    return "Altamente Concentrado (HHI > 2.500)", "badge-high", "Risco sistêmico elevado: oligopólio pode reduzir acesso ao crédito."

@st.cache_data
def calcular_pareto(df, col):
    df_s = df.sort_values(col, ascending=False).reset_index(drop=True)
    df_s["pct_acum"] = (df_s[col].cumsum() / df_s[col].sum())*100
    return df_s

def layout_base(fig, height=450, showlegend=True):
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=height,
        margin=dict(l=50, r=40, t=60, b=50),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=COR_TEXTO, family="IBM Plex Sans", size=12),
        hovermode="x unified",
        showlegend=showlegend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=11))
    )
    fig.update_xaxes(showgrid=False, color=COR_TEXTO, title_font_size=12, linecolor=COR_BORDA)
    fig.update_yaxes(showgrid=True, gridcolor=COR_GRID, color=COR_TEXTO, title_font_size=12)
    # Habilita barra de ferramentas para exportar PNG
    fig.update_layout(autosize=True)
    return fig

# ============================================================
# FIX 1 – PROJEÇÃO HOLT-WINTERS (com segurança)
# ============================================================
@st.cache_data
def projetar_holt_winters(series_volume: pd.Series, datas: pd.Series, periodos=3):
    if len(series_volume) < 4:
        return None, None, None, None
    modelo = ExponentialSmoothing(
        series_volume.values,
        trend="add",
        seasonal=None,
        initialization_method="estimated"
    ).fit(optimized=True)
    previsao = modelo.forecast(periodos)
    datas_futuras = pd.date_range(datas.max(), periods=periodos+1, freq="MS")[1:]
    sigma = np.std(modelo.resid)
    lower = previsao - 1.96*sigma
    upper = previsao + 1.96*sigma
    return datas_futuras, previsao, lower, upper

# ============================================================
# FIX 2 – K-MEANS COM ROTULAGEM DINÂMICA
# ============================================================
@st.cache_data
def clusterizar_bancos(df, col_banco):
    dados = df.groupby(col_banco).agg(
        numero_operacoes=("numero_operacoes","sum"),
        volume_operacoes=("volume_operacoes","sum")
    ).reset_index()
    dados["ticket_medio"] = dados["volume_operacoes"] / dados["numero_operacoes"]
    dados = dados[dados["numero_operacoes"] > 100]

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
        if alto_vol and not alto_ticket:
            return "Alto Volume / Baixo Ticket"
        if not alto_vol and alto_ticket:
            return "Baixo Volume / Alto Ticket"
        return "Perfil Equilibrado"

    dados["cluster_nome"] = dados["cluster"].map(rotulo)

    fig = go.Figure()
    cores_cluster = {"Alto Volume / Baixo Ticket": COR_SECUNDARIA,
                     "Baixo Volume / Alto Ticket": COR_ATENCAO,
                     "Perfil Equilibrado": COR_SUCESSO}
    for nome, grp in dados.groupby("cluster_nome"):
        # Escala logarítmica do tamanho da bolha para evitar distorção
        size_norm = np.log1p(grp["volume_operacoes"] / grp["volume_operacoes"].max()) * 30 + 8
        fig.add_trace(go.Scatter(
            x=grp["numero_operacoes"], y=grp["ticket_medio"],
            mode="markers", name=nome,
            marker=dict(size=size_norm,
                        color=cores_cluster.get(nome, "#64748B"), opacity=0.8,
                        line=dict(width=1, color=COR_BORDA)),
            hovertemplate="<b>%{customdata}</b><br>Operações: %{x:,.0f}<br>Ticket Médio: R$ %{y:,.2f}<extra></extra>",
            customdata=grp[col_banco]
        ))
    fig.update_layout(
        title=dict(text="Agrupamento de Instituições por Comportamento (K-Means)", font_size=14),
        xaxis_title="Número de Operações",
        yaxis_title="Ticket Médio (R$)"
    )
    layout_base(fig, height=500)
    return fig, dados

# ============================================================
# FIX 3 – QUALIDADE DE DADOS
# ============================================================
def calcular_data_quality(df_original, df_limpo):
    total_raw = len(df_original) if df_original is not None else len(df_limpo)
    total_limpo = len(df_limpo)
    completude = (df_limpo.notna().sum() / len(df_limpo)) * 100
    periodo_min = df_limpo["data_base"].min().strftime("%m/%Y") if not df_limpo["data_base"].isna().all() else "N/D"
    periodo_max = df_limpo["data_base"].max().strftime("%m/%Y") if not df_limpo["data_base"].isna().all() else "N/D"
    meses_cobertos = df_limpo["data_base"].nunique()
    return {
        "total_registros": total_limpo,
        "registros_descartados": total_raw - total_limpo,
        "completude_volume": completude.get("volume_operacoes", 100),
        "completude_operacoes": completude.get("numero_operacoes", 100),
        "periodo_inicio": periodo_min,
        "periodo_fim": periodo_max,
        "meses_cobertos": meses_cobertos,
        "ultima_data": periodo_max
    }

# ============================================================
# INSIGHTS AUTOMÁTICOS (cacheado)
# ============================================================
@st.cache_data
def gerar_alertas(evolucao, hhi, ticket_medio_geral):
    alertas = []
    if len(evolucao) >= 2:
        cresc_ultimo = evolucao["crescimento"].dropna().iloc[-1] if len(evolucao["crescimento"].dropna()) > 0 else 0
        if cresc_ultimo < -15:
            alertas.append(("error", "🔴 Queda Abrupta", f"Volume caiu {cresc_ultimo:.1f}% no último mês – investigar sazonalidade ou mudança de política."))
        elif cresc_ultimo < -5:
            alertas.append(("warning", "🟡 Desaceleração", f"Queda de {cresc_ultimo:.1f}% sinaliza perda de ritmo do programa."))
        elif cresc_ultimo > 20:
            alertas.append(("success", "🟢 Aceleração Forte", f"Crescimento de +{cresc_ultimo:.1f}% – verificar campanhas ou novas adesões."))

    if hhi > 2500:
        alertas.append(("error", "🔴 Concentração Elevada", "HHI acima de 2.500: mercado oligopolizado. Risco de acesso desigual."))
    elif hhi > 1500:
        alertas.append(("warning", "🟡 Concentração Moderada", "HHI entre 1.500–2.500: monitorar para evitar aumento de concentração."))

    if ticket_medio_geral > 5000:
        alertas.append(("info", "ℹ️ Ticket Alto", f"Ticket médio de {fmt_brl(ticket_medio_geral)} sugere perfil de dívida maior – avaliar capacidade de pagamento."))

    return alertas

# ============================================================
# CARREGAMENTO DE DADOS
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
                        df[col].astype(str).str.replace(".","",regex=False).str.replace(",",".",regex=False),
                        errors="coerce"
                    )
            df["data_base"] = pd.to_datetime(df["data_base"].astype(str), format="%Y%m", errors="coerce")
            df["tipo_banco"] = df["nome_conglomerado_financeiro"].apply(classificar_banco)
            df["regiao"]     = df["unidade_federacao"].apply(agrupar_regiao)
            df_limpo = df.dropna(subset=["volume_operacoes","numero_operacoes"])
            return df, df_limpo
        except Exception:
            continue
    return None, None

df_raw, df = carregar_dados()
if df is None:
    st.error("Erro ao carregar dados. Verifique se 'dados_desenrola.csv' está presente.")
    st.stop()

dq = calcular_data_quality(df_raw, df)

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("### ⚙️ Controles")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("☀️ Claro", use_container_width=True):
            st.session_state.tema = "claro"; st.rerun()
    with c2:
        if st.button("🌙 Escuro", use_container_width=True):
            st.session_state.tema = "escuro"; st.rerun()

    st.markdown("---")
    tipos   = sorted(df["tipo_desenrola"].unique())
    tipo    = st.multiselect("Faixa do Programa", tipos, default=tipos)
    regioes = sorted(df["regiao"].unique())
    regiao  = st.multiselect("Região", regioes, default=regioes)
    bancos  = sorted(df["tipo_banco"].unique())
    banco   = st.multiselect("Segmento", bancos, default=bancos)
    if st.button("🔄 Limpar Filtros", use_container_width=True): st.rerun()

    # ── Qualidade dos dados no sidebar ──────────────────────
    st.markdown("---")
    st.markdown("### 📋 Qualidade dos Dados")
    st.markdown(f"""
    <div class="dq-card">
    <b>Registros válidos:</b> <span class="mono">{fmt_num(dq['total_registros'])}</span><br>
    <b>Descartados:</b> <span class="mono">{fmt_num(dq['registros_descartados'])}</span><br>
    <b>Período:</b> <span class="mono">{dq['periodo_inicio']} → {dq['periodo_fim']}</span><br>
    <b>Meses cobertos:</b> <span class="mono">{dq['meses_cobertos']}</span><br>
    <b>Completude volume:</b> <span class="mono">{dq['completude_volume']:.1f}%</span><br>
    <b>Completude ops:</b> <span class="mono">{dq['completude_operacoes']:.1f}%</span>
    </div>
    """, unsafe_allow_html=True)

df_f = df[df["tipo_desenrola"].isin(tipo) & df["regiao"].isin(regiao) & df["tipo_banco"].isin(banco)]

if df_f.empty:
    st.warning("Nenhum dado encontrado. Ajuste os filtros.")
    st.stop()

# ============================================================
# KPIs PRINCIPAIS
# ============================================================
total_volume    = df_f["volume_operacoes"].sum()
total_ops       = df_f["numero_operacoes"].sum()
ticket_medio    = total_volume / total_ops if total_ops > 0 else 0
num_inst        = df_f["nome_conglomerado_financeiro"].nunique()
col_banco       = "nome_conglomerado_financeiro"

evolucao_global = df_f.groupby("data_base")["volume_operacoes"].sum().sort_index()
delta_str = ""
if len(evolucao_global) >= 2:
    delta_pct = (evolucao_global.iloc[-1] / evolucao_global.iloc[-2] - 1)*100
    delta_str = f"{'▲' if delta_pct > 0 else '▼'} {abs(delta_pct):.1f}% vs mês anterior"

k1, k2, k3, k4 = st.columns(4)
with k1:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">Volume de Renegociação</div>
        <div class="kpi-value">{fmt_brl(total_volume)}</div>
        <div class="kpi-sub">{delta_str}</div>
    </div>""", unsafe_allow_html=True)
with k2:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">Total de Contratos</div>
        <div class="kpi-value">{fmt_num(total_ops)}</div>
    </div>""", unsafe_allow_html=True)
with k3:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">Ticket Médio</div>
        <div class="kpi-value">{fmt_brl(ticket_medio)}</div>
        <div class="kpi-sub">Volume / Contratos</div>
    </div>""", unsafe_allow_html=True)
with k4:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">Instituições Atuantes</div>
        <div class="kpi-value">{fmt_num(num_inst)}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("---")

# ============================================================
# ABAS
# ============================================================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈 Evolução e Projeção",
    "🏦 Mercado e Concentração",
    "🗺️ Distribuição Regional",
    "🔬 Análise por Segmento",
    "🤖 Agrupamento (ML)",
    "📊 Conclusões e Exportação"
])

# ──────────────────────────────────────────────────────────
# TAB 1  –  EVOLUÇÃO + HOLT-WINTERS
# ──────────────────────────────────────────────────────────
with tab1:
    evolucao = df_f.groupby("data_base").agg(
        volume_operacoes=("volume_operacoes","sum"),
        numero_operacoes=("numero_operacoes","sum")
    ).reset_index()
    evolucao["crescimento"]  = evolucao["volume_operacoes"].pct_change()*100
    evolucao["media_movel3"] = evolucao["volume_operacoes"].rolling(3, min_periods=1).mean()

    market_hhi = df_f.groupby(col_banco)["numero_operacoes"].sum().reset_index()
    hhi_val    = calcular_hhi(market_hhi, "numero_operacoes")
    alertas    = gerar_alertas(evolucao, hhi_val, ticket_medio)

    if alertas:
        st.markdown("#### 🔔 Alertas Automáticos")
        for tipo_alerta, titulo_al, msg_al in alertas:
            fn = getattr(st, tipo_alerta, st.info)
            fn(f"**{titulo_al}:** {msg_al}")

    st.markdown("#### Histórico de Volume e Contratos")
    fig_ev = make_subplots(rows=2, cols=1, shared_xaxes=True,
                           row_heights=[0.65, 0.35],
                           vertical_spacing=0.08,
                           subplot_titles=["Volume de Renegociação (R$)", "Número de Contratos"])
    fig_ev.add_trace(go.Scatter(
        x=evolucao["data_base"], y=evolucao["volume_operacoes"],
        name="Volume Mensal", mode="lines+markers",
        line=dict(color=COR_SECUNDARIA, width=2.5),
        marker=dict(size=5)
    ), row=1, col=1)
    fig_ev.add_trace(go.Scatter(
        x=evolucao["data_base"], y=evolucao["media_movel3"],
        name="Média Móvel 3M", mode="lines",
        line=dict(color=COR_ATENCAO, dash="dash", width=1.8)
    ), row=1, col=1)
    fig_ev.add_trace(go.Bar(
        x=evolucao["data_base"], y=evolucao["numero_operacoes"],
        name="Contratos", marker_color=COR_SECUNDARIA, opacity=0.6
    ), row=2, col=1)
    layout_base(fig_ev, height=500)
    st.plotly_chart(fig_ev, use_container_width=True, config={'displayModeBar': True})

    if len(evolucao) >= 4:
        st.markdown("#### Projeção Holt-Winters (3 meses) — com intervalo de confiança 95%")
        st.caption("Suavização exponencial com tendência – captura aceleração/desaceleração melhor que regressão linear.")
        datas_fut, prev, lower, upper = projetar_holt_winters(
            evolucao["volume_operacoes"], evolucao["data_base"]
        )
        if datas_fut is not None:
            fig_prev = go.Figure()
            fig_prev.add_trace(go.Scatter(
                x=evolucao["data_base"], y=evolucao["volume_operacoes"],
                name="Realizado", mode="lines+markers",
                line=dict(color=COR_SECUNDARIA, width=2.5)
            ))
            fig_prev.add_trace(go.Scatter(
                x=list(datas_fut)+list(datas_fut[::-1]),
                y=list(upper)+list(lower[::-1]),
                fill="toself", fillcolor="rgba(217,119,6,0.15)",
                line=dict(color="rgba(0,0,0,0)"),
                name="IC 95%", hoverinfo="skip"
            ))
            fig_prev.add_trace(go.Scatter(
                x=datas_fut, y=prev,
                name="Projeção HW", mode="lines+markers",
                line=dict(color=COR_ATENCAO, dash="dot", width=2),
                marker=dict(symbol="diamond", size=8)
            ))
            layout_base(fig_prev, height=420)
            st.plotly_chart(fig_prev, use_container_width=True, config={'displayModeBar': True})

            col_p1, col_p2, col_p3 = st.columns(3)
            for col_p, (d, v) in zip([col_p1,col_p2,col_p3], zip(datas_fut, prev)):
                with col_p:
                    st.markdown(f"""
                    <div class="insight-box">
                        <div class="insight-label">Projeção {d.strftime('%b/%Y')}</div>
                        <div class="kpi-value" style="font-size:1.2rem">{fmt_brl(v)}</div>
                    </div>""", unsafe_allow_html=True)

    st.markdown("#### Variação Mensal Recente")
    tab_var = evolucao[["data_base","volume_operacoes","crescimento"]].tail(6).copy()
    tab_var["data_base"] = tab_var["data_base"].dt.strftime("%m/%Y")
    tab_var["crescimento"] = tab_var["crescimento"].apply(lambda x: f"{x:+.2f}%" if pd.notna(x) else "—")
    tab_var["volume_operacoes"] = tab_var["volume_operacoes"].apply(fmt_brl)
    tab_var.columns = ["Mês","Volume","Variação"]
    st.dataframe(tab_var, use_container_width=True, hide_index=True)

    st.markdown("#### Comparativo Ano a Ano (YoY)")
    yoy = df_f.copy()
    yoy["ano"] = yoy["data_base"].dt.year
    yoy["mes"] = yoy["data_base"].dt.month
    yoy_data = yoy.groupby(["ano","mes"])["volume_operacoes"].sum().reset_index()
    yoy_data = yoy_data[yoy_data["ano"] >= yoy_data["ano"].max()-1]

    fig_yoy = go.Figure()
    cores_ano = [COR_SECUNDARIA, COR_ATENCAO, COR_SUCESSO]
    for i, ano in enumerate(sorted(yoy_data["ano"].unique())):
        d = yoy_data[yoy_data["ano"]==ano]
        fig_yoy.add_trace(go.Scatter(
            x=d["mes"], y=d["volume_operacoes"],
            name=str(ano), mode="lines+markers",
            line=dict(color=cores_ano[i%3], width=2.5)
        ))
    meses_label = ['Jan','Fev','Mar','Abr','Mai','Jun','Jul','Ago','Set','Out','Nov','Dez']
    fig_yoy.update_xaxes(tickvals=list(range(1,13)), ticktext=meses_label)
    layout_base(fig_yoy, height=400)
    st.plotly_chart(fig_yoy, use_container_width=True, config={'displayModeBar': True})

# ──────────────────────────────────────────────────────────
# TAB 2  –  MERCADO E CONCENTRAÇÃO
# ──────────────────────────────────────────────────────────
with tab2:
    market = df_f.groupby(col_banco)["numero_operacoes"].sum().sort_values(ascending=False).reset_index()
    hhi    = calcular_hhi(market, "numero_operacoes")
    class_hhi, css_hhi, expl_hhi = interpretar_hhi(hhi)

    col_hhi, col_pareto = st.columns(2)
    with col_hhi:
        st.markdown("#### Índice de Concentração de Mercado (HHI)")
        st.markdown(f"""
        <div class="insight-box">
            <div class="insight-label">Herfindahl-Hirschman Index</div>
            <div class="kpi-value" style="font-size:2.2rem">{hhi:.0f}</div>
            <span class="badge {css_hhi}">{class_hhi}</span>
            <p style="margin-top:0.7rem; font-size:0.82rem; color:#94A3B8; line-height:1.5">{expl_hhi}</p>
            <p style="font-size:0.75rem; color:#64748B">
            <b>Como ler:</b> HHI &lt; 1.500 = competitivo | 1.500–2.500 = moderado | &gt; 2.500 = concentrado.<br>
            Calculado sobre número de contratos entre as principais instituições.
            </p>
        </div>""", unsafe_allow_html=True)

    with col_pareto:
        st.markdown("#### Análise de Pareto (80/20)")
        pareto = calcular_pareto(market.head(10), "numero_operacoes")
        fig_p = make_subplots(specs=[[{"secondary_y":True}]])
        fig_p.add_trace(go.Bar(
            x=pareto[col_banco], y=pareto["numero_operacoes"],
            name="Contratos", marker_color=COR_SECUNDARIA, opacity=0.85
        ), secondary_y=False)
        fig_p.add_trace(go.Scatter(
            x=pareto[col_banco], y=pareto["pct_acum"],
            name="% Acumulado", mode="lines+markers",
            line=dict(color=COR_ALERTA, width=2.5),
            marker=dict(size=7)
        ), secondary_y=True)
        fig_p.add_hline(y=80, line_dash="dash", line_color=COR_ATENCAO,
                        annotation_text="80%", annotation_position="top right",
                        secondary_y=True)
        fig_p.update_yaxes(title_text="Contratos", secondary_y=False)
        fig_p.update_yaxes(title_text="% Acumulado", secondary_y=True, range=[0,105])
        layout_base(fig_p, height=400)
        st.plotly_chart(fig_p, use_container_width=True, config={'displayModeBar': True})

    st.markdown("#### Ranking de Mercado – Top 15 Instituições")
    ranking = market.head(15).copy()
    total_r = ranking["numero_operacoes"].sum()
    ranking["% Individual"] = (ranking["numero_operacoes"]/total_r*100).round(1)
    ranking["% Acumulado"]  = ranking["% Individual"].cumsum().round(1)
    ranking["Volume (R$)"]  = df_f.groupby(col_banco)["volume_operacoes"].sum().reindex(ranking[col_banco].values).values
    ranking["Volume (R$)"]  = ranking["Volume (R$)"].apply(fmt_brl)
    ranking["numero_operacoes"] = ranking["numero_operacoes"].apply(fmt_num)
    ranking.columns = ["Instituição","Contratos","% Individual","% Acumulado","Volume"]
    st.dataframe(ranking, use_container_width=True, hide_index=True)

    lider = market.iloc[0][col_banco]
    part_lider = market.iloc[0]["numero_operacoes"]/market["numero_operacoes"].sum()*100
    top3_pct   = market.head(3)["numero_operacoes"].sum()/market["numero_operacoes"].sum()*100

    st.markdown(f"""
    <div class="insight-box">
        <div class="insight-label">📖 Interpretação Executiva</div>
        <div class="insight-text">
        <b>{lider}</b> lidera com <b>{part_lider:.1f}%</b> dos contratos do programa.
        As 3 maiores instituições juntas concentram <b>{top3_pct:.1f}%</b> do total —
        {'o que indica dependência crítica de poucos players e risco de interrupção sistêmica.' if top3_pct > 60 else 'distribuição relativamente equilibrada, com risco sistêmico moderado.'}
        </div>
    </div>""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────
# TAB 3  –  REGIONAL (com seletor de período no heatmap)
# ──────────────────────────────────────────────────────────
with tab3:
    reg_data = df_f.groupby("regiao")["volume_operacoes"].sum().reset_index()
    total_reg = reg_data["volume_operacoes"].sum()
    reg_data["pct"] = (reg_data["volume_operacoes"]/total_reg*100).round(1)
    regiao_lider = reg_data.sort_values("volume_operacoes",ascending=False).iloc[0]

    col_r1, col_r2 = st.columns(2)
    with col_r1:
        st.markdown("#### Participação Regional por Volume")
        fig_donut = go.Figure(go.Pie(
            labels=reg_data["regiao"], values=reg_data["volume_operacoes"],
            hole=0.55,
            textinfo="percent+label",
            marker=dict(colors=[COR_SECUNDARIA, COR_ATENCAO, COR_SUCESSO, COR_ALERTA, "#64748B"],
                        line=dict(color=COR_FUNDO, width=2))
        ))
        fig_donut.add_annotation(text=f"{regiao_lider['pct']:.0f}%<br>{regiao_lider['regiao']}",
                                 x=0.5, y=0.5, showarrow=False,
                                 font=dict(size=14, color=COR_TEXTO))
        layout_base(fig_donut, height=420)
        st.plotly_chart(fig_donut, use_container_width=True, config={'displayModeBar': True})

        st.markdown(f"""
        <div class="insight-box">
            <div class="insight-label">📖 Leitura Regional</div>
            <div class="insight-text">
            A região <b>{regiao_lider['regiao']}</b> concentra <b>{regiao_lider['pct']:.1f}%</b>
            das renegociações, indicando forte dependência regional do programa.
            {'Isso sugere que o impacto do Desenrola é desigualmente distribuído no território nacional.' if regiao_lider['pct'] > 40 else 'A distribuição está relativamente balanceada entre as regiões.'}
            </div>
        </div>""", unsafe_allow_html=True)

    with col_r2:
        st.markdown("#### Evolução do Volume por Região (Heatmap)")
        # Seletor de período para o heatmap
        datas_heat = sorted(df_f["data_base"].dt.strftime("%Y-%m").unique())
        if len(datas_heat) > 12:
            default_heat = datas_heat[-12:]
            heat_period = st.multiselect("Selecione os meses para o heatmap", datas_heat, default=default_heat)
        else:
            heat_period = datas_heat

        heat_df = df_f[df_f["data_base"].dt.strftime("%Y-%m").isin(heat_period)]
        heat = heat_df.groupby(["regiao", heat_df["data_base"].dt.strftime("%Y-%m")])["volume_operacoes"].sum().reset_index()
        heat.columns = ["regiao","mes","volume"]
        pivot = heat.pivot(index="regiao", columns="mes", values="volume").fillna(0)/1e6
        if not pivot.empty:
            fig_heat = go.Figure(go.Heatmap(
                z=pivot.values, x=pivot.columns.tolist(), y=pivot.index.tolist(),
                colorscale="Blues", text=np.round(pivot.values,1),
                texttemplate="%{text}M",
                colorbar=dict(title="R$ Milhões")
            ))
            layout_base(fig_heat, height=420)
            st.plotly_chart(fig_heat, use_container_width=True, config={'displayModeBar': True})
        else:
            st.info("Nenhum dado para o período selecionado.")

    st.markdown("#### Líderes de Mercado por Estado")
    uf_banco = df_f.groupby(["unidade_federacao", col_banco])["numero_operacoes"].sum().reset_index()
    uf_banco = uf_banco.sort_values(["unidade_federacao","numero_operacoes"], ascending=[True,False])
    top3_uf  = uf_banco.groupby("unidade_federacao").head(3).reset_index(drop=True)
    top3_uf["rank"] = top3_uf.groupby("unidade_federacao").cumcount()+1
    top3_uf["disp"] = top3_uf.apply(lambda x: f"{x[col_banco]} ({fmt_num(x['numero_operacoes'])})", axis=1)
    piv_uf = top3_uf.pivot_table(index="unidade_federacao", columns="rank", values="disp", aggfunc="first").reset_index()
    piv_uf.columns = ["UF","🥇 Líder","🥈 2º","🥉 3º"]
    st.dataframe(piv_uf, use_container_width=True, hide_index=True)

    st.markdown("#### Ticket Médio por Região e Tipo de Banco")
    cruzado = df_f.groupby(["regiao","tipo_banco"]).agg(
        numero_operacoes=("numero_operacoes","sum"),
        volume_operacoes=("volume_operacoes","sum")
    ).reset_index()
    cruzado["ticket_medio"] = cruzado["volume_operacoes"]/cruzado["numero_operacoes"]
    fig_tree = px.treemap(cruzado, path=["regiao","tipo_banco"], values="volume_operacoes",
                          color="ticket_medio", color_continuous_scale="Blues",
                          hover_data={"ticket_medio":":.2f"})
    layout_base(fig_tree, height=480)
    st.plotly_chart(fig_tree, use_container_width=True, config={'displayModeBar': True})

# ──────────────────────────────────────────────────────────
# TAB 4  –  SEGMENTOS
# ──────────────────────────────────────────────────────────
with tab4:
    dispersao = df_f.groupby(col_banco).agg(
        numero_operacoes=("numero_operacoes","sum"),
        volume_operacoes=("volume_operacoes","sum"),
        tipo_banco=("tipo_banco","first")
    ).reset_index()
    dispersao["ticket_medio"] = dispersao["volume_operacoes"]/dispersao["numero_operacoes"]
    dispersao = dispersao[dispersao["numero_operacoes"] > 1000]

    st.markdown("#### Dispersão por Segmento Institucional")
    fig_disp = go.Figure()
    cores_seg = {"Banco Digital":"#10B981","Banco Tradicional":"#2563EB",
                 "Banco de Investimento":"#D97706","Outras Instituições":"#64748B"}
    for seg, grp in dispersao.groupby("tipo_banco"):
        size_norm = np.log1p(grp["volume_operacoes"] / dispersao["volume_operacoes"].max()) * 40 + 8
        fig_disp.add_trace(go.Scatter(
            x=grp["numero_operacoes"], y=grp["ticket_medio"],
            mode="markers", name=seg,
            marker=dict(size=size_norm, color=cores_seg.get(seg,"#64748B"), opacity=0.75,
                        line=dict(width=1, color=COR_BORDA)),
            hovertemplate="<b>%{customdata}</b><br>Operações: %{x:,.0f}<br>Ticket: R$ %{y:,.0f}<extra></extra>",
            customdata=grp[col_banco]
        ))
    layout_base(fig_disp, height=480)
    st.plotly_chart(fig_disp, use_container_width=True, config={'displayModeBar': True})

    st.markdown("#### Comparativo: Operações vs Ticket Médio por Segmento")
    comp = df_f.groupby("tipo_banco").agg(
        numero_operacoes=("numero_operacoes","sum"),
        volume_operacoes=("volume_operacoes","sum")
    ).reset_index()
    comp["ticket_medio"] = comp["volume_operacoes"]/comp["numero_operacoes"]
    comp["pct_ops"]      = (comp["numero_operacoes"]/comp["numero_operacoes"].sum()*100).round(1)

    fig_comp = make_subplots(rows=1, cols=2,
                             subplot_titles=["Número de Contratos","Ticket Médio (R$)"])
    colors_comp = [cores_seg.get(s,"#64748B") for s in comp["tipo_banco"]]
    fig_comp.add_trace(go.Bar(
        x=comp["tipo_banco"], y=comp["numero_operacoes"],
        text=comp["numero_operacoes"].apply(fmt_num), textposition="outside",
        marker_color=colors_comp, name="Contratos"
    ), row=1, col=1)
    fig_comp.add_trace(go.Bar(
        x=comp["tipo_banco"], y=comp["ticket_medio"],
        text=comp["ticket_medio"].apply(fmt_brl), textposition="outside",
        marker_color=colors_comp, name="Ticket", showlegend=False
    ), row=1, col=2)
    layout_base(fig_comp, height=420, showlegend=False)
    st.plotly_chart(fig_comp, use_container_width=True, config={'displayModeBar': True})

    st.markdown("#### Distribuição de Contratos (Detecção de Outliers)")
    st.caption("Caixas com bigodes longos indicam instituições com comportamento atípico – meses de pico ou campanhas pontuais.")
    top10 = df_f.groupby(col_banco)["numero_operacoes"].sum().nlargest(10).index
    df_top = df_f[df_f[col_banco].isin(top10)]
    fig_box = go.Figure()
    for inst in top10:
        d = df_top[df_top[col_banco]==inst]["numero_operacoes"]
        fig_box.add_trace(go.Box(y=d, name=inst, boxpoints="outliers",
                                  marker_color=COR_SECUNDARIA, line_color=COR_SECUNDARIA))
    fig_box.update_layout(showlegend=False)
    layout_base(fig_box, height=450)
    st.plotly_chart(fig_box, use_container_width=True, config={'displayModeBar': True})

# ──────────────────────────────────────────────────────────
# TAB 5  –  K-MEANS CORRIGIDO
# ──────────────────────────────────────────────────────────
with tab5:
    st.markdown("#### Agrupamento de Instituições (K-Means Dinâmico)")
    st.markdown("""
    <div class="insight-box">
        <div class="insight-label">Como funciona</div>
        <div class="insight-text">
        O algoritmo K-Means agrupa automaticamente as instituições por similaridade de comportamento,
        considerando <b>volume de contratos</b> e <b>ticket médio</b>. Os rótulos são atribuídos
        <b>dinamicamente</b> com base nas médias de cada grupo — sem assumir ordem fixa dos clusters,
        o que garante consistência na interpretação independentemente da execução.
        </div>
    </div>""", unsafe_allow_html=True)

    fig_cl, cluster_data = clusterizar_bancos(df_f, col_banco)
    if fig_cl:
        st.plotly_chart(fig_cl, use_container_width=True, config={'displayModeBar': True})

        if cluster_data is not None:
            st.markdown("#### Resumo por Grupo")
            resumo = cluster_data.groupby("cluster_nome").agg(
                Instituições=("nome_conglomerado_financeiro","count"),
                Contratos_Médio=("numero_operacoes","mean"),
                Ticket_Médio=("ticket_medio","mean")
            ).reset_index()
            resumo["Contratos_Médio"] = resumo["Contratos_Médio"].apply(fmt_num)
            resumo["Ticket_Médio"]    = resumo["Ticket_Médio"].apply(fmt_brl)
            resumo.columns = ["Grupo","Qtd Instituições","Média de Contratos","Ticket Médio"]
            st.dataframe(resumo, use_container_width=True, hide_index=True)

            with st.expander("Ver todas as instituições por grupo"):
                for nome_cl, grp in cluster_data.groupby("cluster_nome"):
                    st.markdown(f"**{nome_cl}**")
                    st.dataframe(
                        grp[[col_banco,"numero_operacoes","ticket_medio"]].rename(columns={
                            col_banco:"Instituição","numero_operacoes":"Contratos","ticket_medio":"Ticket Médio"
                        }),
                        use_container_width=True, hide_index=True
                    )
    else:
        st.info("Dados insuficientes para realizar o agrupamento.")

# ──────────────────────────────────────────────────────────
# TAB 6  –  CONCLUSÕES E EXPORTAÇÃO
# ──────────────────────────────────────────────────────────
with tab6:
    st.markdown("#### 📌 Narrativa Executiva")

    cresc_medio = evolucao["crescimento"].mean() if not evolucao["crescimento"].isna().all() else 0
    regiao_top  = reg_data.sort_values("volume_operacoes",ascending=False).iloc[0]
    lider_banco = market.iloc[0][col_banco]
    part_banco  = market.iloc[0]["numero_operacoes"]/market["numero_operacoes"].sum()*100
    corr_val    = df_f[["numero_operacoes","volume_operacoes"]].corr().iloc[0,1]
    df_saz      = df_f.copy()
    df_saz["mes"] = df_f["data_base"].dt.month
    mes_pico    = df_saz.groupby("mes")["volume_operacoes"].sum().idxmax()
    nomes_meses = ['Janeiro','Fevereiro','Março','Abril','Maio','Junho',
                   'Julho','Agosto','Setembro','Outubro','Novembro','Dezembro']
    _, _, expl_hhi2 = interpretar_hhi(hhi_val)

    insights = [
        ("📈 Tendência Geral",
         f"O programa apresentou {'crescimento' if cresc_medio > 0 else 'retração'} médio de "
         f"**{abs(cresc_medio):.2f}% ao mês** no período analisado, "
         f"{'sugerindo expansão sustentada do alcance do programa.' if cresc_medio > 0 else 'sinalizando perda de momentum que merece atenção.'}"),

        ("🗺️ Concentração Regional",
         f"A região **{regiao_top['regiao']}** concentra **{regiao_top['pct']:.1f}%** do volume total de renegociações. "
         f"{'Essa dependência indica que o impacto social do Desenrola é desigualmente distribuído no território nacional, com risco de exclusão financeira em outras regiões.' if regiao_top['pct'] > 40 else 'A distribuição regional é relativamente equilibrada.'}"),

        ("🏦 Liderança Bancária",
         f"**{lider_banco}** responde por **{part_banco:.1f}%** de todos os contratos. "
         f"{'Nível de concentração elevado — a continuidade do programa fica vulnerável às decisões de poucos players.' if part_banco > 30 else 'Participação razoável, sem domínio excessivo.'}"),

        ("📊 Concentração de Mercado (HHI)",
         f"O índice HHI calculado é **{hhi_val:.0f}**. {expl_hhi2}"),

        ("📅 Sazonalidade",
         f"O mês de **{nomes_meses[mes_pico-1]}** historicamente registra o maior volume de renegociações. "
         f"Campanhas e incentivos concentrados nesse período podem maximizar o alcance do programa."),

        ("🔗 Correlação Operações × Volume",
         f"A correlação entre número de contratos e volume financeiro é **{corr_val:.3f}** "
         f"({'forte' if corr_val > 0.7 else 'moderada' if corr_val > 0.4 else 'fraca'}). "
         f"{'Isso indica que mais contratos resultam proporcionalmente em maior volume — perfil homogêneo de dívidas.' if corr_val > 0.7 else 'Variação no ticket médio por tipo de contrato é relevante para a análise.'}")
    ]

    for titulo_ins, texto_ins in insights:
        st.markdown(f"""
        <div class="insight-box">
            <div class="insight-label">{titulo_ins}</div>
            <div class="insight-text">{texto_ins}</div>
        </div>""", unsafe_allow_html=True)

    alertas_fin = gerar_alertas(evolucao, hhi_val, ticket_medio)
    if alertas_fin:
        st.markdown("#### 🔔 Pontos de Atenção")
        for tipo_a, titulo_a, msg_a in alertas_fin:
            fn = getattr(st, tipo_a, st.info)
            fn(f"**{titulo_a}:** {msg_a}")

    st.markdown("---")
    st.markdown("#### 📥 Exportação")
    csv = df_f.to_csv(index=False).encode("utf-8")
    relatorio_txt = f"""RELATÓRIO DESENROLA BRASIL – {datetime.now().strftime('%d/%m/%Y %H:%M')}
{'='*55}
INDICADORES GERAIS
  Volume Total          : {fmt_brl(total_volume)}
  Total de Contratos    : {fmt_num(total_ops)}
  Ticket Médio          : {fmt_brl(ticket_medio)}
  Instituições Atuantes : {fmt_num(num_inst)}
  Período               : {dq['periodo_inicio']} → {dq['periodo_fim']}

CONCENTRAÇÃO
  HHI                   : {hhi_val:.0f}
  Classificação         : {class_hhi}

DESTAQUES
  Banco Líder           : {lider_banco} ({part_banco:.1f}% dos contratos)
  Região Líder          : {regiao_top['regiao']} ({regiao_top['pct']:.1f}% do volume)
  Crescimento Médio     : {cresc_medio:+.2f}% / mês
  Pico de Sazonalidade  : {nomes_meses[mes_pico-1]}
  Correlação Ops×Volume : {corr_val:.3f}

QUALIDADE DOS DADOS
  Registros Válidos     : {fmt_num(dq['total_registros'])}
  Registros Descartados : {fmt_num(dq['registros_descartados'])}
  Completude Volume     : {dq['completude_volume']:.1f}%
  Completude Operações  : {dq['completude_operacoes']:.1f}%

Fonte: Banco Central do Brasil – Sistema de Informações de Crédito (SCR)
Metodologia: Holt-Winters para projeção | K-Means dinâmico para segmentação | HHI para concentração
"""

    col_exp1, col_exp2 = st.columns(2)
    with col_exp1:
        st.download_button("📥 Dados Filtrados (CSV)", data=csv,
                           file_name=f"desenrola_{datetime.now().strftime('%Y%m%d')}.csv",
                           mime="text/csv", use_container_width=True)
    with col_exp2:
        st.download_button("📝 Relatório Executivo (TXT)", data=relatorio_txt,
                           file_name=f"relatorio_desenrola_{datetime.now().strftime('%Y%m%d')}.txt",
                           mime="text/plain", use_container_width=True)

# ============================================================
# RODAPÉ
# ============================================================
st.markdown("---")
st.markdown(f"""
<p style='text-align:center; color:#64748B; font-size:0.68rem;'>
    Dashboard Desenrola Brasil · Fonte: BCB/SCR ·
    Projeção: Holt-Winters · Segmentação: K-Means dinâmico · HHI ·
    Última atualização: {dq.get('ultima_data', 'N/D')}
</p>""", unsafe_allow_html=True)
