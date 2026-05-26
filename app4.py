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

st.set_page_config(
    page_title="Desenrola Brasil – Painel Executivo",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Tema ──────────────────────────────────────────────────────
if "tema" not in st.session_state:
    st.session_state.tema = "claro"
T = st.session_state.tema

if T == "claro":
    BG      = "#F5F4F0"
    CARD    = "#FFFFFF"
    TXT     = "#1C2321"
    TXT2    = "#5C6B5E"
    BORDA   = "#D6D9D2"
    P1      = "#2D6A4F"   # verde-esmeralda
    P2      = "#40916C"
    P3      = "#52B788"
    VERDE   = "#1B7A52"
    VERM    = "#C1440E"
    AMBER   = "#E09F3E"
    AZUL    = "#2196A6"
    TPLOTE  = "plotly_white"
    GRID    = "rgba(45,106,79,0.08)"
    SIDEBAR = "#2D6A4F"
    SDTXT   = "#FFFFFF"
else:
    BG      = "#0D1117"
    CARD    = "#161B22"
    TXT     = "#E6EDF3"
    TXT2    = "#8B949E"
    BORDA   = "#30363D"
    P1      = "#3FB68C"
    P2      = "#56D6A8"
    P3      = "#79E4C0"
    VERDE   = "#3FB68C"
    VERM    = "#FF6B47"
    AMBER   = "#F4A535"
    AZUL    = "#58A6FF"
    TPLOTE  = "plotly_dark"
    GRID    = "rgba(63,182,140,0.08)"
    SIDEBAR = "#0D1117"
    SDTXT   = "#E6EDF3"

# paleta qualitativa (cores fixas para gráficos, sem TXT2)
CORES_GRAFICOS = [P1, AMBER, VERM, AZUL, P2, P3, "#7B5EA7", "#C97C3A"]

# ── CSS ───────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&family=Playfair+Display:wght@700&display=swap');
html,body,.stApp{{background:{BG};color:{TXT};font-family:'DM Sans',sans-serif;}}
.block-container{{padding:1.2rem 1.8rem;max-width:1600px;}}
[data-testid="stSidebar"]{{background:{SIDEBAR} !important;border-right:1px solid {BORDA};}}
[data-testid="stSidebar"] *{{color:{SDTXT} !important;}}
.kpi{{background:{CARD};border-left:4px solid {P1};border-radius:12px;padding:1rem 1.2rem;margin-bottom:.6rem;box-shadow:0 2px 8px rgba(0,0,0,.06);}}
.kpi.am{{border-left-color:{AMBER};}} .kpi.vd{{border-left-color:{VERDE};}} .kpi.az{{border-left-color:{AZUL};}} .kpi.vm{{border-left-color:{VERM};}}
.kpi-ic{{font-size:1rem;display:block;margin-bottom:.25rem;}}
.kpi-tt{{font-size:.6rem;text-transform:uppercase;letter-spacing:.1em;color:{TXT2};font-weight:700;}}
.kpi-vl{{font-size:1.6rem;font-weight:700;color:{TXT};font-family:'DM Mono',monospace;line-height:1.1;margin-top:.15rem;}}
.kpi-sb{{font-size:.7rem;color:{TXT2};margin-top:.2rem;}}
.ibox{{background:{CARD};border:1px solid {BORDA};border-top:3px solid {P1};border-radius:10px;padding:1rem 1.2rem;margin-bottom:.7rem;}}
.ibox.warn{{border-top-color:{AMBER};}} .ibox.dang{{border-top-color:{VERM};}} .ibox.info{{border-top-color:{AZUL};}}
.il{{font-size:.6rem;text-transform:uppercase;letter-spacing:.1em;color:{TXT2};font-weight:700;margin-bottom:.3rem;}}
.it{{font-size:.87rem;color:{TXT};line-height:1.6;}}
.sh{{font-family:'Playfair Display',serif;font-size:1.25rem;font-weight:700;color:{TXT};margin:1.4rem 0 .7rem;padding-bottom:.35rem;border-bottom:2px solid {P1}44;}}
.al{{padding:.55rem 1rem;border-radius:8px;margin-bottom:.45rem;font-size:.83rem;display:flex;align-items:center;gap:.5rem;}}
.al.er{{background:{VERM}18;border-left:3px solid {VERM};color:{VERM};}}
.al.wa{{background:{AMBER}18;border-left:3px solid {AMBER};color:{AMBER};}}
.al.ok{{background:{VERDE}18;border-left:3px solid {VERDE};color:{VERDE};}}
.al.in{{background:{AZUL}18;border-left:3px solid {AZUL};color:{AZUL};}}
.dqc{{background:{CARD};border:1px solid {BORDA};border-radius:10px;padding:.9rem 1rem;font-size:.78rem;line-height:1.75;}}
.mono{{font-family:'DM Mono',monospace;font-size:.82rem;}}
</style>
""", unsafe_allow_html=True)

# ── Utilitários ───────────────────────────────────────────────
def rgba(hex_color, a=0.15):
    h = hex_color.lstrip('#')
    r,g,b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
    return f"rgba({r},{g},{b},{a})"

def fmt_brl(v):
    if pd.isna(v) or v==0: return "R$ 0"
    if v>=1e9: return f"R$ {v/1e9:.2f}B".replace(".",",")
    if v>=1e6: return f"R$ {v/1e6:.1f}M".replace(".",",")
    return "R$ {:,.0f}".format(v).replace(",",".")

def fmt_num(v):
    if pd.isna(v): return "0"
    return "{:,}".format(int(v)).replace(",",".")

def fmt_pct(v):
    s = f"{v:+.1f}%" if v>=0 else f"{v:.1f}%"
    return s

def class_banco(nome):
    n = re.sub(r'\s*-\s*PRUDENCIAL$','',str(nome).upper().strip())
    if any(x in n for x in ["NUBANK","INTER","C6","NEON","ORIGINAL","PAN","NEXT"]): return "Banco Digital"
    if any(x in n for x in ["ITAU","BRADESCO","SANTANDER","CAIXA","BANCO DO BRASIL","BB"]): return "Banco Tradicional"
    if any(x in n for x in ["BTG","XP","MODAL","GENIAL"]): return "Banco de Investimento"
    if any(x in n for x in ["SICOOB","SICREDI","CRESOL"]): return "Cooperativa"
    return "Outras Instituições"

def class_regiao(uf):
    m = {"Norte":["AC","AM","AP","PA","RO","RR","TO"],
         "Nordeste":["AL","BA","CE","MA","PB","PE","PI","RN","SE"],
         "Centro-Oeste":["DF","GO","MS","MT"],
         "Sudeste":["ES","MG","RJ","SP"],
         "Sul":["PR","RS","SC"]}
    for r,l in m.items():
        if uf in l: return r
    return "Não Identificado"

def hhi(df, col):
    t = df[col].sum()
    return 0 if t==0 else ((df[col]/t)**2).sum()*10000

def gini(s):
    a = np.sort(s.dropna().values)
    n = len(a)
    if n==0 or a.sum()==0: return 0
    return (2*np.sum(np.arange(1,n+1)*a)/(n*a.sum())-(n+1)/n)

def base_layout(fig, h=440, leg=True):
    fig.update_layout(
        template=TPLOTE, height=h,
        margin=dict(l=50,r=40,t=55,b=50),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TXT,family="DM Sans",size=12),
        hovermode="x unified",
        showlegend=leg,
        legend=dict(orientation="h",yanchor="bottom",y=1.02,
                    xanchor="right",x=1,font=dict(size=11),
                    bgcolor="rgba(0,0,0,0)")
    )
    fig.update_xaxes(showgrid=False,color=TXT,linecolor=BORDA,tickfont=dict(size=11))
    fig.update_yaxes(showgrid=True,gridcolor=GRID,color=TXT,tickfont=dict(size=11))
    return fig

# ── Dados ─────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def load():
    for enc in ["utf-8","latin1","cp1252"]:
        try:
            df = pd.read_csv("dados_desenrola.csv",sep=";",encoding=enc,low_memory=False)
            df.columns = df.columns.str.lower().str.strip()
            for c in ["numero_operacoes","volume_operacoes"]:
                if c in df.columns:
                    df[c] = pd.to_numeric(
                        df[c].astype(str).str.replace(".",""  ,regex=False)
                                         .str.replace(",",".",regex=False),errors="coerce")
            df["data_base"] = pd.to_datetime(df["data_base"].astype(str),format="%Y%m",errors="coerce")
            df["tipo_banco"] = df["nome_conglomerado_financeiro"].apply(class_banco)
            df["regiao"]     = df["unidade_federacao"].apply(class_regiao)
            raw = df.copy()
            df  = df.dropna(subset=["volume_operacoes","numero_operacoes"])
            return raw, df
        except Exception:
            continue
    return None, None

df_raw, df = load()
if df is None:
    st.error("Erro ao carregar 'dados_desenrola.csv'. Verifique o arquivo e recarregue.")
    st.stop()

# qualidade
total_raw  = len(df_raw)
total_ok   = len(df)
pct_drop   = (total_raw-total_ok)/total_raw*100 if total_raw>0 else 0
comp_vol   = df["volume_operacoes"].notna().mean()*100
comp_ops   = df["numero_operacoes"].notna().mean()*100
dt_ini     = df["data_base"].min().strftime("%m/%Y") if df["data_base"].notna().any() else "N/D"
dt_fim     = df["data_base"].max().strftime("%m/%Y") if df["data_base"].notna().any() else "N/D"
meses      = df["data_base"].nunique()

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style="padding:.5rem 0 1rem;border-bottom:1px solid {BORDA}55;margin-bottom:1rem;">
    <div style="font-family:'Playfair Display',serif;font-size:1.05rem;font-weight:700;">🏦 Desenrola Brasil</div>
    <div style="font-size:.72rem;opacity:.7;margin-top:.2rem;">Painel Executivo · BCB/SCR</div>
    </div>""",unsafe_allow_html=True)

    st.markdown("**⚙️ Aparência**")
    c1,c2 = st.columns(2)
    with c1:
        if st.button("☀️ Claro",use_container_width=True):
            st.session_state.tema="claro"; st.rerun()
    with c2:
        if st.button("🌙 Escuro",use_container_width=True):
            st.session_state.tema="escuro"; st.rerun()

    st.markdown("---")
    st.markdown("**🔍 Filtros**")
    tipos   = sorted(df["tipo_desenrola"].unique())
    sel_tip = st.multiselect("Faixa do Programa",tipos,default=tipos)
    regs    = sorted(df["regiao"].unique())
    sel_reg = st.multiselect("Região",regs,default=regs)
    segs    = sorted(df["tipo_banco"].unique())
    sel_seg = st.multiselect("Segmento",segs,default=segs)

    datas_d = sorted(df["data_base"].dropna().unique())
    if len(datas_d)>1:
        i0,i1 = st.select_slider("Período",options=list(range(len(datas_d))),
            value=(0,len(datas_d)-1),
            format_func=lambda i: pd.Timestamp(datas_d[i]).strftime("%m/%Y"))
        d_ini, d_fim = datas_d[i0], datas_d[i1]
    else:
        d_ini, d_fim = datas_d[0], datas_d[-1]

    if st.button("🔄 Limpar",use_container_width=True): st.rerun()

    st.markdown("---")
    st.markdown("**📋 Qualidade dos Dados**")
    dq_cor = VERDE if pct_drop<5 else (AMBER if pct_drop<15 else VERM)
    st.markdown(f"""
    <div class="dqc">
    <b>Registros válidos:</b> <span class="mono">{fmt_num(total_ok)}</span><br>
    <b>Descartados:</b> <span class="mono" style="color:{dq_cor}">{fmt_num(total_raw-total_ok)} ({pct_drop:.1f}%)</span><br>
    <b>Período:</b> <span class="mono">{dt_ini} → {dt_fim}</span><br>
    <b>Meses cobertos:</b> <span class="mono">{meses}</span><br>
    <b>Completude volume:</b> <span class="mono">{comp_vol:.1f}%</span><br>
    <b>Completude ops:</b> <span class="mono">{comp_ops:.1f}%</span>
    </div>""",unsafe_allow_html=True)

# ── Filtro ────────────────────────────────────────────────────
dff = df[
    df["tipo_desenrola"].isin(sel_tip) &
    df["regiao"].isin(sel_reg) &
    df["tipo_banco"].isin(sel_seg) &
    (df["data_base"]>=pd.Timestamp(d_ini)) &
    (df["data_base"]<=pd.Timestamp(d_fim))
]

if dff.empty:
    st.warning("⚠️ Nenhum dado com os filtros selecionados.")
    st.stop()

COL_B = "nome_conglomerado_financeiro"

# ── Cabeçalho ─────────────────────────────────────────────────
st.markdown(f"""
<div style="margin-bottom:.5rem;">
  <span style="font-family:'Playfair Display',serif;font-size:1.9rem;font-weight:700;">🏦 Desenrola Brasil</span>
  <span style="font-size:.95rem;opacity:.55;margin-left:.7rem;">Painel Executivo</span>
</div>""",unsafe_allow_html=True)
st.caption("Monitoramento de renegociação de dívidas · Banco Central do Brasil (SCR)")

# ── KPIs ──────────────────────────────────────────────────────
vol_tot  = dff["volume_operacoes"].sum()
ops_tot  = dff["numero_operacoes"].sum()
ticket   = vol_tot/ops_tot if ops_tot>0 else 0
n_inst   = dff[COL_B].nunique()
n_uf     = dff["unidade_federacao"].nunique() if "unidade_federacao" in dff.columns else 0

evol_g = (dff.groupby("data_base")["volume_operacoes"].sum()
            .reset_index().sort_values("data_base"))
evol_g["mom"] = evol_g["volume_operacoes"].pct_change()*100
mom_last = evol_g["mom"].dropna().iloc[-1] if len(evol_g["mom"].dropna())>0 else 0
mom_cor  = VERDE if mom_last>=0 else VERM
mom_str  = f'<span style="color:{mom_cor};font-size:.72rem;font-weight:600;">{fmt_pct(mom_last)} vs mês ant.</span>'

k1,k2,k3,k4,k5 = st.columns(5)
for col_, ic, tt, vl, sb, cls_ in [
    (k1,"💵","Volume Renegociado",fmt_brl(vol_tot),mom_str,""),
    (k2,"📄","Contratos",fmt_num(ops_tot),"operações registradas","az"),
    (k3,"🎫","Ticket Médio",fmt_brl(ticket),"Volume ÷ Contratos","am"),
    (k4,"🏛️","Instituições",fmt_num(n_inst),"financeiras participantes",""),
    (k5,"📍","Estados",fmt_num(n_uf),"unidades da federação","vd"),
]:
    with col_:
        st.markdown(f"""
        <div class="kpi {cls_}">
          <span class="kpi-ic">{ic}</span>
          <div class="kpi-tt">{tt}</div>
          <div class="kpi-vl">{vl}</div>
          <div class="kpi-sb">{sb}</div>
        </div>""",unsafe_allow_html=True)

# ── Métricas de concentração ───────────────────────────────────
b_agg = (dff.groupby(COL_B)["numero_operacoes"].sum().reset_index())
hhi_v = hhi(b_agg,"numero_operacoes")

reg_v = (dff.groupby("regiao")["volume_operacoes"].sum())
gini_r = gini(reg_v)

def hhi_label(h):
    if h<1500: return "Competitivo","ok","Baixo risco de concentração – ambiente saudável."
    if h<2500: return "Mod. Concentrado","wa","Atenção: poucos bancos lideram o programa."
    return "Alt. Concentrado","er","Risco sistêmico: oligopólio pode limitar acesso."

hhi_lbl, hhi_cls, hhi_exp = hhi_label(hhi_v)

# ── Alertas ───────────────────────────────────────────────────
alertas = []
if mom_last < -15: alertas.append(("er",f"🔴 Queda Abrupta – volume caiu {mom_last:.1f}% no último mês."))
elif mom_last < -5: alertas.append(("wa",f"🟡 Desaceleração – queda de {mom_last:.1f}%."))
elif mom_last > 20: alertas.append(("ok",f"🟢 Aceleração Forte – crescimento de +{mom_last:.1f}%."))
elif mom_last > 0: alertas.append(("ok",f"🟢 Crescimento Estável – +{mom_last:.1f}% no mês."))
if hhi_v>2500: alertas.append(("er","🔴 Concentração Elevada – HHI > 2.500 (oligopólio)."))
elif hhi_v>1500: alertas.append(("wa","🟡 Concentração Moderada – HHI 1.500–2.500."))
else: alertas.append(("ok","🟢 Mercado Competitivo – HHI < 1.500."))
if gini_r>0.7: alertas.append(("er",f"🔴 Alta Desigualdade Regional – Gini = {gini_r:.2f}."))
elif gini_r>0.5: alertas.append(("wa",f"🟡 Desigualdade Regional – Gini = {gini_r:.2f}."))
if ticket>8000: alertas.append(("wa",f"🟡 Ticket Alto – {fmt_brl(ticket)}: pode excluir devedores menores."))

if alertas:
    st.markdown('<div class="sh">⚡ Alertas Automáticos</div>',unsafe_allow_html=True)
    ac = st.columns(min(len(alertas),3))
    for i,(cls,msg) in enumerate(alertas):
        with ac[i%3]:
            st.markdown(f'<div class="al {cls}">{msg}</div>',unsafe_allow_html=True)

# ── Resumo executivo ──────────────────────────────────────────
reg_df = dff.groupby("regiao")["volume_operacoes"].sum().reset_index()
reg_df["pct"] = reg_df["volume_operacoes"]/reg_df["volume_operacoes"].sum()*100
lider_reg = reg_df.loc[reg_df["volume_operacoes"].idxmax()]
top2_pct  = reg_df.nlargest(2,"volume_operacoes")["pct"].sum()

if ops_tot>0:
    lider_b   = b_agg.loc[b_agg["numero_operacoes"].idxmax(), COL_B]
    part_b    = b_agg["numero_operacoes"].max()/ops_tot*100
    top5_part = b_agg.nlargest(5,"numero_operacoes")["numero_operacoes"].sum()/ops_tot*100
else:
    lider_b, part_b, top5_part = "N/A", 0, 0

cresc_med = evol_g["mom"].dropna().mean() if len(evol_g)>1 else 0
tend_txt = (f"crescimento médio de <b>+{cresc_med:.1f}%</b>/mês – expansão sustentada"
            if cresc_med>=0
            else f"retração média de <b>{cresc_med:.1f}%</b>/mês – perda de momentum")

st.markdown('<div class="sh">📌 Resumo Executivo</div>',unsafe_allow_html=True)
rc1,rc2,rc3 = st.columns(3)
with rc1:
    st.markdown(f"""<div class="ibox">
    <div class="il">🎯 Concentração Regional</div>
    <div class="it"><b>{lider_reg['regiao']}</b> lidera com <b>{lider_reg['pct']:.1f}%</b> do volume.
    As 2 maiores regiões somam <b>{top2_pct:.1f}%</b>, indicando oportunidade de expansão nas demais.</div>
    </div>""",unsafe_allow_html=True)
with rc2:
    cls2 = "dang" if part_b>40 else ("warn" if part_b>25 else "")
    st.markdown(f"""<div class="ibox {cls2}">
    <div class="il">🏦 Liderança Bancária</div>
    <div class="it"><b>{lider_b}</b> detém <b>{part_b:.1f}%</b> dos contratos.
    Top 5 bancos = <b>{top5_part:.1f}%</b>. HHI: <b>{hhi_v:.0f}</b> – {hhi_exp}</div>
    </div>""",unsafe_allow_html=True)
with rc3:
    cls3 = "dang" if cresc_med<-5 else ("warn" if cresc_med<0 else "")
    st.markdown(f"""<div class="ibox {cls3}">
    <div class="il">📈 Dinâmica Temporal</div>
    <div class="it">Programa apresenta {tend_txt}.
    Ticket médio <b>{fmt_brl(ticket)}</b> · Gini regional <b>{gini_r:.2f}</b>.</div>
    </div>""",unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────
tabs = st.tabs(["📈 Evolução Temporal","🏛️ Concentração Bancária","🗺️ Análise Regional","🔬 Análise Avançada","📊 Pareto"])

# ════════════════════════════════════════════════
# TAB 1 – EVOLUÇÃO TEMPORAL
# ════════════════════════════════════════════════
with tabs[0]:
    st.markdown('<div class="sh">📈 Evolução Temporal do Programa</div>',unsafe_allow_html=True)

    evol_tp = (dff.groupby(["data_base","tipo_desenrola"])["volume_operacoes"]
               .sum().reset_index())

    col_ev1, col_ev2 = st.columns([2,1])

    with col_ev1:
        fig1 = go.Figure()
        tipos_u = sorted(evol_tp["tipo_desenrola"].unique())
        for i,tp in enumerate(tipos_u):
            g = evol_tp[evol_tp["tipo_desenrola"]==tp]
            cor = CORES_GRAFICOS[i % len(CORES_GRAFICOS)]
            fig1.add_trace(go.Scatter(
                x=g["data_base"], y=g["volume_operacoes"],
                mode="lines+markers", name=str(tp),
                line=dict(color=cor, width=2.5),
                marker=dict(size=6, color=cor),
                hovertemplate=f"<b>{tp}</b><br>%{{x|%b/%Y}}<br>R$ %{{y:,.0f}}<extra></extra>"
            ))

        # projeção Holt-Winters
        try:
            serie = evol_g["volume_operacoes"]
            if len(serie)>=4:
                hw = ExponentialSmoothing(
                    serie.values, trend="add", seasonal=None,
                    initialization_method="estimated").fit(optimized=True)
                prev = hw.forecast(3)
                dt_fut = pd.date_range(evol_g["data_base"].max(), periods=4, freq="MS")[1:]
                sigma  = float(np.std(hw.resid))
                low    = [float(v) for v in prev-1.96*sigma]
                upp    = [float(v) for v in prev+1.96*sigma]
                xband  = list(dt_fut)+list(dt_fut[::-1])
                yband  = upp + low[::-1]
                fig1.add_trace(go.Scatter(
                    x=xband, y=yband,
                    fill="toself",
                    fillcolor=rgba(AMBER, 0.15),
                    line=dict(color="rgba(0,0,0,0)"),
                    name="IC 95%", showlegend=True,
                    hoverinfo="skip"
                ))
                fig1.add_trace(go.Scatter(
                    x=list(dt_fut), y=[float(v) for v in prev],
                    mode="lines+markers", name="Projeção (HW)",
                    line=dict(color=AMBER, width=2, dash="dash"),
                    marker=dict(size=7, symbol="diamond", color=AMBER),
                    hovertemplate="<b>Projeção</b><br>%{x|%b/%Y}<br>R$ %{y:,.0f}<extra></extra>"
                ))
        except Exception:
            pass

        fig1.update_layout(title="Volume Renegociado por Faixa + Projeção 3 meses")
        base_layout(fig1, h=420)
        fig1.update_yaxes(tickprefix="R$ ", tickformat=",.0f")
        st.plotly_chart(fig1, use_container_width=True)

    with col_ev2:
        cresc_v = evol_g.dropna(subset=["mom"])
        cores_b = [VERDE if v>=0 else VERM for v in cresc_v["mom"]]
        fig2 = go.Figure(go.Bar(
            x=cresc_v["data_base"], y=cresc_v["mom"],
            marker_color=cores_b,
            hovertemplate="%{x|%b/%Y}<br>MoM: %{y:.1f}%<extra></extra>",
            name="MoM %"
        ))
        fig2.add_hline(y=0, line_color=BORDA, line_width=1.5)
        fig2.update_layout(title="Crescimento Mensal (MoM %)")
        base_layout(fig2, h=420, leg=False)
        fig2.update_yaxes(ticksuffix="%")
        st.plotly_chart(fig2, use_container_width=True)

    # Heatmap
    st.markdown("**Mapa de Calor – Volume por Faixa e Mês**")
    try:
        pv = dff.pivot_table(index="tipo_desenrola", columns="data_base",
                              values="volume_operacoes", aggfunc="sum")
        pv.columns = [pd.Timestamp(c).strftime("%b/%y") for c in pv.columns]
        cs_heat = [[0,"#0D1117"],[0.5,P2],[1,P3]] if T=="escuro" else [[0,"#F5F4F0"],[0.5,P2],[1,P1]]
        fig3 = go.Figure(go.Heatmap(
            z=pv.values, x=pv.columns.tolist(), y=pv.index.tolist(),
            colorscale=cs_heat,
            hovertemplate="Faixa: %{y}<br>Mês: %{x}<br>R$ %{z:,.0f}<extra></extra>",
            colorbar=dict(title="Volume",tickprefix="R$ ",tickformat=".2s")
        ))
        base_layout(fig3, h=270, leg=False)
        fig3.update_layout(title="Mapa de Calor")
        st.plotly_chart(fig3, use_container_width=True)
    except Exception:
        st.info("Heatmap indisponível para os filtros selecionados.")

# ════════════════════════════════════════════════
# TAB 2 – CONCENTRAÇÃO BANCÁRIA
# ════════════════════════════════════════════════
with tabs[1]:
    st.markdown('<div class="sh">🏛️ Concentração Bancária</div>',unsafe_allow_html=True)

    top_n = st.slider("Top N instituições",5,30,15,key="tn")
    bagg2 = (dff.groupby(COL_B).agg(
        volume=("volume_operacoes","sum"),
        ops=("numero_operacoes","sum")).reset_index())
    bagg2["ticket"] = bagg2["volume"]/bagg2["ops"].replace(0,np.nan)
    bagg2["seg"]    = bagg2[COL_B].apply(class_banco)
    bagg2 = bagg2.nlargest(top_n,"volume")

    # Prepara customdata sem NaN (substitui NaN por "N/A")
    bagg2["ticket_str"] = bagg2["ticket"].apply(lambda x: f"R$ {x:,.0f}" if pd.notna(x) else "N/A")
    customdata_raw = bagg2[["ops","ticket_str","seg"]].values

    cb1, cb2 = st.columns(2)
    with cb1:
        fig4 = go.Figure(go.Bar(
            x=bagg2["volume"],
            y=bagg2[COL_B].str[:25],
            orientation="h",
            marker_color=P1,
            customdata=customdata_raw,
            hovertemplate="<b>%{y}</b><br>Volume: R$ %{x:,.0f}<br>Ops: %{customdata[0]:,.0f}<br>Ticket: %{customdata[1]}<br>Seg: %{customdata[2]}<extra></extra>"
        ))
        fig4.update_layout(title=f"Volume – Top {top_n} Instituições",
                           yaxis=dict(autorange="reversed"))
        base_layout(fig4, h=480, leg=False)
        fig4.update_xaxes(tickprefix="R$ ",tickformat=".2s")
        st.plotly_chart(fig4, use_container_width=True)

    with cb2:
        try:
            fig5 = px.treemap(bagg2, path=["seg",COL_B], values="volume",
                              color="ticket",
                              color_continuous_scale=[P1, P2, P3])
            fig5.update_traces(
                hovertemplate="<b>%{label}</b><br>Vol: R$ %{value:,.0f}<extra></extra>",
                textfont=dict(family="DM Sans",size=11))
            fig5.update_layout(title=f"Treemap – Top {top_n}",
                               template=TPLOTE, height=480,
                               paper_bgcolor="rgba(0,0,0,0)",
                               font=dict(color=TXT))
            st.plotly_chart(fig5, use_container_width=True)
        except Exception:
            st.info("Treemap indisponível para os filtros.")

    # Indicadores HHI, CR, Gini (usando hhi_v já calculado)
    cr3 = bagg2.nlargest(3,"volume")["volume"].sum()/bagg2["volume"].sum()*100 if len(bagg2)>=3 else 0
    cr5 = bagg2.nlargest(5,"volume")["volume"].sum()/bagg2["volume"].sum()*100 if len(bagg2)>=5 else 0
    gini_b = gini(bagg2["volume"])

    m1,m2,m3,m4 = st.columns(4)
    for col_,tt,vl,sb in [
        (m1,"📐 HHI",f"{hhi_v:.0f}",hhi_lbl),
        (m2,"🏆 CR3",f"{cr3:.1f}%","3 maiores bancos"),
        (m3,"🏆 CR5",f"{cr5:.1f}%","5 maiores bancos"),
        (m4,"⚖️ Gini",f"{gini_b:.3f}","desigualdade bancária"),
    ]:
        with col_:
            st.markdown(f"""<div class="kpi">
            <div class="kpi-tt">{tt}</div>
            <div class="kpi-vl">{vl}</div>
            <div class="kpi-sb">{sb}</div>
            </div>""",unsafe_allow_html=True)

# ════════════════════════════════════════════════
# TAB 3 – ANÁLISE REGIONAL
# ════════════════════════════════════════════════
with tabs[2]:
    st.markdown('<div class="sh">🗺️ Análise Regional</div>',unsafe_allow_html=True)

    rdf = dff.groupby("regiao").agg(
        volume=("volume_operacoes","sum"),
        ops=("numero_operacoes","sum")).reset_index()
    rdf["ticket"] = rdf["volume"]/rdf["ops"].replace(0,np.nan)
    rdf["pct"]    = rdf["volume"]/rdf["volume"].sum()*100

    cr1,cr2 = st.columns(2)
    with cr1:
        fig6 = make_subplots(specs=[[{"secondary_y":True}]])
        fig6.add_trace(go.Bar(
            x=rdf["regiao"], y=rdf["volume"], name="Volume",
            marker_color=[CORES_GRAFICOS[i%len(CORES_GRAFICOS)] for i in range(len(rdf))],
            hovertemplate="<b>%{x}</b><br>R$ %{y:,.0f}<extra></extra>"
        ), secondary_y=False)
        fig6.add_trace(go.Scatter(
            x=rdf["regiao"], y=rdf["ticket"], name="Ticket Médio",
            mode="lines+markers",
            line=dict(color=AMBER,width=2.5),
            marker=dict(size=9,symbol="diamond",color=AMBER),
            hovertemplate="<b>%{x}</b><br>Ticket: R$ %{y:,.0f}<extra></extra>"
        ), secondary_y=True)
        fig6.update_layout(title="Volume e Ticket Médio por Região")
        base_layout(fig6, h=400)
        fig6.update_yaxes(title_text="Volume",tickprefix="R$ ",tickformat=".2s",secondary_y=False)
        fig6.update_yaxes(title_text="Ticket",tickprefix="R$ ",secondary_y=True,showgrid=False,color=AMBER)
        st.plotly_chart(fig6, use_container_width=True)

    with cr2:
        fig7 = go.Figure(go.Pie(
            labels=rdf["regiao"], values=rdf["volume"],
            hole=0.55,
            marker=dict(colors=CORES_GRAFICOS[:len(rdf)],line=dict(color=BG,width=2)),
            hovertemplate="<b>%{label}</b><br>R$ %{value:,.0f}<br>%{percent}<extra></extra>",
            textinfo="label+percent", textfont=dict(size=11)
        ))
        fig7.add_annotation(text=f"<b>{fmt_brl(vol_tot)}</b>",
            x=0.5,y=0.5,showarrow=False,font=dict(size=13,color=TXT))
        fig7.update_layout(title="Participação Regional")
        base_layout(fig7, h=400)
        st.plotly_chart(fig7, use_container_width=True)

    # Área empilhada por região ao longo do tempo
    ereg = (dff.groupby(["data_base","regiao"])["volume_operacoes"]
            .sum().reset_index())
    fig8 = go.Figure()
    for i,reg in enumerate(sorted(ereg["regiao"].unique())):
        g = ereg[ereg["regiao"]==reg]
        cor = CORES_GRAFICOS[i%len(CORES_GRAFICOS)]
        fig8.add_trace(go.Scatter(
            x=g["data_base"], y=g["volume_operacoes"],
            name=reg, stackgroup="one",
            line=dict(color=cor,width=0.8),
            fillcolor=rgba(cor,0.7),
            hovertemplate=f"<b>{reg}</b><br>%{{x|%b/%Y}}<br>R$ %{{y:,.0f}}<extra></extra>"
        ))
    fig8.update_layout(title="Evolução por Região (Área Empilhada)")
    base_layout(fig8, h=370)
    fig8.update_yaxes(tickprefix="R$ ",tickformat=".2s")
    st.plotly_chart(fig8, use_container_width=True)

# ════════════════════════════════════════════════
# TAB 4 – ANÁLISE AVANÇADA (K-Means)
# ════════════════════════════════════════════════
with tabs[3]:
    st.markdown('<div class="sh">🔬 Análise Avançada – Clusterização Bancária</div>',unsafe_allow_html=True)

    try:
        cl_df = (dff.groupby(COL_B).agg(
            ops=("numero_operacoes","sum"),
            vol=("volume_operacoes","sum")).reset_index())
        cl_df["ticket"] = cl_df["vol"]/cl_df["ops"].replace(0,np.nan)
        cl_df["seg"]    = cl_df[COL_B].apply(class_banco)
        cl_df = cl_df[cl_df["ops"]>100].dropna()

        if len(cl_df)<3:
            st.info("Dados insuficientes para clusterização (mín. 3 instituições com >100 ops).")
        else:
            sc   = StandardScaler()
            feat = sc.fit_transform(cl_df[["ops","ticket"]])
            nc   = min(3,len(cl_df))
            km   = KMeans(n_clusters=nc,random_state=42,n_init=10)
            cl_df["cluster"] = km.fit_predict(feat)
            med  = cl_df.groupby("cluster")[["ops","ticket"]].mean()
            rv   = med["ops"].rank(ascending=False).astype(int)
            rt   = med["ticket"].rank(ascending=False).astype(int)
            def rot(c):
                av = rv[c]==1; at = rt[c]==1
                if av and not at: return "Alto Volume / Baixo Ticket"
                if not av and at: return "Baixo Volume / Alto Ticket"
                return "Perfil Equilibrado"
            cl_df["cluster_nome"] = cl_df["cluster"].map(rot)
            cnomes = {
                "Alto Volume / Baixo Ticket": P1,
                "Baixo Volume / Alto Ticket": AMBER,
                "Perfil Equilibrado": AZUL
            }
            ca1,ca2 = st.columns([3,2])
            with ca1:
                fig9 = go.Figure()
                for nm, grp in cl_df.groupby("cluster_nome"):
                    sz = np.log1p(grp["vol"]/grp["vol"].max()+0.01)*25+9
                    fig9.add_trace(go.Scatter(
                        x=grp["ops"], y=grp["ticket"],
                        mode="markers", name=nm,
                        marker=dict(size=sz,color=cnomes.get(nm,TXT2),
                                    opacity=0.82,line=dict(width=1.2,color=BORDA)),
                        customdata=grp[COL_B],
                        hovertemplate="<b>%{customdata}</b><br>Ops: %{x:,.0f}<br>Ticket: R$ %{y:,.2f}<extra></extra>"
                    ))
                fig9.update_layout(title="K-Means – Nº Operações × Ticket Médio",
                                   xaxis_title="Número de Operações",
                                   yaxis_title="Ticket Médio (R$)")
                base_layout(fig9, h=480)
                st.plotly_chart(fig9, use_container_width=True)
            with ca2:
                st.markdown("**Sumário por Cluster**")
                for nm, grp in cl_df.groupby("cluster_nome"):
                    st.markdown(f"""<div class="ibox">
                    <div class="il">{nm}</div>
                    <div class="it">
                    <b>{len(grp)}</b> inst. · Vol: <b>{fmt_brl(grp['vol'].sum())}</b><br>
                    Ops médias: <b>{fmt_num(grp['ops'].mean())}</b><br>
                    Ticket médio: <b>{fmt_brl(grp['ticket'].mean())}</b>
                    </div></div>""",unsafe_allow_html=True)
    except Exception as e:
        st.warning(f"Clusterização indisponível: {e}")

    # Scatter Ticket × Market Share
    st.markdown("**Dispersão: Ticket Médio × Market Share por Instituição**")
    sc2 = (dff.groupby(COL_B).agg(vol=("volume_operacoes","sum"),
                                   ops=("numero_operacoes","sum")).reset_index())
    sc2["ticket"] = sc2["vol"]/sc2["ops"].replace(0,np.nan)
    sc2["ms"]     = sc2["ops"]/sc2["ops"].sum()*100
    sc2["seg"]    = sc2[COL_B].apply(class_banco)
    sc2 = sc2.dropna().query("ops>50")
    seg_cores = {s:CORES_GRAFICOS[i%len(CORES_GRAFICOS)] for i,s in enumerate(sc2["seg"].unique())}
    fig10 = go.Figure()
    for seg,grp in sc2.groupby("seg"):
        sz = np.log1p(grp["vol"]/grp["vol"].max()+0.01)*20+8
        fig10.add_trace(go.Scatter(
            x=grp["ms"], y=grp["ticket"],
            mode="markers", name=seg,
            marker=dict(size=sz,color=seg_cores.get(seg,TXT2),
                        opacity=0.82,line=dict(width=1,color=BORDA)),
            customdata=grp[COL_B],
            hovertemplate="<b>%{customdata}</b><br>MS: %{x:.2f}%<br>Ticket: R$ %{y:,.0f}<extra></extra>"
        ))
    fig10.update_layout(title="Ticket Médio vs. Market Share (tamanho = volume)")
    base_layout(fig10, h=400)
    fig10.update_xaxes(title_text="Market Share (%)")
    fig10.update_yaxes(title_text="Ticket Médio (R$)",tickprefix="R$ ")
    st.plotly_chart(fig10, use_container_width=True)

# ════════════════════════════════════════════════
# TAB 5 – PARETO
# ════════════════════════════════════════════════
with tabs[4]:
    st.markdown('<div class="sh">📊 Curva de Pareto</div>',unsafe_allow_html=True)

    par = (dff.groupby(COL_B)["volume_operacoes"].sum()
             .reset_index().sort_values("volume_operacoes",ascending=False)
             .reset_index(drop=True))
    par["acum"] = par["volume_operacoes"].cumsum()/par["volume_operacoes"].sum()*100
    p80 = int((par["acum"]<=80).sum())
    gini_b2 = gini(par["volume_operacoes"])

    fig11 = make_subplots(specs=[[{"secondary_y":True}]])
    fig11.add_trace(go.Bar(
        x=par[COL_B].str[:20], y=par["volume_operacoes"],
        name="Volume", marker_color=P1,
        hovertemplate="<b>%{x}</b><br>R$ %{y:,.0f}<extra></extra>"
    ), secondary_y=False)
    fig11.add_trace(go.Scatter(
        x=par[COL_B].str[:20], y=par["acum"],
        name="% Acumulado",
        line=dict(color=AMBER,width=2.5),
        marker=dict(size=5,color=AMBER),
        hovertemplate="%{x}<br>Acum: %{y:.1f}%<extra></extra>"
    ), secondary_y=True)
    fig11.add_hline(y=80,line_dash="dot",line_color=VERM,secondary_y=True,
                    annotation_text="80%",annotation_font_color=VERM)
    fig11.update_layout(title=f"Pareto: {p80} bancos = 80% do volume total")
    base_layout(fig11, h=460)
    fig11.update_yaxes(title_text="Volume (R$)",tickprefix="R$ ",tickformat=".2s",secondary_y=False)
    fig11.update_yaxes(title_text="% Acumulado",ticksuffix="%",secondary_y=True,showgrid=False)
    fig11.update_xaxes(tickangle=-45)
    st.plotly_chart(fig11, use_container_width=True)

    conc = "altamente" if gini_b2>0.6 else "moderadamente"
    st.markdown(f"""<div class="ibox">
    <div class="il">📐 Interpretação da Curva de Pareto</div>
    <div class="it">
    <b>{p80} instituições</b> concentram <b>80%</b> do volume total renegociado no período.
    O Gini bancário de <b>{gini_b2:.3f}</b> confirma estrutura <b>{conc}</b> concentrada.
    Incentivar a participação de instituições menores pode democratizar o acesso ao programa.
    </div></div>""",unsafe_allow_html=True)

# ── Rodapé ────────────────────────────────────────────────────
st.markdown("---")
st.markdown(f"""
<div style="text-align:center;font-size:.72rem;color:{TXT2};padding:.4rem 0 1rem;">
🏦 Desenrola Brasil · Painel Executivo &nbsp;|&nbsp;
Fonte: <a href="https://www.bcb.gov.br/estatisticas/scr" target="_blank" style="color:{P1};">Banco Central do Brasil – SCR</a>
&nbsp;|&nbsp; Streamlit + Plotly
</div>""",unsafe_allow_html=True)