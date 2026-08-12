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
import logging
from datetime import datetime
import base64
from io import BytesIO
import os

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================
# DESIGN SYSTEM – TOKENS GLOBAIS
# ============================================================
st.set_page_config(
    page_title="Desenrola Brasil | Intelligence Platform",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inicializa tema e idioma
if "tema" not in st.session_state:
    st.session_state.tema = "escuro"
if "lang" not in st.session_state:
    st.session_state.lang = "pt"  # pt ou en

T = st.session_state.tema
LANG = st.session_state.lang

# ============================================================
# PALETA PREMIUM – 2026
# ============================================================
if T == "claro":
    BG          = "#F7F9FC"
    CARD        = "#FFFFFF"
    CARD_GLASS  = "rgba(255,255,255,0.85)"
    TXT         = "#1A2B4C"
    TXT2        = "#5A6E8A"
    BORDA       = "#E2E8F0"
    BORDA_GLOW  = "rgba(0,168,107,0.2)"
    P1          = "#00A86B"
    P2          = "#0066CC"
    P3          = "#52B788"
    ACCENT      = "#0066CC"
    ACCENT_GLOW = "rgba(0,102,204,0.2)"
    VERDE       = "#00A86B"
    VERM        = "#DC2626"
    AMBER       = "#F59E0B"
    AZUL        = "#3B82F6"
    ROXO        = "#8B5CF6"
    CINZA       = "#6B7280"
    TPLOTE      = "plotly_white"
    GRID        = "rgba(0,0,0,0.05)"
    GLOW_P1     = "rgba(0,168,107,0.3)"
else:
    BG          = "#0A0F1C"
    CARD        = "#111827"
    CARD_GLASS  = "rgba(17,24,39,0.85)"
    TXT         = "#F1F5F9"
    TXT2        = "#94A3B8"
    BORDA       = "#1F2937"
    BORDA_GLOW  = "rgba(56,189,248,0.2)"
    P1          = "#3FB68C"
    P2          = "#3B82F6"
    P3          = "#10B981"
    ACCENT      = "#60A5FA"
    ACCENT_GLOW = "rgba(96,165,250,0.2)"
    VERDE       = "#34D399"
    VERM        = "#F87171"
    AMBER       = "#FBBF24"
    AZUL        = "#60A5FA"
    ROXO        = "#A78BFA"
    CINZA       = "#6B7280"
    TPLOTE      = "plotly_dark"
    GRID        = "rgba(255,255,255,0.05)"
    GLOW_P1     = "rgba(63,182,140,0.3)"

CORES_GRAFICOS = [P1, AMBER, VERM, AZUL, ROXO, P2, VERDE, CINZA]

# ============================================================
# CSS PREMIUM (mesmo do original)
# ============================================================
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Playfair+Display:wght@400;500;600;700&display=swap');

* {{ margin: 0; padding: 0; box-sizing: border-box; }}

.stApp {{
    background: {BG};
    font-family: 'Inter', sans-serif;
}}

.block-container {{
    padding: 1rem 1.5rem !important;
    max-width: 1600px;
    margin: 0 auto;
}}

.hero {{
    background: linear-gradient(135deg, rgba(0,168,107,0.08) 0%, rgba(0,102,204,0.05) 100%);
    backdrop-filter: blur(10px);
    border: 1px solid {BORDA_GLOW};
    border-radius: 24px;
    padding: 1.8rem 2rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}}
.hero::before {{
    content: '';
    position: absolute;
    top: -50%;
    right: -20%;
    width: 300px;
    height: 300px;
    background: radial-gradient(circle, {ACCENT_GLOW} 0%, transparent 70%);
    pointer-events: none;
}}
.hero h1 {{
    font-size: 2.5rem;
    font-weight: 800;
    background: linear-gradient(135deg, {TXT}, {ACCENT});
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
    margin-bottom: 0.25rem;
    font-family: 'Playfair Display', serif;
}}
.hero p {{
    font-size: 0.9rem;
    color: {TXT2};
    margin-bottom: 1rem;
}}
.hero-badge {{
    display: inline-block;
    background: rgba(0,168,107,0.15);
    border: 1px solid {BORDA_GLOW};
    padding: 0.25rem 0.8rem;
    border-radius: 40px;
    font-size: 0.7rem;
    color: {P1};
    margin-right: 0.5rem;
    margin-bottom: 0.5rem;
}}

.kpi-grid {{ display: grid; grid-template-columns: repeat(5, 1fr); gap: 1rem; margin-bottom: 1.5rem; }}
@media (max-width: 1000px) {{ .kpi-grid {{ grid-template-columns: repeat(2, 1fr); }} }}
@media (max-width: 600px) {{ .kpi-grid {{ grid-template-columns: 1fr; }} }}

.kpi-card {{
    background: {CARD_GLASS};
    backdrop-filter: blur(12px);
    border: 1px solid {BORDA};
    border-radius: 20px;
    padding: 1rem 1.2rem;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
    overflow: hidden;
}}
.kpi-card::after {{
    content: '';
    position: absolute;
    bottom: 0;
    left: 0;
    right: 0;
    height: 3px;
    background: linear-gradient(90deg, {P1}, {ACCENT});
    transform: scaleX(0);
    transition: transform 0.3s;
}}
.kpi-card:hover::after {{ transform: scaleX(1); }}
.kpi-card:hover {{
    transform: translateY(-4px);
    border-color: {P1};
    box-shadow: 0 8px 25px rgba(0,0,0,0.1);
}}
.kpi-icon {{ font-size: 1.5rem; margin-bottom: 0.3rem; }}
.kpi-title {{ font-size: 0.6rem; text-transform: uppercase; letter-spacing: 0.08em; color: {TXT2}; font-weight: 600; }}
.kpi-value {{ font-size: 1.6rem; font-weight: 800; color: {TXT}; font-family: monospace; margin-top: 0.3rem; line-height: 1.2; }}
.kpi-trend {{ font-size: 0.65rem; margin-top: 0.3rem; display: inline-block; padding: 0.15rem 0.4rem; border-radius: 20px; background: rgba(0,168,107,0.1); color: {VERDE}; }}

.insight-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin: 1rem 0; }}
@media (max-width: 800px) {{ .insight-grid {{ grid-template-columns: 1fr; }} }}
.insight-card {{
    background: {CARD_GLASS};
    backdrop-filter: blur(8px);
    border: 1px solid {BORDA};
    border-radius: 16px;
    padding: 1rem 1.2rem;
    border-left: 3px solid {P1};
    transition: all 0.3s;
}}
.insight-card:hover {{ transform: translateX(4px); border-color: {ACCENT}; }}
.insight-title {{ font-size: 0.65rem; font-weight: 700; text-transform: uppercase; color: {TXT2}; margin-bottom: 0.5rem; }}
.insight-text {{ font-size: 0.85rem; color: {TXT}; line-height: 1.4; }}
.insight-value {{ font-size: 1.1rem; font-weight: 700; color: {P1}; margin-top: 0.5rem; }}

.al {{ padding: 0.6rem 1rem; border-radius: 12px; margin-bottom: 0.5rem; font-size: 0.8rem; display: flex; align-items: center; gap: 0.5rem; }}
.al.er {{ background: rgba(220,38,38,0.1); border-left: 3px solid {VERM}; color: {VERM}; }}
.al.wa {{ background: rgba(245,158,11,0.1); border-left: 3px solid {AMBER}; color: {AMBER}; }}
.al.ok {{ background: rgba(0,168,107,0.1); border-left: 3px solid {VERDE}; color: {VERDE}; }}
.al.in {{ background: rgba(59,130,246,0.1); border-left: 3px solid {AZUL}; color: {AZUL}; }}

.stTabs [data-baseweb="tab-list"] {{ gap: 0.5rem; background: transparent; }}
.stTabs [data-baseweb="tab"] {{
    background: {CARD_GLASS};
    backdrop-filter: blur(8px);
    border-radius: 40px;
    padding: 0.5rem 1.2rem;
    font-size: 0.8rem;
    font-weight: 500;
    color: {TXT2};
    border: 1px solid {BORDA};
    transition: all 0.2s;
}}
.stTabs [aria-selected="true"] {{ background: {P1}; color: white; border-color: {P1}; }}

section[data-testid="stSidebar"] {{ background: {CARD_GLASS}; backdrop-filter: blur(12px); border-right: 1px solid {BORDA}; }}

.footer {{ text-align: center; padding: 1.5rem 0 0.5rem; margin-top: 2rem; border-top: 1px solid {BORDA}; font-size: 0.65rem; color: {TXT2}; }}
</style>
""", unsafe_allow_html=True)

# ============================================================
# FUNÇÕES UTILITÁRIAS
# ============================================================
def rgba(hex_color, a=0.15):
    h = hex_color.lstrip('#')
    r,g,b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
    return f"rgba({r},{g},{b},{a})"

def fmt_brl(v):
    if pd.isna(v) or v==0: return "R$ 0"
    if v>=1e9: return f"R$ {v/1e9:.2f}B".replace(".",",")
    if v>=1e6: return f"R$ {v/1e6:.1f}M".replace(".",",")
    return f"R$ {v:,.0f}".replace(",",".")

def fmt_num(v):
    if pd.isna(v): return "0"
    return f"{int(v):,}".replace(",",".")

def class_banco(nome):
    n = re.sub(r'\s*-\s*PRUDENCIAL$','',str(nome).upper().strip())
    if any(x in n for x in ["NUBANK","INTER","C6","NEON","ORIGINAL","PAN","NEXT"]): return "Digital"
    if any(x in n for x in ["ITAU","BRADESCO","SANTANDER","CAIXA","BANCO DO BRASIL","BB"]): return "Tradicional"
    if any(x in n for x in ["BTG","XP","MODAL","GENIAL"]): return "Investimento"
    if any(x in n for x in ["SICOOB","SICREDI"]): return "Cooperativa"
    return "Outros"

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
        margin=dict(l=50,r=40,t=55,b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TXT, family="Inter", size=12),
        hovermode="x unified",
        showlegend=leg,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        transition=dict(duration=300)
    )
    fig.update_xaxes(showgrid=False, color=TXT, linecolor=BORDA)
    fig.update_yaxes(showgrid=True, gridcolor=GRID, color=TXT)
    return fig

# ============================================================
# GERADOR DE DADOS SINTÉTICOS (CORRIGIDO, SEM ERROS DE PROBABILIDADE)
# ============================================================
@st.cache_data(ttl=3600)
def generate_sample_data(n_records=5000):
    """Gera dados sintéticos realistas para demonstração."""
    np.random.seed(42)
    dates = pd.date_range("2023-07-01", "2024-12-01", freq="MS")
    ufs = ["SP","RJ","MG","RS","PR","BA","PE","CE","PA","MA","SC","DF","GO","MT","MS","AM","ES","PB","RN","AL"]
    bancos = [
        "ITAU UNIBANCO - PRUDENCIAL", "BANCO BRADESCO - PRUDENCIAL", 
        "NUBANK", "CAIXA ECONOMICA FEDERAL", "BANCO DO BRASIL", 
        "SANTANDER", "BANCO INTER", "C6 BANK", "SICOOB", "BTG PACTUAL",
        "BANCO PAN", "BANCO ORIGINAL", "XP INVESTIMENTOS", "BANCO SAFRA",
        "SICREDI", "MERCADO PAGO", "PICPAY", "PAGBANK", "BANCO NEXT", "BANCO NEON"
    ]
    tipos = ["Faixa 1", "Faixa 2", "Faixa 3", "Faixa 4"]
    areas = ["Crédito Pessoal", "Cartão de Crédito", "Cheque Especial", "Financiamento"]
    
    # pesos para distribuição realista
    uf_weights = np.array([0.25,0.15,0.10,0.08,0.07,0.06,0.05,0.04,0.03,0.03,
                           0.03,0.02,0.02,0.02,0.01,0.01,0.01,0.01,0.01,0.01])
    banco_weights = np.array([0.18,0.15,0.12,0.10,0.08,0.07,0.06,0.05,0.04,0.03,
                              0.02,0.02,0.02,0.02,0.01,0.01,0.01,0.005,0.003,0.002])
    uf_weights = uf_weights / uf_weights.sum()
    banco_weights = banco_weights / banco_weights.sum()
    
    data = []
    for _ in range(n_records):
        # Usa randint simples (sem probabilidades) para escolher data
        date_idx = np.random.randint(0, len(dates))
        date = dates[date_idx]
        trend_factor = 1 + (date_idx / len(dates)) * 0.5
        
        base_volume = np.random.lognormal(mean=14, sigma=1.2)
        ops = max(1, int(base_volume / np.random.lognormal(mean=3, sigma=0.5)))
        volume = base_volume * np.random.uniform(0.8, 1.2)
        
        data.append({
            "data_base": date.strftime("%Y%m"),
            "unidade_federacao": np.random.choice(ufs, p=uf_weights),
            "nome_conglomerado_financeiro": np.random.choice(bancos, p=banco_weights),
            "tipo_desenrola": np.random.choice(tipos),
            "grande_area": np.random.choice(areas),
            "numero_operacoes": int(ops * trend_factor),
            "volume_operacoes": float(volume * trend_factor)
        })
    df = pd.DataFrame(data)
    logger.info(f"Generated {len(df)} synthetic records")
    return df

# ============================================================
# CARREGAMENTO DE DADOS (COM FALLBACK)
# ============================================================
@st.cache_data(ttl=3600)
def load_data():
    # Tenta carregar o arquivo CSV
    try:
        for enc in ["utf-8","latin1","cp1252"]:
            try:
                df = pd.read_csv("dados_desenrola.csv", sep=";", encoding=enc, low_memory=False)
                df.columns = df.columns.str.lower().str.strip()
                for c in ["numero_operacoes","volume_operacoes"]:
                    if c in df.columns:
                        df[c] = pd.to_numeric(df[c].astype(str).str.replace(".","",regex=False).str.replace(",",".",regex=False), errors="coerce")
                df["data_base"] = pd.to_datetime(df["data_base"].astype(str), format="%Y%m", errors="coerce")
                df["tipo_banco"] = df["nome_conglomerado_financeiro"].apply(class_banco)
                df["regiao"] = df["unidade_federacao"].apply(class_regiao)
                df = df.dropna(subset=["volume_operacoes","numero_operacoes"])
                if len(df) > 0:
                    return df, False  # False = dados reais
            except:
                continue
        # Se falhou, gera dados sintéticos
        df = generate_sample_data()
        return df, True  # True = dados de demonstração
    except Exception as e:
        logger.error(f"Erro ao carregar dados: {e}")
        return None, False

with st.spinner("🔄 Carregando dados..."):
    df, is_demo = load_data()

if df is None:
    st.error("❌ Erro ao carregar dados. Verifique o arquivo 'dados_desenrola.csv'.")
    st.stop()

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown(f"""
    <div style="text-align:center; padding:0.5rem 0 1rem; border-bottom:1px solid {BORDA}; margin-bottom:1rem;">
        <div style="font-family:'Playfair Display',serif; font-size:1.2rem; font-weight:700;">🏦 Desenrola</div>
        <div style="font-size:0.65rem; color:{TXT2};">Intelligence Platform</div>
    </div>
    """, unsafe_allow_html=True)
    
    col_a1, col_a2 = st.columns(2)
    with col_a1:
        if st.button("☀️ Light", use_container_width=True):
            st.session_state.tema = "claro"
            st.rerun()
    with col_a2:
        if st.button("🌙 Dark", use_container_width=True):
            st.session_state.tema = "escuro"
            st.rerun()
    
    # Opção de idioma (simples)
    lang_options = {"pt": "🇧🇷 PT", "en": "🇺🇸 EN"}
    new_lang = st.selectbox("Idioma", options=["pt","en"], index=0 if LANG=="pt" else 1,
                            format_func=lambda x: lang_options[x])
    if new_lang != LANG:
        st.session_state.lang = new_lang
        st.rerun()
    
    st.markdown("---")
    st.markdown("**🔍 Filtros**")
    
    tipos = sorted(df["tipo_desenrola"].unique())
    sel_tip = st.multiselect("Faixa", tipos, default=tipos)
    
    regioes = sorted(df["regiao"].unique())
    sel_reg = st.multiselect("Região", regioes, default=regioes)
    
    segmentos = sorted(df["tipo_banco"].unique())
    sel_seg = st.multiselect("Segmento", segmentos, default=segmentos)
    
    datas = sorted(df["data_base"].unique())
    if len(datas) > 1:
        i0, i1 = st.select_slider("Período", options=list(range(len(datas))),
            value=(0, len(datas)-1),
            format_func=lambda i: pd.Timestamp(datas[i]).strftime("%m/%Y"))
        d_ini, d_fim = datas[i0], datas[i1]
    else:
        d_ini = d_fim = datas[0]
    
    if st.button("🔄 Resetar", use_container_width=True):
        st.rerun()
    
    st.markdown("---")
    st.markdown(f"""
    <div style="background:{CARD_GLASS}; border-radius:12px; padding:0.8rem; border:1px solid {BORDA};">
        <div style="font-size:0.7rem; font-weight:600;">📋 Qualidade</div>
        <div style="font-size:0.65rem; color:{TXT2};">Registros: <b>{len(df):,}</b></div>
        <div style="font-size:0.65rem; color:{TXT2};">Período: {d_ini.strftime('%m/%Y')} → {d_fim.strftime('%m/%Y')}</div>
        <div style="font-size:0.65rem; color:{TXT2}; margin-top:0.3rem;">{'🧪 Modo Demonstração' if is_demo else '📁 Dados Reais'}</div>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# FILTRAGEM
# ============================================================
dff = df[
    (df["tipo_desenrola"].isin(sel_tip)) &
    (df["regiao"].isin(sel_reg)) &
    (df["tipo_banco"].isin(sel_seg)) &
    (df["data_base"] >= d_ini) &
    (df["data_base"] <= d_fim)
]

if dff.empty:
    st.warning("⚠️ Nenhum dado com os filtros selecionados.")
    st.stop()

COL_B = "nome_conglomerado_financeiro"

# ============================================================
# HERO SECTION
# ============================================================
vol_tot = dff["volume_operacoes"].sum()
ops_tot = dff["numero_operacoes"].sum()
ticket = vol_tot/ops_tot if ops_tot>0 else 0

st.markdown(f"""
<div class="hero">
    <h1>🏦 Desenrola Brasil</h1>
    <p>Inteligência analítica para renegociação de dívidas · Fonte: Banco Central (SCR)</p>
    <div>
        <span class="hero-badge">📊 {fmt_num(ops_tot)} contratos</span>
        <span class="hero-badge">💰 {fmt_brl(vol_tot)} renegociados</span>
        <span class="hero-badge">🏛️ {dff[COL_B].nunique()} instituições</span>
        <span class="hero-badge">🗺️ {dff['unidade_federacao'].nunique()} UFs</span>
        {f'<span class="hero-badge">🧪 Demo</span>' if is_demo else ''}
    </div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# KPIs PREMIUM (mesmo do original)
# ============================================================
evol_g = dff.groupby("data_base")["volume_operacoes"].sum().reset_index().sort_values("data_base")
evol_g["mom"] = evol_g["volume_operacoes"].pct_change()*100
mom_last = evol_g["mom"].dropna().iloc[-1] if len(evol_g["mom"].dropna())>0 else 0
mom_cor = VERDE if mom_last>=0 else VERM

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-icon">💰</div>
        <div class="kpi-title">Volume Renegociado</div>
        <div class="kpi-value">{fmt_brl(vol_tot)}</div>
        <div class="kpi-trend" style="color:{mom_cor};">{mom_last:+.1f}% vs mês anterior</div>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-icon">📄</div>
        <div class="kpi-title">Total de Contratos</div>
        <div class="kpi-value">{fmt_num(ops_tot)}</div>
    </div>
    """, unsafe_allow_html=True)
with col3:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-icon">🎫</div>
        <div class="kpi-title">Ticket Médio</div>
        <div class="kpi-value">{fmt_brl(ticket)}</div>
    </div>
    """, unsafe_allow_html=True)
with col4:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-icon">🏛️</div>
        <div class="kpi-title">Instituições</div>
        <div class="kpi-value">{fmt_num(dff[COL_B].nunique())}</div>
    </div>
    """, unsafe_allow_html=True)
with col5:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-icon">🗺️</div>
        <div class="kpi-title">Estados</div>
        <div class="kpi-value">{fmt_num(dff['unidade_federacao'].nunique())}</div>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# ALERTAS AUTOMÁTICOS
# ============================================================
alertas = []
if mom_last < -15: alertas.append(("er", f"🔴 Queda Abrupta – volume caiu {mom_last:.1f}%"))
elif mom_last < -5: alertas.append(("wa", f"🟡 Desaceleração – queda de {mom_last:.1f}%"))
elif mom_last > 20: alertas.append(("ok", f"🟢 Aceleração Forte – +{mom_last:.1f}%"))
elif mom_last > 0: alertas.append(("ok", f"🟢 Crescimento Estável – +{mom_last:.1f}%"))

b_agg = dff.groupby(COL_B)["numero_operacoes"].sum().reset_index()
hhi_v = hhi(b_agg, "numero_operacoes")
if hhi_v > 2500: alertas.append(("er", "🔴 Concentração Elevada – HHI > 2.500"))
elif hhi_v > 1500: alertas.append(("wa", "🟡 Concentração Moderada – HHI 1.500-2.500"))
else: alertas.append(("ok", "🟢 Mercado Competitivo – HHI < 1.500"))

reg_v = dff.groupby("regiao")["volume_operacoes"].sum()
gini_r = gini(reg_v)
if gini_r > 0.7: alertas.append(("er", f"🔴 Alta Desigualdade Regional – Gini = {gini_r:.2f}"))
elif gini_r > 0.5: alertas.append(("wa", f"🟡 Desigualdade Regional – Gini = {gini_r:.2f}"))

if alertas:
    st.markdown("### ⚡ Alertas Inteligentes")
    cols = st.columns(min(len(alertas), 3))
    for i, (cls, msg) in enumerate(alertas):
        with cols[i % 3]:
            st.markdown(f'<div class="al {cls}">{msg}</div>', unsafe_allow_html=True)

# ============================================================
# INSIGHTS AUTOMÁTICOS
# ============================================================
st.markdown("### 🔍 Insights Automáticos")

reg_df = dff.groupby("regiao")["volume_operacoes"].sum().reset_index()
reg_df["pct"] = reg_df["volume_operacoes"]/reg_df["volume_operacoes"].sum()*100
lider_reg = reg_df.loc[reg_df["volume_operacoes"].idxmax()]

area_df = dff.groupby("grande_area")["volume_operacoes"].sum().reset_index() if "grande_area" in dff.columns else None
if area_df is not None:
    lider_area = area_df.loc[area_df["volume_operacoes"].idxmax()]
else:
    lider_area = None

cresc_med = evol_g["mom"].dropna().mean() if len(evol_g) > 1 else 0

insight_cols = st.columns(3)
with insight_cols[0]:
    st.markdown(f"""
    <div class="insight-card">
        <div class="insight-title">📍 Concentração Regional</div>
        <div class="insight-text">A região <b>{lider_reg['regiao']}</b> concentra <b>{lider_reg['pct']:.1f}%</b> do volume total.</div>
        <div class="insight-value">{fmt_brl(lider_reg['volume_operacoes'])}</div>
    </div>
    """, unsafe_allow_html=True)
with insight_cols[1]:
    if lider_area is not None:
        st.markdown(f"""
        <div class="insight-card">
            <div class="insight-title">🧬 Área Líder</div>
            <div class="insight-text"><b>{lider_area['grande_area']}</b> lidera os investimentos.</div>
            <div class="insight-value">{fmt_brl(lider_area['volume_operacoes'])}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="insight-card">
            <div class="insight-title">📈 Tendência</div>
            <div class="insight-text">Crescimento médio mensal de <b>{cresc_med:+.1f}%</b>.</div>
        </div>
        """, unsafe_allow_html=True)
with insight_cols[2]:
    st.markdown(f"""
    <div class="insight-card">
        <div class="insight-title">⚖️ Concentração (HHI)</div>
        <div class="insight-text">Índice Herfindahl-Hirschman: <b>{hhi_v:.0f}</b></div>
        <div class="insight-value">{'Baixa' if hhi_v<1500 else 'Moderada' if hhi_v<2500 else 'Alta'} concentração</div>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# TABS COM GRÁFICOS (mesmo do original, sem alterações)
# ============================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Evolução Temporal", "🏦 Concentração Bancária", "🗺️ Análise Regional", 
    "🔬 Análise Avançada", "📊 Distribuição"
])

# ---------- TAB 1: EVOLUÇÃO TEMPORAL ----------
with tab1:
    st.markdown("### 📈 Evolução do Programa")
    
    col_ev1, col_ev2 = st.columns([2, 1])
    
    with col_ev1:
        fig1 = go.Figure()
        tipos_u = sorted(dff["tipo_desenrola"].unique())
        for i, tp in enumerate(tipos_u):
            g = dff[dff["tipo_desenrola"]==tp].groupby("data_base")["volume_operacoes"].sum().reset_index()
            cor = CORES_GRAFICOS[i % len(CORES_GRAFICOS)]
            fig1.add_trace(go.Scatter(
                x=g["data_base"], y=g["volume_operacoes"],
                mode="lines+markers", name=f"Faixa {tp}",
                line=dict(color=cor, width=2.5),
                marker=dict(size=6, color=cor),
                hovertemplate=f"<b>Faixa {tp}</b><br>%{{x|%b/%Y}}<br>R$ %{{y:,.0f}}<extra></extra>"
            ))
        
        try:
            serie = evol_g["volume_operacoes"]
            if len(serie) >= 4:
                hw = ExponentialSmoothing(serie.values, trend="add", seasonal=None, initialization_method="estimated").fit()
                prev = hw.forecast(3)
                dt_fut = pd.date_range(evol_g["data_base"].max(), periods=4, freq="MS")[1:]
                sigma = float(np.std(hw.resid))
                low = [float(v) for v in prev - 1.96*sigma]
                upp = [float(v) for v in prev + 1.96*sigma]
                xband = list(dt_fut) + list(dt_fut[::-1])
                yband = upp + low[::-1]
                fig1.add_trace(go.Scatter(
                    x=xband, y=yband, fill="toself",
                    fillcolor=rgba(AMBER, 0.15), line=dict(color="rgba(0,0,0,0)"),
                    name="IC 95%", hoverinfo="skip"
                ))
                fig1.add_trace(go.Scatter(
                    x=dt_fut, y=prev, mode="lines+markers", name="Projeção",
                    line=dict(color=AMBER, width=2, dash="dash"),
                    marker=dict(size=7, symbol="diamond", color=AMBER)
                ))
        except Exception:
            pass
        
        fig1.update_layout(title="Volume Renegociado por Faixa + Projeção")
        base_layout(fig1, h=420)
        fig1.update_yaxes(tickprefix="R$ ", tickformat=",.0f")
        st.plotly_chart(fig1, use_container_width=True)
    
    with col_ev2:
        fig2 = go.Figure(go.Bar(
            x=evol_g["data_base"], y=evol_g["mom"],
            marker_color=[VERDE if v>=0 else VERM for v in evol_g["mom"]],
            hovertemplate="%{x|%b/%Y}<br>MoM: %{y:.1f}%<extra></extra>"
        ))
        fig2.add_hline(y=0, line_color=BORDA, line_width=1.5)
        fig2.update_layout(title="Crescimento Mensal (MoM)")
        base_layout(fig2, h=420, leg=False)
        fig2.update_yaxes(ticksuffix="%")
        st.plotly_chart(fig2, use_container_width=True)
    
    # Heatmap
    st.markdown("**🔥 Mapa de Calor – Volume por Faixa**")
    try:
        heat_data = dff.pivot_table(index="tipo_desenrola", columns="data_base", values="volume_operacoes", aggfunc="sum")
        heat_data.columns = [pd.Timestamp(c).strftime("%b/%y") for c in heat_data.columns]
        fig3 = go.Figure(go.Heatmap(
            z=heat_data.values, x=heat_data.columns, y=heat_data.index,
            colorscale=[[0, BG], [0.5, P2], [1, P1]],
            hovertemplate="Faixa: %{y}<br>Mês: %{x}<br>R$ %{z:,.0f}<extra></extra>"
        ))
        base_layout(fig3, h=320, leg=False)
        fig3.update_layout(title="Intensidade de Renegociação")
        st.plotly_chart(fig3, use_container_width=True)
    except Exception:
        st.info("Heatmap indisponível para os filtros atuais.")

# ---------- TAB 2: CONCENTRAÇÃO BANCÁRIA ----------
with tab2:
    st.markdown("### 🏦 Concentração Bancária")
    
    top_n = st.slider("Top N instituições", 5, 30, 15, key="top_n_bancos")
    
    banco_agg = dff.groupby(COL_B).agg(volume=("volume_operacoes","sum"), ops=("numero_operacoes","sum")).reset_index()
    banco_agg["ticket"] = banco_agg["volume"] / banco_agg["ops"]
    banco_agg["seg"] = banco_agg[COL_B].apply(class_banco)
    banco_agg = banco_agg.nlargest(top_n, "volume")
    
    fig4 = go.Figure(go.Bar(
        x=banco_agg["volume"], y=banco_agg[COL_B].str[:30], orientation="h",
        marker=dict(color=P1, line=dict(width=0)),
        text=banco_agg["volume"].apply(lambda x: fmt_brl(x)), textposition="outside",
        hovertemplate="<b>%{y}</b><br>Volume: R$ %{x:,.0f}<br>Ops: %{customdata[0]:,.0f}<br>Ticket: R$ %{customdata[1]:,.0f}<extra></extra>",
        customdata=banco_agg[["ops", "ticket"]].values
    ))
    fig4.update_layout(title=f"Top {top_n} Instituições por Volume")
    base_layout(fig4, h=500, leg=False)
    fig4.update_xaxes(tickprefix="R$ ", tickformat=".2s")
    st.plotly_chart(fig4, use_container_width=True)
    
    fig5 = px.treemap(banco_agg, path=["seg", COL_B], values="volume", color="ticket",
                      color_continuous_scale=[P1, P2, P3], title="Treemap – Distribuição por Segmento")
    fig5.update_layout(template=TPLOTE, height=450)
    st.plotly_chart(fig5, use_container_width=True)
    
    cr3 = banco_agg.nlargest(3, "volume")["volume"].sum() / banco_agg["volume"].sum() * 100
    cr5 = banco_agg.nlargest(5, "volume")["volume"].sum() / banco_agg["volume"].sum() * 100
    gini_b = gini(banco_agg["volume"])
    
    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        st.markdown(f"<div class='kpi-card'><div class='kpi-title'>📐 CR3</div><div class='kpi-value'>{cr3:.1f}%</div><div class='kpi-sub'>3 maiores bancos</div></div>", unsafe_allow_html=True)
    with col_m2:
        st.markdown(f"<div class='kpi-card'><div class='kpi-title'>📐 CR5</div><div class='kpi-value'>{cr5:.1f}%</div><div class='kpi-sub'>5 maiores bancos</div></div>", unsafe_allow_html=True)
    with col_m3:
        st.markdown(f"<div class='kpi-card'><div class='kpi-title'>⚖️ Gini</div><div class='kpi-value'>{gini_b:.3f}</div><div class='kpi-sub'>desigualdade bancária</div></div>", unsafe_allow_html=True)

# ---------- TAB 3: ANÁLISE REGIONAL ----------
with tab3:
    st.markdown("### 🗺️ Distribuição Regional")
    
    try:
        uf_data = dff.groupby("unidade_federacao")["volume_operacoes"].sum().reset_index()
        uf_data.columns = ["uf", "volume"]
        fig_map = px.choropleth(
            uf_data, locations="uf", locationmode="BRA-states",
            color="volume", color_continuous_scale="Blues",
            title="🌡️ Intensidade de Renegociação por Estado",
            hover_name="uf", hover_data={"volume": ":,.0f"}
        )
        fig_map.update_layout(geo=dict(bgcolor="rgba(0,0,0,0)"), height=500, margin=dict(l=0,r=0,t=40,b=0))
        fig_map.update_layout(template=TPLOTE, paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_map, use_container_width=True)
    except Exception:
        st.info("Mapa indisponível para os filtros atuais.")
    
    reg_data = dff.groupby("regiao").agg(volume=("volume_operacoes","sum"), ops=("numero_operacoes","sum")).reset_index()
    reg_data["ticket"] = reg_data["volume"] / reg_data["ops"]
    reg_data["pct"] = reg_data["volume"] / reg_data["volume"].sum() * 100
    
    fig6 = make_subplots(specs=[[{"secondary_y": True}]])
    fig6.add_trace(go.Bar(x=reg_data["regiao"], y=reg_data["volume"], name="Volume", marker_color=CORES_GRAFICOS[:len(reg_data)]), secondary_y=False)
    fig6.add_trace(go.Scatter(x=reg_data["regiao"], y=reg_data["ticket"], name="Ticket Médio", mode="lines+markers",
                              line=dict(color=AMBER, width=2.5), marker=dict(size=8, symbol="diamond")), secondary_y=True)
    fig6.update_layout(title="Volume e Ticket Médio por Região")
    base_layout(fig6, h=450)
    fig6.update_yaxes(title_text="Volume (R$)", tickprefix="R$ ", tickformat=".2s", secondary_y=False)
    fig6.update_yaxes(title_text="Ticket Médio (R$)", tickprefix="R$ ", secondary_y=True, showgrid=False)
    st.plotly_chart(fig6, use_container_width=True)
    
    fig7 = go.Figure(go.Pie(labels=reg_data["regiao"], values=reg_data["volume"], hole=0.5,
                            marker=dict(colors=CORES_GRAFICOS[:len(reg_data)], line=dict(color=BG, width=2)),
                            textinfo="label+percent", hovertemplate="<b>%{label}</b><br>R$ %{value:,.0f}<br>%{percent}<extra></extra>"))
    fig7.update_layout(title="Participação Regional", height=400)
    base_layout(fig7, h=400)
    st.plotly_chart(fig7, use_container_width=True)

# ---------- TAB 4: ANÁLISE AVANÇADA ----------
with tab4:
    st.markdown("### 🔬 Análises Avançadas")
    
    try:
        cl_df = dff.groupby(COL_B).agg(ops=("numero_operacoes","sum"), vol=("volume_operacoes","sum")).reset_index()
        cl_df["ticket"] = cl_df["vol"] / cl_df["ops"]
        cl_df = cl_df[cl_df["ops"] > 100].dropna()
        
        if len(cl_df) >= 3:
            sc = StandardScaler()
            feat = sc.fit_transform(cl_df[["ops", "ticket"]])
            km = KMeans(n_clusters=min(3, len(cl_df)), random_state=42, n_init=10)
            cl_df["cluster"] = km.fit_predict(feat)
            
            fig8 = go.Figure()
            for c in cl_df["cluster"].unique():
                grp = cl_df[cl_df["cluster"] == c]
                sz = np.log1p(grp["vol"] / grp["vol"].max() + 0.01) * 25 + 9
                fig8.add_trace(go.Scatter(
                    x=grp["ops"], y=grp["ticket"], mode="markers", name=f"Cluster {c+1}",
                    marker=dict(size=sz, color=CORES_GRAFICOS[c % len(CORES_GRAFICOS)], opacity=0.8,
                                line=dict(width=1, color=BORDA)),
                    text=grp[COL_B], hovertemplate="<b>%{text}</b><br>Ops: %{x:,.0f}<br>Ticket: R$ %{y:,.2f}<extra></extra>"
                ))
            fig8.update_layout(title="Clusterização (Operações × Ticket)", xaxis_title="Operações", yaxis_title="Ticket Médio (R$)")
            base_layout(fig8, h=450)
            st.plotly_chart(fig8, use_container_width=True)
        else:
            st.info("Dados insuficientes para clusterização.")
    except Exception as e:
        st.warning(f"Clusterização indisponível: {e}")
    
    st.markdown("**📡 Radar de Concentração**")
    radar_data = pd.DataFrame({
        "Métrica": ["HHI", "CR3", "CR5", "Gini Regional", "Ticket Médio"],
        "Valor Normalizado": [min(hhi_v/3000, 1), cr3/100, cr5/100, min(gini_r, 1), min(ticket/15000, 1)]
    })
    fig9 = px.line_polar(radar_data, r="Valor Normalizado", theta="Métrica", line_close=True,
                         color_discrete_sequence=[P1], title="Índices de Concentração")
    fig9.update_layout(template=TPLOTE, height=450)
    st.plotly_chart(fig9, use_container_width=True)
    
    sc_df = dff.groupby(COL_B).agg(vol=("volume_operacoes","sum"), ops=("numero_operacoes","sum")).reset_index()
    sc_df["ticket"] = sc_df["vol"] / sc_df["ops"]
    sc_df["ms"] = sc_df["ops"] / sc_df["ops"].sum() * 100
    sc_df = sc_df.dropna().query("ops>50")
    
    fig10 = px.scatter(sc_df, x="ms", y="ticket", size="vol", color="ticket",
                       hover_name=COL_B, title="Ticket Médio vs Market Share",
                       labels={"ms": "Market Share (%)", "ticket": "Ticket Médio (R$)"},
                       color_continuous_scale="Viridis")
    fig10.update_layout(template=TPLOTE, height=450)
    st.plotly_chart(fig10, use_container_width=True)

# ---------- TAB 5: DISTRIBUIÇÃO E PARETO ----------
with tab5:
    st.markdown("### 📊 Curva de Pareto")
    
    pareto_data = dff.groupby(COL_B)["volume_operacoes"].sum().sort_values(ascending=False).reset_index()
    pareto_data["acum"] = pareto_data["volume_operacoes"].cumsum() / pareto_data["volume_operacoes"].sum() * 100
    p80 = (pareto_data["acum"] <= 80).sum()
    
    fig11 = make_subplots(specs=[[{"secondary_y": True}]])
    fig11.add_trace(go.Bar(x=pareto_data[COL_B].str[:20], y=pareto_data["volume_operacoes"], name="Volume", marker_color=P1), secondary_y=False)
    fig11.add_trace(go.Scatter(x=pareto_data[COL_B].str[:20], y=pareto_data["acum"], name="% Acumulado", mode="lines+markers",
                               line=dict(color=AMBER, width=2.5)), secondary_y=True)
    fig11.add_hline(y=80, line_dash="dot", line_color=VERM, secondary_y=True, annotation_text="80%")
    fig11.update_layout(title=f"Pareto: {p80} instituições = 80% do volume")
    base_layout(fig11, h=450)
    fig11.update_yaxes(title_text="Volume (R$)", tickprefix="R$ ", secondary_y=False)
    fig11.update_yaxes(title_text="% Acumulado", ticksuffix="%", secondary_y=True)
    st.plotly_chart(fig11, use_container_width=True)
    
    st.markdown(f"""
    <div class="insight-card">
        <div class="insight-title">📐 Interpretação</div>
        <div class="insight-text"><b>{p80} instituições</b> concentram <b>80%</b> do volume total renegociado.</div>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# EXPORTAÇÃO
# ============================================================
st.markdown("---")
st.markdown("### 📥 Exportar")

col_exp1, col_exp2 = st.columns(2)
with col_exp1:
    csv_data = dff.to_csv(index=False).encode("utf-8")
    st.download_button("📄 CSV (dados filtrados)", csv_data, file_name=f"desenrola_{datetime.now().strftime('%Y%m%d')}.csv", mime="text/csv")
with col_exp2:
    relatorio_txt = f"""RELATÓRIO DESENROLA BRASIL
Volume: {fmt_brl(vol_tot)}
Contratos: {fmt_num(ops_tot)}
Ticket Médio: {fmt_brl(ticket)}
HHI: {hhi_v:.0f}
Gini Regional: {gini_r:.3f}
Fonte: Banco Central do Brasil (SCR)
"""
    st.download_button("📝 Relatório TXT", relatorio_txt, file_name=f"relatorio_{datetime.now().strftime('%Y%m%d')}.txt", mime="text/plain")

# ============================================================
# FOOTER
# ============================================================
st.markdown(f"""
<div class="footer">
    🏦 Desenrola Brasil · Inteligência Financeira<br>
    Fonte: <a href="https://www.bcb.gov.br/estatisticas/scr" target="_blank" style="color:{P1};">Banco Central do Brasil (SCR)</a>
</div>
""", unsafe_allow_html=True)
