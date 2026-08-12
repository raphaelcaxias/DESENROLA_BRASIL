"""
Debt Settlement Brazil – Intelligence Platform
Refactored version with performance, maintainability, and robustness improvements.
"""

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
from typing import Optional, Tuple, List, Dict, Any
import base64

# ============================================================
# CONFIGURATION
# ============================================================
CONFIG = {
    "hhi_threshold_high": 2500,
    "hhi_threshold_moderate": 1500,
    "gini_threshold_high": 0.7,
    "gini_threshold_moderate": 0.5,
    "mom_drop_sharp": -15,
    "mom_drop_moderate": -5,
    "mom_growth_strong": 20,
    "default_top_n": 15,
    "min_cluster_samples": 3,
    "forecast_periods": 3,
    "min_ops_for_cluster": 100,
}

# ============================================================
# INTERNATIONALIZATION (i18n)
# ============================================================
TEXTS = {
    "en": {
        "app_title": "Debt Settlement Brazil",
        "app_subtitle": "Intelligence Platform",
        "page_title": "Debt Settlement Brazil | Intelligence Platform",
        "hero_title": "Debt Settlement Brazil",
        "hero_subtitle": "Analytical intelligence for debt renegotiation · Source: Central Bank of Brazil (SCR)",
        "volume": "Renegotiated Volume",
        "contracts": "Total Contracts",
        "avg_ticket": "Average Ticket",
        "institutions": "Institutions",
        "states": "States",
        "vs_prev_month": "vs previous month",
        "smart_alerts": "Smart Alerts",
        "automated_insights": "Automated Insights",
        "regional_concentration": "Regional Concentration",
        "leading_area": "Leading Area",
        "trend": "Trend",
        "concentration_hhi": "Concentration (HHI)",
        "low_concentration": "Low",
        "moderate_concentration": "Moderate",
        "high_concentration": "High",
        "time_series": "Time Series",
        "bank_concentration": "Bank Concentration",
        "regional_analysis": "Regional Analysis",
        "advanced_analytics": "Advanced Analytics",
        "distribution": "Distribution",
        "program_evolution": "Program Evolution",
        "monthly_growth": "Monthly Growth (MoM)",
        "heatmap_title": "Heatmap – Volume by Tranche",
        "ren_intensity": "Renegotiation Intensity",
        "top_n_institutions": "Top N Institutions",
        "treemap_title": "Treemap – Distribution by Segment",
        "regional_distribution": "Regional Distribution",
        "volume_and_ticket_by_region": "Volume and Average Ticket by Region",
        "regional_share": "Regional Share",
        "advanced_analytics_title": "Advanced Analytics",
        "clustering_title": "Clustering (Operations × Ticket)",
        "concentration_radar": "Concentration Radar",
        "scatter_title": "Avg Ticket vs Market Share",
        "pareto_title": "Pareto Curve",
        "pareto_interpretation": "Interpretation",
        "pareto_text": "institutions concentrate 80% of the total renegotiated volume.",
        "export_section": "Export",
        "csv_download": "CSV (filtered data)",
        "report_download": "TXT Report",
        "footer_text": "Debt Settlement Brazil · Financial Intelligence",
        "footer_source": "Source: Central Bank of Brazil (SCR)",
        "alert_sharp_drop": "Sharp Drop – volume fell {:.1f}%",
        "alert_slowdown": "Slowdown – drop of {:.1f}%",
        "alert_strong_accel": "Strong Acceleration – +{:.1f}%",
        "alert_stable_growth": "Stable Growth – +{:.1f}%",
        "alert_high_concentration": "High Concentration – HHI > {}",
        "alert_moderate_concentration": "Moderate Concentration – HHI {}-{}",
        "alert_competitive": "Competitive Market – HHI < {}",
        "alert_high_regional_inequality": "High Regional Inequality – Gini = {:.2f}",
        "alert_regional_inequality": "Regional Inequality – Gini = {:.2f}",
        "insight_regional": "The <b>{}</b> region concentrates <b>{:.1f}%</b> of total volume.",
        "insight_area": "<b>{}</b> leads investments.",
        "insight_trend": "Average monthly growth of <b>{:+.1f}%</b>.",
        "insight_hhi": "Herfindahl-Hirschman Index: <b>{:.0f}</b>",
    },
    "pt": {
        "app_title": "Desenrola Brasil",
        "app_subtitle": "Plataforma de Inteligência",
        "page_title": "Desenrola Brasil | Plataforma de Inteligência",
        "hero_title": "Desenrola Brasil",
        "hero_subtitle": "Inteligência analítica para renegociação de dívidas · Fonte: Banco Central (SCR)",
        "volume": "Volume Renegociado",
        "contracts": "Total de Contratos",
        "avg_ticket": "Ticket Médio",
        "institutions": "Instituições",
        "states": "Estados",
        "vs_prev_month": "vs mês anterior",
        "smart_alerts": "Alertas Inteligentes",
        "automated_insights": "Insights Automáticos",
        "regional_concentration": "Concentração Regional",
        "leading_area": "Área Líder",
        "trend": "Tendência",
        "concentration_hhi": "Concentração (HHI)",
        "low_concentration": "Baixa",
        "moderate_concentration": "Moderada",
        "high_concentration": "Alta",
        "time_series": "Evolução Temporal",
        "bank_concentration": "Concentração Bancária",
        "regional_analysis": "Análise Regional",
        "advanced_analytics": "Análise Avançada",
        "distribution": "Distribuição",
        "program_evolution": "Evolução do Programa",
        "monthly_growth": "Crescimento Mensal (MoM)",
        "heatmap_title": "Mapa de Calor – Volume por Faixa",
        "ren_intensity": "Intensidade de Renegociação",
        "top_n_institutions": "Top N Instituições",
        "treemap_title": "Treemap – Distribuição por Segmento",
        "regional_distribution": "Distribuição Regional",
        "volume_and_ticket_by_region": "Volume e Ticket Médio por Região",
        "regional_share": "Participação Regional",
        "advanced_analytics_title": "Análises Avançadas",
        "clustering_title": "Clusterização (Operações × Ticket)",
        "concentration_radar": "Radar de Concentração",
        "scatter_title": "Ticket Médio vs Market Share",
        "pareto_title": "Curva de Pareto",
        "pareto_interpretation": "Interpretação",
        "pareto_text": "instituições concentram 80% do volume total renegociado.",
        "export_section": "Exportar",
        "csv_download": "CSV (dados filtrados)",
        "report_download": "Relatório TXT",
        "footer_text": "Desenrola Brasil · Inteligência Financeira",
        "footer_source": "Fonte: Banco Central do Brasil (SCR)",
        "alert_sharp_drop": "Queda Abrupta – volume caiu {:.1f}%",
        "alert_slowdown": "Desaceleração – queda de {:.1f}%",
        "alert_strong_accel": "Aceleração Forte – +{:.1f}%",
        "alert_stable_growth": "Crescimento Estável – +{:.1f}%",
        "alert_high_concentration": "Concentração Elevada – HHI > {}",
        "alert_moderate_concentration": "Concentração Moderada – HHI {}-{}",
        "alert_competitive": "Mercado Competitivo – HHI < {}",
        "alert_high_regional_inequality": "Alta Desigualdade Regional – Gini = {:.2f}",
        "alert_regional_inequality": "Desigualdade Regional – Gini = {:.2f}",
        "insight_regional": "A região <b>{}</b> concentra <b>{:.1f}%</b> do volume total.",
        "insight_area": "<b>{}</b> lidera os investimentos.",
        "insight_trend": "Crescimento médio mensal de <b>{:+.1f}%</b>.",
        "insight_hhi": "Índice Herfindahl-Hirschman: <b>{:.0f}</b>",
    }
}

# ============================================================
# LOGGING
# ============================================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================
# DESIGN SYSTEM – GLOBAL TOKENS (with caching)
# ============================================================
st.set_page_config(
    page_title="Debt Settlement Brazil | Intelligence Platform",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize theme and language
if "tema" not in st.session_state:
    st.session_state.tema = "dark"
if "lang" not in st.session_state:
    st.session_state.lang = "en"  # default English

T = st.session_state.tema
LANG = st.session_state.lang
TEXT = TEXTS[LANG]

# ============================================================
# COLOR PALETTE (unchanged, but with function)
# ============================================================
def get_colors(theme: str) -> Dict[str, str]:
    """Return color palette based on theme."""
    if theme == "light":
        return {
            "BG": "#F7F9FC",
            "CARD": "#FFFFFF",
            "CARD_GLASS": "rgba(255,255,255,0.85)",
            "TXT": "#1A2B4C",
            "TXT2": "#5A6E8A",
            "BORDA": "#E2E8F0",
            "BORDA_GLOW": "rgba(0,168,107,0.2)",
            "P1": "#00A86B",
            "P2": "#0066CC",
            "P3": "#52B788",
            "ACCENT": "#0066CC",
            "ACCENT_GLOW": "rgba(0,102,204,0.2)",
            "VERDE": "#00A86B",
            "VERM": "#DC2626",
            "AMBER": "#F59E0B",
            "AZUL": "#3B82F6",
            "ROXO": "#8B5CF6",
            "CINZA": "#6B7280",
            "TPLOTE": "plotly_white",
            "GRID": "rgba(0,0,0,0.05)",
            "GLOW_P1": "rgba(0,168,107,0.3)",
            "CORES_GRAFICOS": ["#00A86B", "#F59E0B", "#DC2626", "#3B82F6", "#8B5CF6", "#0066CC", "#00A86B", "#6B7280"]
        }
    else:  # dark
        return {
            "BG": "#0A0F1C",
            "CARD": "#111827",
            "CARD_GLASS": "rgba(17,24,39,0.85)",
            "TXT": "#F1F5F9",
            "TXT2": "#94A3B8",
            "BORDA": "#1F2937",
            "BORDA_GLOW": "rgba(56,189,248,0.2)",
            "P1": "#3FB68C",
            "P2": "#3B82F6",
            "P3": "#10B981",
            "ACCENT": "#60A5FA",
            "ACCENT_GLOW": "rgba(96,165,250,0.2)",
            "VERDE": "#34D399",
            "VERM": "#F87171",
            "AMBER": "#FBBF24",
            "AZUL": "#60A5FA",
            "ROXO": "#A78BFA",
            "CINZA": "#6B7280",
            "TPLOTE": "plotly_dark",
            "GRID": "rgba(255,255,255,0.05)",
            "GLOW_P1": "rgba(63,182,140,0.3)",
            "CORES_GRAFICOS": ["#3FB68C", "#FBBF24", "#F87171", "#60A5FA", "#A78BFA", "#3B82F6", "#34D399", "#6B7280"]
        }

COLORS = get_colors(T)
CORES_GRAFICOS = COLORS["CORES_GRAFICOS"]

# ============================================================
# CSS (could be moved to external file)
# ============================================================
CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Playfair+Display:wght@400;500;600;700&display=swap');

* {{ margin: 0; padding: 0; box-sizing: border-box; }}

.stApp {{
    background: {COLORS["BG"]};
    font-family: 'Inter', sans-serif;
}}

.block-container {{
    padding: 1rem 1.5rem !important;
    max-width: 1600px;
    margin: 0 auto;
}}

/* ===== HERO ===== */
.hero {{
    background: linear-gradient(135deg, rgba(0,168,107,0.08) 0%, rgba(0,102,204,0.05) 100%);
    backdrop-filter: blur(10px);
    border: 1px solid {COLORS["BORDA_GLOW"]};
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
    background: radial-gradient(circle, {COLORS["ACCENT_GLOW"]} 0%, transparent 70%);
    pointer-events: none;
}}

.hero h1 {{
    font-size: 2.5rem;
    font-weight: 800;
    background: linear-gradient(135deg, {COLORS["TXT"]}, {COLORS["ACCENT"]});
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
    margin-bottom: 0.25rem;
    font-family: 'Playfair Display', serif;
}}

.hero p {{
    font-size: 0.9rem;
    color: {COLORS["TXT2"]};
    margin-bottom: 1rem;
}}

.hero-badge {{
    display: inline-block;
    background: rgba(0,168,107,0.15);
    border: 1px solid {COLORS["BORDA_GLOW"]};
    padding: 0.25rem 0.8rem;
    border-radius: 40px;
    font-size: 0.7rem;
    color: {COLORS["P1"]};
    margin-right: 0.5rem;
    margin-bottom: 0.5rem;
}}

/* ===== KPI CARDS (now using st.metric, but keep card style) ===== */
/* We'll use st.metric with custom styling via st.markdown wrapper */

/* ===== INSIGHT CARDS ===== */
.insight-grid {{
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1rem;
    margin: 1rem 0;
}}

@media (max-width: 800px) {{
    .insight-grid {{ grid-template-columns: 1fr; }}
}}

.insight-card {{
    background: {COLORS["CARD_GLASS"]};
    backdrop-filter: blur(8px);
    border: 1px solid {COLORS["BORDA"]};
    border-radius: 16px;
    padding: 1rem 1.2rem;
    border-left: 3px solid {COLORS["P1"]};
    transition: all 0.3s;
}}

.insight-card:hover {{
    transform: translateX(4px);
    border-color: {COLORS["ACCENT"]};
}}

.insight-title {{
    font-size: 0.65rem;
    font-weight: 700;
    text-transform: uppercase;
    color: {COLORS["TXT2"]};
    margin-bottom: 0.5rem;
}}

.insight-text {{
    font-size: 0.85rem;
    color: {COLORS["TXT"]};
    line-height: 1.4;
}}

.insight-value {{
    font-size: 1.1rem;
    font-weight: 700;
    color: {COLORS["P1"]};
    margin-top: 0.5rem;
}}

/* ===== ALERTS ===== */
.al {{
    padding: 0.6rem 1rem;
    border-radius: 12px;
    margin-bottom: 0.5rem;
    font-size: 0.8rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}}
.al.er {{ background: rgba(220,38,38,0.1); border-left: 3px solid {COLORS["VERM"]}; color: {COLORS["VERM"]}; }}
.al.wa {{ background: rgba(245,158,11,0.1); border-left: 3px solid {COLORS["AMBER"]}; color: {COLORS["AMBER"]}; }}
.al.ok {{ background: rgba(0,168,107,0.1); border-left: 3px solid {COLORS["VERDE"]}; color: {COLORS["VERDE"]}; }}
.al.in {{ background: rgba(59,130,246,0.1); border-left: 3px solid {COLORS["AZUL"]}; color: {COLORS["AZUL"]}; }}

/* ===== PREMIUM TABS ===== */
.stTabs [data-baseweb="tab-list"] {{
    gap: 0.5rem;
    background: transparent;
}}
.stTabs [data-baseweb="tab"] {{
    background: {COLORS["CARD_GLASS"]};
    backdrop-filter: blur(8px);
    border-radius: 40px;
    padding: 0.5rem 1.2rem;
    font-size: 0.8rem;
    font-weight: 500;
    color: {COLORS["TXT2"]};
    border: 1px solid {COLORS["BORDA"]};
    transition: all 0.2s;
}}
.stTabs [aria-selected="true"] {{
    background: {COLORS["P1"]};
    color: white;
    border-color: {COLORS["P1"]};
}}

/* ===== SIDEBAR ===== */
section[data-testid="stSidebar"] {{
    background: {COLORS["CARD_GLASS"]};
    backdrop-filter: blur(12px);
    border-right: 1px solid {COLORS["BORDA"]};
}}

/* ===== FOOTER ===== */
.footer {{
    text-align: center;
    padding: 1.5rem 0 0.5rem;
    margin-top: 2rem;
    border-top: 1px solid {COLORS["BORDA"]};
    font-size: 0.65rem;
    color: {COLORS["TXT2"]};
}}
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)

# ============================================================
# UTILITY FUNCTIONS (with type hints and docstrings)
# ============================================================
def rgba(hex_color: str, a: float = 0.15) -> str:
    """Convert hex color to rgba string."""
    h = hex_color.lstrip('#')
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{a})"

def fmt_brl(v: float) -> str:
    """Format number as Brazilian Real currency."""
    if pd.isna(v) or v == 0:
        return "R$ 0"
    if v >= 1e9:
        return f"R$ {v/1e9:.2f}B".replace(".", ",")
    if v >= 1e6:
        return f"R$ {v/1e6:.1f}M".replace(".", ",")
    return f"R$ {v:,.0f}".replace(",", ".")

def fmt_num(v: float) -> str:
    """Format number with commas as thousand separators."""
    if pd.isna(v):
        return "0"
    return f"{int(v):,}".replace(",", ".")

def classify_bank(name: str) -> str:
    """Classify bank into segment based on name."""
    n = re.sub(r'\s*-\s*PRUDENCIAL$', '', str(name).upper().strip())
    if any(x in n for x in ["NUBANK", "INTER", "C6", "NEON", "ORIGINAL", "PAN", "NEXT"]):
        return "Digital"
    if any(x in n for x in ["ITAU", "BRADESCO", "SANTANDER", "CAIXA", "BANCO DO BRASIL", "BB"]):
        return "Traditional"
    if any(x in n for x in ["BTG", "XP", "MODAL", "GENIAL"]):
        return "Investment"
    if any(x in n for x in ["SICOOB", "SICREDI"]):
        return "Cooperative"
    return "Other"

def classify_region(uf: str) -> str:
    """Map Brazilian state code to region."""
    mapping = {
        "North": ["AC", "AM", "AP", "PA", "RO", "RR", "TO"],
        "Northeast": ["AL", "BA", "CE", "MA", "PB", "PE", "PI", "RN", "SE"],
        "Central-West": ["DF", "GO", "MS", "MT"],
        "Southeast": ["ES", "MG", "RJ", "SP"],
        "South": ["PR", "RS", "SC"]
    }
    for region, states in mapping.items():
        if uf in states:
            return region
    return "Not Identified"

def hhi(df: pd.DataFrame, col: str) -> float:
    """Calculate Herfindahl-Hirschman Index."""
    total = df[col].sum()
    if total == 0:
        return 0.0
    return ((df[col] / total) ** 2).sum() * 10000

def gini(s: pd.Series) -> float:
    """Calculate Gini coefficient."""
    a = np.sort(s.dropna().values)
    n = len(a)
    if n == 0 or a.sum() == 0:
        return 0.0
    return (2 * np.sum(np.arange(1, n + 1) * a) / (n * a.sum()) - (n + 1) / n)

def base_layout(fig: go.Figure, h: int = 440, leg: bool = True) -> go.Figure:
    """Apply common layout styling to Plotly figures."""
    fig.update_layout(
        template=COLORS["TPLOTE"], height=h,
        margin=dict(l=50, r=40, t=55, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=COLORS["TXT"], family="Inter", size=12),
        hovermode="x unified",
        showlegend=leg,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        transition=dict(duration=300)
    )
    fig.update_xaxes(showgrid=False, color=COLORS["TXT"], linecolor=COLORS["BORDA"])
    fig.update_yaxes(showgrid=True, gridcolor=COLORS["GRID"], color=COLORS["TXT"])
    return fig

# ============================================================
# DATA LOADING WITH CACHE
# ============================================================
@st.cache_data(ttl=3600)
def load_data(file_path: str = "dados_desenrola.csv") -> Optional[pd.DataFrame]:
    """Load and preprocess the dataset."""
    for enc in ["utf-8", "latin1", "cp1252"]:
        try:
            df = pd.read_csv(file_path, sep=";", encoding=enc, low_memory=False)
            df.columns = df.columns.str.lower().str.strip()
            # Convert numeric columns
            for c in ["numero_operacoes", "volume_operacoes"]:
                if c in df.columns:
                    df[c] = pd.to_numeric(
                        df[c].astype(str).str.replace(".", "", regex=False)
                        .str.replace(",", ".", regex=False),
                        errors="coerce"
                    )
            df["data_base"] = pd.to_datetime(df["data_base"].astype(str), format="%Y%m", errors="coerce")
            df["tipo_banco"] = df["nome_conglomerado_financeiro"].apply(classify_bank)
            df["regiao"] = df["unidade_federacao"].apply(classify_region)
            df = df.dropna(subset=["volume_operacoes", "numero_operacoes"])
            logger.info(f"Data loaded successfully: {len(df)} records")
            return df
        except Exception as e:
            logger.warning(f"Failed with encoding {enc}: {e}")
            continue
    logger.error("Could not load data with any encoding.")
    return None

# ============================================================
# CACHED AGGREGATION FUNCTIONS
# ============================================================
@st.cache_data
def get_aggregates(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute all aggregate metrics from filtered dataframe."""
    vol_tot = df["volume_operacoes"].sum()
    ops_tot = df["numero_operacoes"].sum()
    ticket = vol_tot / ops_tot if ops_tot > 0 else 0.0

    # Evolution
    evol_g = df.groupby("data_base")["volume_operacoes"].sum().reset_index().sort_values("data_base")
    evol_g["mom"] = evol_g["volume_operacoes"].pct_change() * 100
    mom_last = evol_g["mom"].dropna().iloc[-1] if not evol_g["mom"].dropna().empty else 0.0

    # Concentration
    b_agg = df.groupby("nome_conglomerado_financeiro")["numero_operacoes"].sum().reset_index()
    hhi_val = hhi(b_agg, "numero_operacoes")

    # Regional inequality
    reg_v = df.groupby("regiao")["volume_operacoes"].sum()
    gini_r = gini(reg_v)

    # Other
    n_banks = df["nome_conglomerado_financeiro"].nunique()
    n_uf = df["unidade_federacao"].nunique()

    return {
        "vol_tot": vol_tot,
        "ops_tot": ops_tot,
        "ticket": ticket,
        "evol_g": evol_g,
        "mom_last": mom_last,
        "hhi": hhi_val,
        "gini_r": gini_r,
        "n_banks": n_banks,
        "n_uf": n_uf,
        "reg_df": reg_v.reset_index(),
        "b_agg": b_agg,
    }

@st.cache_data
def get_region_leader(reg_df: pd.DataFrame) -> Tuple[str, float, float]:
    """Get region with highest volume."""
    if reg_df.empty:
        return "", 0.0, 0.0
    idx = reg_df["volume_operacoes"].idxmax()
    region = reg_df.loc[idx, "regiao"]
    vol = reg_df.loc[idx, "volume_operacoes"]
    pct = vol / reg_df["volume_operacoes"].sum() * 100
    return region, vol, pct

# ============================================================
# CACHED CHART FUNCTIONS
# ============================================================
@st.cache_data
def create_timeline_chart(df: pd.DataFrame, evol_g: pd.DataFrame, colors: List[str]) -> go.Figure:
    """Create timeline chart with tranche breakdown and forecast."""
    fig = go.Figure()
    tipos = sorted(df["tipo_desenrola"].unique())
    for i, tp in enumerate(tipos):
        g = df[df["tipo_desenrola"] == tp].groupby("data_base")["volume_operacoes"].sum().reset_index()
        cor = colors[i % len(colors)]
        fig.add_trace(go.Scatter(
            x=g["data_base"], y=g["volume_operacoes"],
            mode="lines+markers", name=f"Tranche {tp}",
            line=dict(color=cor, width=2.5),
            marker=dict(size=6, color=cor),
            hovertemplate=f"<b>Tranche {tp}</b><br>%{{x|%b/%Y}}<br>R$ %{{y:,.0f}}<extra></extra>"
        ))

    # Forecast using Exponential Smoothing
    try:
        serie = evol_g["volume_operacoes"]
        if len(serie) >= 4:
            hw = ExponentialSmoothing(serie.values, trend="add", seasonal=None,
                                      initialization_method="estimated").fit()
            periods = CONFIG["forecast_periods"]
            prev = hw.forecast(periods)
            dt_fut = pd.date_range(evol_g["data_base"].max(), periods=periods+1, freq="MS")[1:]
            sigma = float(np.std(hw.resid))
            low = [float(v) for v in prev - 1.96 * sigma]
            upp = [float(v) for v in prev + 1.96 * sigma]
            xband = list(dt_fut) + list(dt_fut[::-1])
            yband = upp + low[::-1]
            fig.add_trace(go.Scatter(
                x=xband, y=yband, fill="toself",
                fillcolor=rgba(COLORS["AMBER"], 0.15), line=dict(color="rgba(0,0,0,0)"),
                name="95% CI", hoverinfo="skip"
            ))
            fig.add_trace(go.Scatter(
                x=dt_fut, y=prev, mode="lines+markers", name="Projection",
                line=dict(color=COLORS["AMBER"], width=2, dash="dash"),
                marker=dict(size=7, symbol="diamond", color=COLORS["AMBER"])
            ))
    except Exception as e:
        logger.exception("Forecast failed: %s", e)

    fig.update_layout(title=TEXT["program_evolution"])
    base_layout(fig, h=420)
    fig.update_yaxes(tickprefix="R$ ", tickformat=",.0f")
    return fig

@st.cache_data
def create_mom_chart(evol_g: pd.DataFrame) -> go.Figure:
    """Create month-over-month growth bar chart."""
    fig = go.Figure(go.Bar(
        x=evol_g["data_base"], y=evol_g["mom"],
        marker_color=[COLORS["VERDE"] if v >= 0 else COLORS["VERM"] for v in evol_g["mom"]],
        hovertemplate="%{x|%b/%Y}<br>MoM: %{y:.1f}%<extra></extra>"
    ))
    fig.add_hline(y=0, line_color=COLORS["BORDA"], line_width=1.5)
    fig.update_layout(title=TEXT["monthly_growth"])
    base_layout(fig, h=420, leg=False)
    fig.update_yaxes(ticksuffix="%")
    return fig

@st.cache_data
def create_heatmap(df: pd.DataFrame) -> Optional[go.Figure]:
    """Create heatmap of volume by tranche and month."""
    try:
        heat_data = df.pivot_table(index="tipo_desenrola", columns="data_base",
                                   values="volume_operacoes", aggfunc="sum")
        heat_data.columns = [pd.Timestamp(c).strftime("%b/%y") for c in heat_data.columns]
        fig = go.Figure(go.Heatmap(
            z=heat_data.values, x=heat_data.columns, y=heat_data.index,
            colorscale=[[0, COLORS["BG"]], [0.5, COLORS["P2"]], [1, COLORS["P1"]]],
            hovertemplate="Tranche: %{y}<br>Month: %{x}<br>R$ %{z:,.0f}<extra></extra>"
        ))
        base_layout(fig, h=320, leg=False)
        fig.update_layout(title=TEXT["ren_intensity"])
        return fig
    except Exception as e:
        logger.exception("Heatmap creation failed: %s", e)
        return None

@st.cache_data
def create_concentration_chart(df: pd.DataFrame, top_n: int) -> go.Figure:
    """Create horizontal bar chart of top N banks by volume."""
    banco_agg = df.groupby("nome_conglomerado_financeiro").agg(
        volume=("volume_operacoes", "sum"),
        ops=("numero_operacoes", "sum")
    ).reset_index()
    banco_agg["ticket"] = banco_agg["volume"] / banco_agg["ops"]
    banco_agg["seg"] = banco_agg["nome_conglomerado_financeiro"].apply(classify_bank)
    banco_agg = banco_agg.nlargest(top_n, "volume")

    fig = go.Figure(go.Bar(
        x=banco_agg["volume"], y=banco_agg["nome_conglomerado_financeiro"].str[:30],
        orientation="h",
        marker=dict(color=COLORS["P1"], line=dict(width=0)),
        text=banco_agg["volume"].apply(lambda x: fmt_brl(x)), textposition="outside",
        hovertemplate="<b>%{y}</b><br>Volume: R$ %{x:,.0f}<br>Ops: %{customdata[0]:,.0f}<br>Ticket: R$ %{customdata[1]:,.0f}<extra></extra>",
        customdata=banco_agg[["ops", "ticket"]].values
    ))
    fig.update_layout(title=f"{TEXT['top_n_institutions']} ({top_n})")
    base_layout(fig, h=500, leg=False)
    fig.update_xaxes(tickprefix="R$ ", tickformat=".2s")
    return fig

@st.cache_data
def create_treemap(df: pd.DataFrame, top_n: int) -> go.Figure:
    """Create treemap by segment."""
    banco_agg = df.groupby("nome_conglomerado_financeiro").agg(
        volume=("volume_operacoes", "sum"),
        ops=("numero_operacoes", "sum")
    ).reset_index()
    banco_agg["ticket"] = banco_agg["volume"] / banco_agg["ops"]
    banco_agg["seg"] = banco_agg["nome_conglomerado_financeiro"].apply(classify_bank)
    banco_agg = banco_agg.nlargest(top_n, "volume")
    fig = px.treemap(banco_agg, path=["seg", "nome_conglomerado_financeiro"],
                     values="volume", color="ticket",
                     color_continuous_scale=[COLORS["P1"], COLORS["P2"], COLORS["P3"]],
                     title=TEXT["treemap_title"])
    fig.update_layout(template=COLORS["TPLOTE"], height=450)
    return fig

@st.cache_data
def create_regional_charts(df: pd.DataFrame) -> Tuple[go.Figure, go.Figure, go.Figure]:
    """Create regional charts: map, bar+scatter, donut."""
    uf_data = df.groupby("unidade_federacao")["volume_operacoes"].sum().reset_index()
    uf_data.columns = ["uf", "volume"]

    # Map
    try:
        fig_map = px.choropleth(
            uf_data, locations="uf", locationmode="BRA-states",
            color="volume", color_continuous_scale="Blues",
            title=TEXT["regional_distribution"],
            hover_name="uf", hover_data={"volume": ":,.0f"}
        )
        fig_map.update_layout(geo=dict(bgcolor="rgba(0,0,0,0)"), height=500,
                              margin=dict(l=0, r=0, t=40, b=0))
        fig_map.update_layout(template=COLORS["TPLOTE"], paper_bgcolor="rgba(0,0,0,0)")
    except Exception as e:
        logger.exception("Map creation failed: %s", e)
        fig_map = go.Figure()
        fig_map.add_annotation(text="Map unavailable", x=0.5, y=0.5, showarrow=False)

    # Bar + scatter by region
    reg_data = df.groupby("regiao").agg(
        volume=("volume_operacoes", "sum"),
        ops=("numero_operacoes", "sum")
    ).reset_index()
    reg_data["ticket"] = reg_data["volume"] / reg_data["ops"]

    fig_bar = make_subplots(specs=[[{"secondary_y": True}]])
    fig_bar.add_trace(go.Bar(x=reg_data["regiao"], y=reg_data["volume"], name="Volume",
                             marker_color=CORES_GRAFICOS[:len(reg_data)]), secondary_y=False)
    fig_bar.add_trace(go.Scatter(x=reg_data["regiao"], y=reg_data["ticket"], name="Avg Ticket",
                                 mode="lines+markers",
                                 line=dict(color=COLORS["AMBER"], width=2.5),
                                 marker=dict(size=8, symbol="diamond")), secondary_y=True)
    fig_bar.update_layout(title=TEXT["volume_and_ticket_by_region"])
    base_layout(fig_bar, h=450)
    fig_bar.update_yaxes(title_text="Volume (R$)", tickprefix="R$ ", tickformat=".2s", secondary_y=False)
    fig_bar.update_yaxes(title_text="Avg Ticket (R$)", tickprefix="R$ ", secondary_y=True, showgrid=False)

    # Donut
    fig_donut = go.Figure(go.Pie(
        labels=reg_data["regiao"], values=reg_data["volume"], hole=0.5,
        marker=dict(colors=CORES_GRAFICOS[:len(reg_data)], line=dict(color=COLORS["BG"], width=2)),
        textinfo="label+percent",
        hovertemplate="<b>%{label}</b><br>R$ %{value:,.0f}<br>%{percent}<extra></extra>"
    ))
    fig_donut.update_layout(title=TEXT["regional_share"], height=400)
    base_layout(fig_donut, h=400)

    return fig_map, fig_bar, fig_donut

@st.cache_data
def create_cluster_chart(df: pd.DataFrame) -> Optional[go.Figure]:
    """Create clustering scatter plot."""
    try:
        cl_df = df.groupby("nome_conglomerado_financeiro").agg(
            ops=("numero_operacoes", "sum"),
            vol=("volume_operacoes", "sum")
        ).reset_index()
        cl_df["ticket"] = cl_df["vol"] / cl_df["ops"]
        cl_df = cl_df[cl_df["ops"] > CONFIG["min_ops_for_cluster"]].dropna()

        if len(cl_df) < CONFIG["min_cluster_samples"]:
            return None

        scaler = StandardScaler()
        feat = scaler.fit_transform(cl_df[["ops", "ticket"]])
        n_clusters = min(3, len(cl_df))
        km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cl_df["cluster"] = km.fit_predict(feat)

        fig = go.Figure()
        for c in cl_df["cluster"].unique():
            grp = cl_df[cl_df["cluster"] == c]
            sz = np.log1p(grp["vol"] / grp["vol"].max() + 0.01) * 25 + 9
            fig.add_trace(go.Scatter(
                x=grp["ops"], y=grp["ticket"], mode="markers", name=f"Cluster {c+1}",
                marker=dict(size=sz, color=CORES_GRAFICOS[c % len(CORES_GRAFICOS)], opacity=0.8,
                            line=dict(width=1, color=COLORS["BORDA"])),
                text=grp["nome_conglomerado_financeiro"],
                hovertemplate="<b>%{text}</b><br>Ops: %{x:,.0f}<br>Ticket: R$ %{y:,.2f}<extra></extra>"
            ))
        fig.update_layout(title=TEXT["clustering_title"],
                          xaxis_title="Operations", yaxis_title="Avg Ticket (R$)")
        base_layout(fig, h=450)
        return fig
    except Exception as e:
        logger.exception("Clustering failed: %s", e)
        return None

@st.cache_data
def create_radar_chart(hhi_val: float, cr3: float, cr5: float, gini_r: float, ticket: float) -> go.Figure:
    """Create radar chart of concentration indices."""
    # Normalize values
    radar_data = pd.DataFrame({
        "Metric": ["HHI", "CR3", "CR5", "Regional Gini", "Avg Ticket"],
        "Normalized Value": [
            min(hhi_val / 3000, 1),
            cr3 / 100,
            cr5 / 100,
            min(gini_r, 1),
            min(ticket / 15000, 1)
        ]
    })
    fig = px.line_polar(radar_data, r="Normalized Value", theta="Metric", line_close=True,
                        color_discrete_sequence=[COLORS["P1"]],
                        title=TEXT["concentration_radar"])
    fig.update_layout(template=COLORS["TPLOTE"], height=450)
    return fig

@st.cache_data
def create_scatter_chart(df: pd.DataFrame) -> go.Figure:
    """Create scatter plot of avg ticket vs market share."""
    sc_df = df.groupby("nome_conglomerado_financeiro").agg(
        vol=("volume_operacoes", "sum"),
        ops=("numero_operacoes", "sum")
    ).reset_index()
    sc_df["ticket"] = sc_df["vol"] / sc_df["ops"]
    sc_df["ms"] = sc_df["ops"] / sc_df["ops"].sum() * 100
    sc_df = sc_df.dropna().query("ops>50")

    fig = px.scatter(sc_df, x="ms", y="ticket", size="vol", color="ticket",
                     hover_name="nome_conglomerado_financeiro",
                     title=TEXT["scatter_title"],
                     labels={"ms": "Market Share (%)", "ticket": "Avg Ticket (R$)"},
                     color_continuous_scale="Viridis")
    fig.update_layout(template=COLORS["TPLOTE"], height=450)
    return fig

@st.cache_data
def create_pareto_chart(df: pd.DataFrame) -> Tuple[go.Figure, int]:
    """Create Pareto chart and return figure and count for 80%."""
    pareto_data = df.groupby("nome_conglomerado_financeiro")["volume_operacoes"].sum().sort_values(ascending=False).reset_index()
    pareto_data["acum"] = pareto_data["volume_operacoes"].cumsum() / pareto_data["volume_operacoes"].sum() * 100
    p80 = (pareto_data["acum"] <= 80).sum()

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=pareto_data["nome_conglomerado_financeiro"].str[:20],
                         y=pareto_data["volume_operacoes"],
                         name="Volume", marker_color=COLORS["P1"]), secondary_y=False)
    fig.add_trace(go.Scatter(x=pareto_data["nome_conglomerado_financeiro"].str[:20],
                             y=pareto_data["acum"],
                             name="Cumulative %", mode="lines+markers",
                             line=dict(color=COLORS["AMBER"], width=2.5)), secondary_y=True)
    fig.add_hline(y=80, line_dash="dot", line_color=COLORS["VERM"], secondary_y=True, annotation_text="80%")
    fig.update_layout(title=f"Pareto: {p80} institutions = 80% of volume")
    base_layout(fig, h=450)
    fig.update_yaxes(title_text="Volume (R$)", tickprefix="R$ ", secondary_y=False)
    fig.update_yaxes(title_text="Cumulative %", ticksuffix="%", secondary_y=True)
    return fig, p80

# ============================================================
# MAIN APP
# ============================================================
# Load data
with st.spinner("🔄 Loading data..."):
    df = load_data()

if df is None:
    st.error("❌ Failed to load 'dados_desenrola.csv'. Please check file path.")
    st.stop()

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown(f"""
    <div style="text-align:center; padding:0.5rem 0 1rem; border-bottom:1px solid {COLORS['BORDA']}; margin-bottom:1rem;">
        <div style="font-family:'Playfair Display',serif; font-size:1.2rem; font-weight:700;">🏦 {TEXT['app_title']}</div>
        <div style="font-size:0.65rem; color:{COLORS['TXT2']};">{TEXT['app_subtitle']}</div>
    </div>
    """, unsafe_allow_html=True)

    # Theme and Language toggles
    col_a1, col_a2 = st.columns(2)
    with col_a1:
        if st.button("☀️ Light", use_container_width=True):
            st.session_state.tema = "light"
            st.rerun()
    with col_a2:
        if st.button("🌙 Dark", use_container_width=True):
            st.session_state.tema = "dark"
            st.rerun()

    # Language selector
    lang_sel = st.selectbox("🌐 Language", options=["en", "pt"], index=0 if LANG=="en" else 1,
                            format_func=lambda x: "English" if x=="en" else "Português")
    if lang_sel != LANG:
        st.session_state.lang = lang_sel
        st.rerun()

    st.markdown("---")
    st.markdown("**🔍 Filters**")

    tipos = sorted(df["tipo_desenrola"].unique())
    sel_tip = st.multiselect("Tranche", tipos, default=tipos)

    regioes = sorted(df["regiao"].unique())
    sel_reg = st.multiselect("Region", regioes, default=regioes)

    segmentos = sorted(df["tipo_banco"].unique())
    sel_seg = st.multiselect("Segment", segmentos, default=segmentos)

    datas = sorted(df["data_base"].unique())
    if len(datas) > 1:
        i0, i1 = st.select_slider("Period", options=list(range(len(datas))),
                                  value=(0, len(datas) - 1),
                                  format_func=lambda i: pd.Timestamp(datas[i]).strftime("%m/%Y"))
        d_ini, d_fim = datas[i0], datas[i1]
    else:
        d_ini = d_fim = datas[0]

    if st.button("🔄 Reset", use_container_width=True):
        st.rerun()

    st.markdown("---")
    st.markdown(f"""
    <div style="background:{COLORS['CARD_GLASS']}; border-radius:12px; padding:0.8rem; border:1px solid {COLORS['BORDA']};">
        <div style="font-size:0.7rem; font-weight:600;">📋 Data Quality</div>
        <div style="font-size:0.65rem; color:{COLORS['TXT2']};">Records: <b>{len(df):,}</b></div>
        <div style="font-size:0.65rem; color:{COLORS['TXT2']};">Period: {d_ini.strftime('%m/%Y')} → {d_fim.strftime('%m/%Y')}</div>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# FILTERING
# ============================================================
dff = df[
    (df["tipo_desenrola"].isin(sel_tip)) &
    (df["regiao"].isin(sel_reg)) &
    (df["tipo_banco"].isin(sel_seg)) &
    (df["data_base"] >= d_ini) &
    (df["data_base"] <= d_fim)
]

if dff.empty:
    st.warning("⚠️ No data matches the selected filters.")
    st.stop()

# ============================================================
# COMPUTE AGGREGATES (cached)
# ============================================================
agg = get_aggregates(dff)
vol_tot = agg["vol_tot"]
ops_tot = agg["ops_tot"]
ticket = agg["ticket"]
evol_g = agg["evol_g"]
mom_last = agg["mom_last"]
hhi_val = agg["hhi"]
gini_r = agg["gini_r"]
n_banks = agg["n_banks"]
n_uf = agg["n_uf"]
reg_df = agg["reg_df"]
b_agg = agg["b_agg"]

# Additional metrics
cr3 = b_agg.nlargest(3, "numero_operacoes")["numero_operacoes"].sum() / b_agg["numero_operacoes"].sum() * 100 if not b_agg.empty else 0
cr5 = b_agg.nlargest(5, "numero_operacoes")["numero_operacoes"].sum() / b_agg["numero_operacoes"].sum() * 100 if not b_agg.empty else 0

# ============================================================
# HERO SECTION
# ============================================================
st.markdown(f"""
<div class="hero">
    <h1>🏦 {TEXT['hero_title']}</h1>
    <p>{TEXT['hero_subtitle']}</p>
    <div>
        <span class="hero-badge">📊 {fmt_num(ops_tot)} contracts</span>
        <span class="hero-badge">💰 {fmt_brl(vol_tot)} renegotiated</span>
        <span class="hero-badge">🏛️ {n_banks} institutions</span>
        <span class="hero-badge">🗺️ {n_uf} states</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# KPIs USING ST.METRIC
# ============================================================
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(
        label=TEXT["volume"],
        value=fmt_brl(vol_tot),
        delta=f"{mom_last:+.1f}% {TEXT['vs_prev_month']}",
        delta_color="normal"
    )
with col2:
    st.metric(label=TEXT["contracts"], value=fmt_num(ops_tot))
with col3:
    st.metric(label=TEXT["avg_ticket"], value=fmt_brl(ticket))
with col4:
    st.metric(label=TEXT["institutions"], value=fmt_num(n_banks))
with col5:
    st.metric(label=TEXT["states"], value=fmt_num(n_uf))

# ============================================================
# ALERTS
# ============================================================
alertas = []
mom_last = agg["mom_last"]
if mom_last < CONFIG["mom_drop_sharp"]:
    alertas.append(("er", TEXT["alert_sharp_drop"].format(mom_last)))
elif mom_last < CONFIG["mom_drop_moderate"]:
    alertas.append(("wa", TEXT["alert_slowdown"].format(mom_last)))
elif mom_last > CONFIG["mom_growth_strong"]:
    alertas.append(("ok", TEXT["alert_strong_accel"].format(mom_last)))
elif mom_last > 0:
    alertas.append(("ok", TEXT["alert_stable_growth"].format(mom_last)))

if hhi_val > CONFIG["hhi_threshold_high"]:
    alertas.append(("er", TEXT["alert_high_concentration"].format(CONFIG["hhi_threshold_high"])))
elif hhi_val > CONFIG["hhi_threshold_moderate"]:
    alertas.append(("wa", TEXT["alert_moderate_concentration"].format(CONFIG["hhi_threshold_moderate"], CONFIG["hhi_threshold_high"])))
else:
    alertas.append(("ok", TEXT["alert_competitive"].format(CONFIG["hhi_threshold_moderate"])))

if gini_r > CONFIG["gini_threshold_high"]:
    alertas.append(("er", TEXT["alert_high_regional_inequality"].format(gini_r)))
elif gini_r > CONFIG["gini_threshold_moderate"]:
    alertas.append(("wa", TEXT["alert_regional_inequality"].format(gini_r)))

if alertas:
    st.markdown(f"### ⚡ {TEXT['smart_alerts']}")
    cols = st.columns(min(len(alertas), 3))
    for i, (cls, msg) in enumerate(alertas):
        with cols[i % 3]:
            st.markdown(f'<div class="al {cls}">{msg}</div>', unsafe_allow_html=True)

# ============================================================
# INSIGHTS
# ============================================================
st.markdown(f"### 🔍 {TEXT['automated_insights']}")

region, vol_leader, pct_leader = get_region_leader(reg_df)

# Check for "grande_area" column
if "grande_area" in dff.columns:
    area_df = dff.groupby("grande_area")["volume_operacoes"].sum().reset_index()
    lider_area = area_df.loc[area_df["volume_operacoes"].idxmax()] if not area_df.empty else None
else:
    lider_area = None

cresc_med = evol_g["mom"].dropna().mean() if not evol_g["mom"].dropna().empty else 0.0

insight_cols = st.columns(3)
with insight_cols[0]:
    st.markdown(f"""
    <div class="insight-card">
        <div class="insight-title">📍 {TEXT['regional_concentration']}</div>
        <div class="insight-text">{TEXT['insight_regional'].format(region, pct_leader)}</div>
        <div class="insight-value">{fmt_brl(vol_leader)}</div>
    </div>
    """, unsafe_allow_html=True)
with insight_cols[1]:
    if lider_area is not None:
        st.markdown(f"""
        <div class="insight-card">
            <div class="insight-title">🧬 {TEXT['leading_area']}</div>
            <div class="insight-text">{TEXT['insight_area'].format(lider_area['grande_area'])}</div>
            <div class="insight-value">{fmt_brl(lider_area['volume_operacoes'])}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="insight-card">
            <div class="insight-title">📈 {TEXT['trend']}</div>
            <div class="insight-text">{TEXT['insight_trend'].format(cresc_med)}</div>
        </div>
        """, unsafe_allow_html=True)
with insight_cols[2]:
    conc_text = TEXT['low_concentration'] if hhi_val < 1500 else (TEXT['moderate_concentration'] if hhi_val < 2500 else TEXT['high_concentration'])
    st.markdown(f"""
    <div class="insight-card">
        <div class="insight-title">⚖️ {TEXT['concentration_hhi']}</div>
        <div class="insight-text">{TEXT['insight_hhi'].format(hhi_val)}</div>
        <div class="insight-value">{conc_text} concentration</div>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# TABS WITH PREMIUM CHARTS (using cached chart functions)
# ============================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    f"📈 {TEXT['time_series']}",
    f"🏦 {TEXT['bank_concentration']}",
    f"🗺️ {TEXT['regional_analysis']}",
    f"🔬 {TEXT['advanced_analytics']}",
    f"📊 {TEXT['distribution']}"
])

# ---------- TAB 1: TIME SERIES ----------
with tab1:
    st.markdown(f"### 📈 {TEXT['program_evolution']}")

    col_ev1, col_ev2 = st.columns([2, 1])

    with col_ev1:
        fig1 = create_timeline_chart(dff, evol_g, CORES_GRAFICOS)
        st.plotly_chart(fig1, use_container_width=True)

    with col_ev2:
        fig2 = create_mom_chart(evol_g)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown(f"**🔥 {TEXT['heatmap_title']}**")
    fig3 = create_heatmap(dff)
    if fig3:
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("Heatmap unavailable for the current filters.")

# ---------- TAB 2: BANK CONCENTRATION ----------
with tab2:
    st.markdown(f"### 🏦 {TEXT['bank_concentration']}")

    top_n = st.slider(TEXT['top_n_institutions'], 5, 30, CONFIG["default_top_n"], key="top_n_bancos")

    fig4 = create_concentration_chart(dff, top_n)
    st.plotly_chart(fig4, use_container_width=True)

    fig5 = create_treemap(dff, top_n)
    st.plotly_chart(fig5, use_container_width=True)

    # Metrics
    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        st.metric("CR3", f"{cr3:.1f}%", help="Top 3 banks")
    with col_m2:
        st.metric("CR5", f"{cr5:.1f}%", help="Top 5 banks")
    with col_m3:
        gini_b = gini(b_agg["numero_operacoes"]) if not b_agg.empty else 0
        st.metric("Gini", f"{gini_b:.3f}", help="Bank inequality")

# ---------- TAB 3: REGIONAL ANALYSIS ----------
with tab3:
    st.markdown(f"### 🗺️ {TEXT['regional_distribution']}")

    fig_map, fig_bar, fig_donut = create_regional_charts(dff)
    st.plotly_chart(fig_map, use_container_width=True)
    col_r1, col_r2 = st.columns([2, 1])
    with col_r1:
        st.plotly_chart(fig_bar, use_container_width=True)
    with col_r2:
        st.plotly_chart(fig_donut, use_container_width=True)

# ---------- TAB 4: ADVANCED ANALYTICS ----------
with tab4:
    st.markdown(f"### 🔬 {TEXT['advanced_analytics_title']}")

    # Clustering
    fig_cluster = create_cluster_chart(dff)
    if fig_cluster:
        st.plotly_chart(fig_cluster, use_container_width=True)
    else:
        st.info("Clustering unavailable (insufficient data).")

    # Radar
    fig_radar = create_radar_chart(hhi_val, cr3, cr5, gini_r, ticket)
    st.plotly_chart(fig_radar, use_container_width=True)

    # Scatter
    fig_scatter = create_scatter_chart(dff)
    st.plotly_chart(fig_scatter, use_container_width=True)

# ---------- TAB 5: DISTRIBUTION ----------
with tab5:
    st.markdown(f"### 📊 {TEXT['pareto_title']}")

    fig_pareto, p80 = create_pareto_chart(dff)
    st.plotly_chart(fig_pareto, use_container_width=True)

    st.markdown(f"""
    <div class="insight-card">
        <div class="insight-title">📐 {TEXT['pareto_interpretation']}</div>
        <div class="insight-text"><b>{p80}</b> {TEXT['pareto_text']}</div>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# EXPORT
# ============================================================
st.markdown("---")
st.markdown(f"### 📥 {TEXT['export_section']}")

col_exp1, col_exp2 = st.columns(2)
with col_exp1:
    csv_data = dff.to_csv(index=False).encode("utf-8")
    st.download_button(
        f"📄 {TEXT['csv_download']}",
        csv_data,
        file_name=f"debt_settlement_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )
with col_exp2:
    relatorio_txt = f"""DEBT SETTLEMENT BRAZIL REPORT
Volume: {fmt_brl(vol_tot)}
Contracts: {fmt_num(ops_tot)}
Avg Ticket: {fmt_brl(ticket)}
HHI: {hhi_val:.0f}
Regional Gini: {gini_r:.3f}
Source: Central Bank of Brazil (SCR)
"""
    st.download_button(
        f"📝 {TEXT['report_download']}",
        relatorio_txt,
        file_name=f"report_{datetime.now().strftime('%Y%m%d')}.txt",
        mime="text/plain"
    )

# ============================================================
# FOOTER
# ============================================================
st.markdown(f"""
<div class="footer">
    🏦 {TEXT['footer_text']}<br>
    {TEXT['footer_source']}: <a href="https://www.bcb.gov.br/estatisticas/scr" target="_blank" style="color:{COLORS['P1']};">Central Bank of Brazil (SCR)</a>
</div>
""", unsafe_allow_html=True)
