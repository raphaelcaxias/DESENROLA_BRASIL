"""
Debt Settlement Brazil – Intelligence Platform v2.0 (Final)
Enterprise-grade dashboard with synthetic data fallback, Brazil map, advanced analytics,
and multi-format exports. Fully internationalized (PT/EN) with theme toggle.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
import re
import warnings
import logging
from datetime import datetime
from typing import Optional, Tuple, List, Dict, Any, Union
import base64
import json
import io
import hashlib
from dataclasses import dataclass

# ============================================================
# CONFIGURATION & CONSTANTS
# ============================================================
@dataclass
class AppConfig:
    """Centralized application configuration."""
    # Thresholds
    hhi_threshold_high: int = 2500
    hhi_threshold_moderate: int = 1500
    gini_threshold_high: float = 0.7
    gini_threshold_moderate: float = 0.5
    mom_drop_sharp: float = -15.0
    mom_drop_moderate: float = -5.0
    mom_growth_strong: float = 20.0
    
    # UI Defaults
    default_top_n: int = 15
    default_language: str = "pt"
    default_theme: str = "dark"
    
    # Analytics
    min_cluster_samples: int = 3
    min_ops_for_cluster: int = 100
    forecast_periods: int = 3
    outlier_contamination: float = 0.05
    
    # GeoJSON URL (official Brazil states)
    geojson_url: str = "https://raw.githubusercontent.com/codeforamerica/click_that_hood/master/public/data/brazil-states.geojson"
    
    # Cache TTL (seconds)
    cache_ttl: int = 3600

CONFIG = AppConfig()

# ============================================================
# INTERNATIONALIZATION (i18n)
# ============================================================
TEXTS = {
    "pt": {
        # App
        "app_title": "Desenrola Brasil",
        "app_subtitle": "Plataforma de Inteligência",
        "hero_title": "Desenrola Brasil",
        "hero_subtitle": "Inteligência analítica para renegociação de dívidas · Fonte: Banco Central (SCR)",
        
        # KPIs (labels for st.metric)
        "volume": "Volume Renegociado",
        "contracts": "Total de Contratos",
        "avg_ticket": "Ticket Médio",
        "institutions": "Instituições",
        "states": "Estados",
        "vs_prev_month": "vs mês anterior",
        
        # Alerts
        "smart_alerts": "Alertas Inteligentes",
        "alert_sharp_drop": "🔴 Queda Abrupta – volume caiu {:.1f}%",
        "alert_slowdown": "🟡 Desaceleração – queda de {:.1f}%",
        "alert_strong_accel": "🟢 Aceleração Forte – +{:.1f}%",
        "alert_stable_growth": "🟢 Crescimento Estável – +{:.1f}%",
        "alert_high_concentration": "🔴 Concentração Elevada – HHI > {}",
        "alert_moderate_concentration": "🟡 Concentração Moderada – HHI {}-{}",
        "alert_competitive": "🟢 Mercado Competitivo – HHI < {}",
        "alert_high_regional_inequality": "🔴 Alta Desigualdade Regional – Gini = {:.2f}",
        "alert_regional_inequality": "🟡 Desigualdade Regional – Gini = {:.2f}",
        "alert_outliers_detected": "🟠 {} outliers detectados nas operações",
        "alert_seasonal_pattern": "🔵 Padrão sazonal identificado",
        
        # Insights
        "automated_insights": "Insights Automáticos",
        "regional_concentration": "Concentração Regional",
        "leading_area": "Área Líder",
        "trend": "Tendência",
        "concentration_hhi": "Concentração (HHI)",
        "low_concentration": "Baixa",
        "moderate_concentration": "Moderada",
        "high_concentration": "Alta",
        "insight_regional": "A região <b>{}</b> concentra <b>{:.1f}%</b> do volume total.",
        "insight_area": "<b>{}</b> lidera os investimentos.",
        "insight_trend": "Crescimento médio mensal de <b>{:+.1f}%</b>.",
        "insight_hhi": "Índice Herfindahl-Hirschman: <b>{:.0f}</b>",
        "insight_correlation": "Correlação Ticket × Volume: <b>{:.2f}</b>",
        "insight_top_banks": "Top 3 bancos controlam <b>{:.1f}%</b> do mercado.",
        
        # Tabs
        "time_series": "Evolução Temporal",
        "bank_concentration": "Concentração Bancária",
        "regional_analysis": "Análise Regional",
        "advanced_analytics": "Análise Avançada",
        "distribution": "Distribuição",
        
        # Tab 1
        "program_evolution": "Evolução do Programa",
        "monthly_growth": "Crescimento Mensal (MoM)",
        "heatmap_title": "Mapa de Calor – Volume por Faixa",
        "ren_intensity": "Intensidade de Renegociação",
        "seasonal_decomposition": "Decomposição Sazonal",
        
        # Tab 2
        "top_n_institutions": "Top N Instituições",
        "treemap_title": "Treemap – Distribuição por Segmento",
        
        # Tab 3
        "regional_distribution": "Distribuição Regional",
        "volume_and_ticket_by_region": "Volume e Ticket Médio por Região",
        "regional_share": "Participação Regional",
        
        # Tab 4
        "advanced_analytics_title": "Análises Avançadas",
        "clustering_title": "Clusterização (Operações × Ticket)",
        "concentration_radar": "Radar de Concentração",
        "scatter_title": "Ticket Médio vs Market Share",
        "outlier_detection": "Detecção de Outliers",
        "correlation_matrix": "Matriz de Correlação",
        
        # Tab 5
        "pareto_title": "Curva de Pareto",
        "pareto_interpretation": "Interpretação",
        "pareto_text": "instituições concentram 80% do volume total renegociado.",
        
        # Export
        "export_section": "Exportar Dados",
        "csv_download": "CSV (dados filtrados)",
        "excel_download": "Excel (multi-abas)",
        "report_download": "Relatório TXT",
        "pdf_download": "PDF Completo",
        
        # Footer
        "footer_text": "Desenrola Brasil · Inteligência Financeira",
        "footer_source": "Fonte: Banco Central do Brasil (SCR)",
        
        # Sidebar
        "data_source": "Fonte de Dados",
        "filters": "Filtros",
        "data_quality": "Qualidade dos Dados",
        "reset_filters": "Resetar Filtros",
        "upload_csv": "Carregar CSV",
        "use_demo_data": "Usar dados de demonstração",
        
        # Status messages
        "loading_data": "🔄 Carregando dados...",
        "processing": "Processando...",
        "no_data": "⚠️ Nenhum dado encontrado com os filtros selecionados.",
        "no_data_suggestion": "💡 Tente expandir o período, selecionar mais categorias ou resetar os filtros.",
        "map_unavailable": "🗺️ Mapa indisponível (falha ao carregar GeoJSON).",
        "heatmap_unavailable": "🔥 Heatmap indisponível para os filtros atuais.",
        "clustering_unavailable": "🔬 Dados insuficientes para clusterização.",
        "outliers_unavailable": "⚠️ Detecção de outliers indisponível.",
    },
    "en": {
        # App
        "app_title": "Debt Settlement Brazil",
        "app_subtitle": "Intelligence Platform",
        "hero_title": "Debt Settlement Brazil",
        "hero_subtitle": "Analytical intelligence for debt renegotiation · Source: Central Bank (SCR)",
        
        # KPIs
        "volume": "Renegotiated Volume",
        "contracts": "Total Contracts",
        "avg_ticket": "Average Ticket",
        "institutions": "Institutions",
        "states": "States",
        "vs_prev_month": "vs previous month",
        
        # Alerts
        "smart_alerts": "Smart Alerts",
        "alert_sharp_drop": "🔴 Sharp Drop – volume fell {:.1f}%",
        "alert_slowdown": "🟡 Slowdown – drop of {:.1f}%",
        "alert_strong_accel": "🟢 Strong Acceleration – +{:.1f}%",
        "alert_stable_growth": "🟢 Stable Growth – +{:.1f}%",
        "alert_high_concentration": "🔴 High Concentration – HHI > {}",
        "alert_moderate_concentration": "🟡 Moderate Concentration – HHI {}-{}",
        "alert_competitive": "🟢 Competitive Market – HHI < {}",
        "alert_high_regional_inequality": "🔴 High Regional Inequality – Gini = {:.2f}",
        "alert_regional_inequality": "🟡 Regional Inequality – Gini = {:.2f}",
        "alert_outliers_detected": "🟠 {} outliers detected in operations",
        "alert_seasonal_pattern": "🔵 Seasonal pattern identified",
        
        # Insights
        "automated_insights": "Automated Insights",
        "regional_concentration": "Regional Concentration",
        "leading_area": "Leading Area",
        "trend": "Trend",
        "concentration_hhi": "Concentration (HHI)",
        "low_concentration": "Low",
        "moderate_concentration": "Moderate",
        "high_concentration": "High",
        "insight_regional": "The <b>{}</b> region concentrates <b>{:.1f}%</b> of total volume.",
        "insight_area": "<b>{}</b> leads investments.",
        "insight_trend": "Average monthly growth of <b>{:+.1f}%</b>.",
        "insight_hhi": "Herfindahl-Hirschman Index: <b>{:.0f}</b>",
        "insight_correlation": "Ticket × Volume Correlation: <b>{:.2f}</b>",
        "insight_top_banks": "Top 3 banks control <b>{:.1f}%</b> of the market.",
        
        # Tabs
        "time_series": "Time Series",
        "bank_concentration": "Bank Concentration",
        "regional_analysis": "Regional Analysis",
        "advanced_analytics": "Advanced Analytics",
        "distribution": "Distribution",
        
        # Tab 1
        "program_evolution": "Program Evolution",
        "monthly_growth": "Monthly Growth (MoM)",
        "heatmap_title": "Heatmap – Volume by Tranche",
        "ren_intensity": "Renegotiation Intensity",
        "seasonal_decomposition": "Seasonal Decomposition",
        
        # Tab 2
        "top_n_institutions": "Top N Institutions",
        "treemap_title": "Treemap – Distribution by Segment",
        
        # Tab 3
        "regional_distribution": "Regional Distribution",
        "volume_and_ticket_by_region": "Volume and Average Ticket by Region",
        "regional_share": "Regional Share",
        
        # Tab 4
        "advanced_analytics_title": "Advanced Analytics",
        "clustering_title": "Clustering (Operations × Ticket)",
        "concentration_radar": "Concentration Radar",
        "scatter_title": "Average Ticket vs Market Share",
        "outlier_detection": "Outlier Detection",
        "correlation_matrix": "Correlation Matrix",
        
        # Tab 5
        "pareto_title": "Pareto Curve",
        "pareto_interpretation": "Interpretation",
        "pareto_text": "institutions concentrate 80% of the total renegotiated volume.",
        
        # Export
        "export_section": "Export Data",
        "csv_download": "CSV (filtered data)",
        "excel_download": "Excel (multi-sheet)",
        "report_download": "TXT Report",
        "pdf_download": "Full PDF",
        
        # Footer
        "footer_text": "Debt Settlement Brazil · Financial Intelligence",
        "footer_source": "Source: Central Bank of Brazil (SCR)",
        
        # Sidebar
        "data_source": "Data Source",
        "filters": "Filters",
        "data_quality": "Data Quality",
        "reset_filters": "Reset Filters",
        "upload_csv": "Upload CSV",
        "use_demo_data": "Use demo data",
        
        # Status messages
        "loading_data": "🔄 Loading data...",
        "processing": "Processing...",
        "no_data": "⚠️ No data matches the selected filters.",
        "no_data_suggestion": "💡 Try expanding the period, selecting more categories, or resetting filters.",
        "map_unavailable": "🗺️ Map unavailable (failed to load GeoJSON).",
        "heatmap_unavailable": "🔥 Heatmap unavailable for current filters.",
        "clustering_unavailable": "🔬 Insufficient data for clustering.",
        "outliers_unavailable": "⚠️ Outlier detection unavailable.",
    }
}

# ============================================================
# LOGGING
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("desenrola_app")

# ============================================================
# STREAMLIT PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Desenrola Brasil | Intelligence Platform",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Session state initialization
if "tema" not in st.session_state:
    st.session_state.tema = CONFIG.default_theme
if "lang" not in st.session_state:
    st.session_state.lang = CONFIG.default_language
if "demo_mode" not in st.session_state:
    st.session_state.demo_mode = False

T = st.session_state.tema
LANG = st.session_state.lang
TEXT = TEXTS[LANG]

# ============================================================
# COLOR PALETTE
# ============================================================
class ColorPalette:
    """Theme-aware color palette."""

    def __init__(self, theme: str):
        self.theme = theme

    @property
    def colors(self) -> Dict[str, str]:
        if self.theme == "light":
            return {
                "BG": "#F7F9FC", "CARD": "#FFFFFF", "CARD_GLASS": "rgba(255,255,255,0.85)",
                "TXT": "#1A2B4C", "TXT2": "#5A6E8A", "BORDA": "#E2E8F0",
                "BORDA_GLOW": "rgba(0,168,107,0.2)", "ACCENT_GLOW": "rgba(0,102,204,0.2)",
                "P1": "#00A86B", "P2": "#0066CC", "P3": "#52B788", "ACCENT": "#0066CC",
                "VERDE": "#00A86B", "VERM": "#DC2626", "AMBER": "#F59E0B",
                "AZUL": "#3B82F6", "ROXO": "#8B5CF6", "CINZA": "#6B7280",
                "TPLOTE": "plotly_white", "GRID": "rgba(0,0,0,0.05)",
                "GLOW_P1": "rgba(0,168,107,0.3)",
                "CHART_COLORS": ["#00A86B", "#F59E0B", "#DC2626", "#3B82F6", "#8B5CF6", "#0066CC", "#52B788", "#6B7280"]
            }
        else:  # dark
            return {
                "BG": "#0A0F1C", "CARD": "#111827", "CARD_GLASS": "rgba(17,24,39,0.85)",
                "TXT": "#F1F5F9", "TXT2": "#94A3B8", "BORDA": "#1F2937",
                "BORDA_GLOW": "rgba(56,189,248,0.2)", "ACCENT_GLOW": "rgba(96,165,250,0.2)",
                "P1": "#3FB68C", "P2": "#3B82F6", "P3": "#10B981", "ACCENT": "#60A5FA",
                "VERDE": "#34D399", "VERM": "#F87171", "AMBER": "#FBBF24",
                "AZUL": "#60A5FA", "ROXO": "#A78BFA", "CINZA": "#6B7280",
                "TPLOTE": "plotly_dark", "GRID": "rgba(255,255,255,0.05)",
                "GLOW_P1": "rgba(63,182,140,0.3)",
                "CHART_COLORS": ["#3FB68C", "#FBBF24", "#F87171", "#60A5FA", "#A78BFA", "#3B82F6", "#34D399", "#6B7280"]
            }

COLORS = ColorPalette(T).colors
CHART_COLORS = COLORS["CHART_COLORS"]

# ============================================================
# CSS STYLES (minimal, as st.metric and built-in components handle most styling)
# ============================================================
CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Playfair+Display:wght@400;500;600;700&display=swap');

* {{ margin: 0; padding: 0; box-sizing: border-box; }}

.stApp {{
    background: {COLORS["BG"]};
    font-family: 'Inter', sans-serif;
    color: {COLORS["TXT"]};
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
    font-weight: 600;
}}

/* ===== METRICS (st.metric) ===== */
[data-testid="stMetric"] {{
    background: {COLORS["CARD_GLASS"]};
    backdrop-filter: blur(12px);
    border: 1px solid {COLORS["BORDA"]};
    border-radius: 16px;
    padding: 1rem 1.2rem;
    transition: all 0.3s;
    box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
}}
[data-testid="stMetric"]:hover {{
    border-color: {COLORS["P1"]};
    transform: translateY(-4px);
    box-shadow: 0 8px 25px rgba(0,0,0,0.15);
}}
[data-testid="stMetricLabel"] {{
    color: {COLORS["TXT2"]};
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}}
[data-testid="stMetricValue"] {{
    color: {COLORS["TXT"]};
    font-weight: 700;
    font-size: 1.5rem;
}}
[data-testid="stMetricDelta"] {{
    font-size: 0.75rem;
    font-weight: 500;
}}

/* ===== INSIGHT CARDS ===== */
.insight-grid {{
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1rem;
    margin: 1rem 0;
}}

@media (max-width: 1000px) {{ .insight-grid {{ grid-template-columns: 1fr; }} }}

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
    line-height: 1.5;
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
    font-weight: 500;
}}
.al.er {{ background: rgba(220,38,38,0.1); border-left: 3px solid {COLORS["VERM"]}; color: {COLORS["VERM"]}; }}
.al.wa {{ background: rgba(245,158,11,0.1); border-left: 3px solid {COLORS["AMBER"]}; color: {COLORS["AMBER"]}; }}
.al.ok {{ background: rgba(0,168,107,0.1); border-left: 3px solid {COLORS["VERDE"]}; color: {COLORS["VERDE"]}; }}
.al.in {{ background: rgba(59,130,246,0.1); border-left: 3px solid {COLORS["AZUL"]}; color: {COLORS["AZUL"]}; }}

/* ===== TABS ===== */
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
.stTabs [data-baseweb="tab"]:hover {{
    border-color: {COLORS["P1"]};
    color: {COLORS["TXT"]};
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

section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stMultiSelect label {{
    font-size: 0.75rem;
    font-weight: 600;
    color: {COLORS["TXT2"]};
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

/* ===== DATA SOURCE CARD ===== */
.source-card {{
    background: {COLORS["CARD_GLASS"]};
    border-radius: 12px;
    padding: 0.8rem;
    border: 1px solid {COLORS["BORDA"]};
    margin-bottom: 1rem;
}}

.source-badge {{
    display: inline-block;
    background: {COLORS["P1"]};
    color: white;
    padding: 0.15rem 0.5rem;
    border-radius: 20px;
    font-size: 0.6rem;
    font-weight: 600;
    margin-right: 0.5rem;
}}
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)

# ============================================================
# UTILITY FUNCTIONS
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
    if abs(v) >= 1e9:
        return f"R$ {v/1e9:.2f}B".replace(".", ",")
    if abs(v) >= 1e6:
        return f"R$ {v/1e6:.1f}M".replace(".", ",")
    if abs(v) >= 1e3:
        return f"R$ {v/1e3:.1f}K".replace(".", ",")
    return f"R$ {v:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

def fmt_num(v: float, decimals: int = 0) -> str:
    """Format number with thousand separators."""
    if pd.isna(v):
        return "0"
    if decimals > 0:
        return f"{v:,.{decimals}f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return f"{int(v):,}".replace(",", ".")

def classify_bank(name: str) -> str:
    """
    Classify financial institution into a segment based on its name.

    Parameters:
        name (str): Bank name.

    Returns:
        str: Segment label (Digital, Tradicional, Investimento, Cooperativa, Fintech, Outros).
    """
    n = re.sub(r'\s*-\s*PRUDENCIAL$', '', str(name).upper().strip())
    
    rules = [
        (["NUBANK", "INTER", "C6", "NEON", "ORIGINAL", "PAN", "NEXT", "WILL"], "Digital"),
        (["ITAU", "BRADESCO", "SANTANDER", "CAIXA", "BANCO DO BRASIL", "BB", "SAFRA"], "Tradicional"),
        (["BTG", "XP", "MODAL", "GENIAL", "RICO"], "Investimento"),
        (["SICOOB", "SICREDI", "CRESOL", "UNICRED"], "Cooperativa"),
        (["MERCADO PAGO", "PICPAY", "PAGBANK", "PAGSEGURO"], "Fintech"),
    ]
    
    for keywords, category in rules:
        if any(k in n for k in keywords):
            return category
    return "Outros"

def classify_region(uf: str) -> str:
    """
    Map Brazilian state code (UF) to its corresponding region.

    Parameters:
        uf (str): Two-letter state code.

    Returns:
        str: Region name (Norte, Nordeste, Centro-Oeste, Sudeste, Sul, or Não Identificado).
    """
    mapping = {
        "Norte": ["AC", "AM", "AP", "PA", "RO", "RR", "TO"],
        "Nordeste": ["AL", "BA", "CE", "MA", "PB", "PE", "PI", "RN", "SE"],
        "Centro-Oeste": ["DF", "GO", "MS", "MT"],
        "Sudeste": ["ES", "MG", "RJ", "SP"],
        "Sul": ["PR", "RS", "SC"]
    }
    uf_upper = str(uf).upper().strip()
    for region, states in mapping.items():
        if uf_upper in states:
            return region
    return "Não Identificado"

def hhi(series: pd.Series) -> float:
    """
    Calculate Herfindahl-Hirschman Index (HHI) for a distribution.

    Parameters:
        series (pd.Series): Values representing market shares or counts.

    Returns:
        float: HHI value (0 to 10000).
    """
    total = series.sum()
    if total == 0 or pd.isna(total):
        return 0.0
    shares = series / total
    return float((shares ** 2).sum() * 10000)

def gini(series: pd.Series) -> float:
    """
    Calculate Gini coefficient for a distribution.

    Parameters:
        series (pd.Series): Values to compute inequality.

    Returns:
        float: Gini coefficient (0 to 1).
    """
    arr = np.sort(series.dropna().values)
    n = len(arr)
    if n == 0 or arr.sum() == 0:
        return 0.0
    index = np.arange(1, n + 1)
    return float((2 * np.sum(index * arr) / (n * arr.sum())) - (n + 1) / n)

def concentration_ratio(series: pd.Series, n: int) -> float:
    """
    Calculate concentration ratio (CRn): sum of top n values as percentage of total.

    Parameters:
        series (pd.Series): Values.
        n (int): Number of top items to sum.

    Returns:
        float: CRn percentage.
    """
    total = series.sum()
    if total == 0 or pd.isna(total):
        return 0.0
    top_n = series.nlargest(n).sum()
    return float(top_n / total * 100)

def base_layout(fig: go.Figure, h: int = 440, leg: bool = True) -> go.Figure:
    """
    Apply standard layout styling to Plotly figures.

    Parameters:
        fig (go.Figure): Figure to style.
        h (int): Height in pixels.
        leg (bool): Whether to show legend.

    Returns:
        go.Figure: Styled figure.
    """
    fig.update_layout(
        template=COLORS["TPLOTE"], height=h,
        margin=dict(l=50, r=40, t=55, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=COLORS["TXT"], family="Inter", size=12),
        hovermode="x unified",
        showlegend=leg,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        transition=dict(duration=300, easing="cubic-in-out"),
        hoverlabel=dict(bgcolor=COLORS["CARD"], font_size=12, font_family="Inter")
    )
    fig.update_xaxes(showgrid=False, color=COLORS["TXT"], linecolor=COLORS["BORDA"], zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor=COLORS["GRID"], color=COLORS["TXT"], zeroline=False)
    return fig

def safe_divide(a: float, b: float, default: float = 0.0) -> float:
    """Safely divide two numbers, returning default if denominator is zero or NaN."""
    return a / b if b != 0 and not pd.isna(b) else default

# ============================================================
# DATA GENERATION & LOADING
# ============================================================
class DataGenerator:
    """Generate realistic synthetic data for demonstration purposes."""

    @staticmethod
    @st.cache_data
    def generate_sample_data(n_records: int = 5000) -> pd.DataFrame:
        """
        Generate synthetic Desenrola data with realistic distributions.

        Parameters:
            n_records (int): Number of records to generate.

        Returns:
            pd.DataFrame: Synthetic dataset.
        """
        np.random.seed(42)
        
        dates = pd.date_range("2023-07-01", "2024-12-01", freq="MS")
        ufs = ["SP", "RJ", "MG", "RS", "PR", "BA", "PE", "CE", "PA", "MA", 
               "SC", "DF", "GO", "MT", "MS", "AM", "ES", "PB", "RN", "AL"]
        bancos = [
            "ITAU UNIBANCO - PRUDENCIAL", "BANCO BRADESCO - PRUDENCIAL", 
            "NUBANK", "CAIXA ECONOMICA FEDERAL", "BANCO DO BRASIL", 
            "SANTANDER", "BANCO INTER", "C6 BANK", "SICOOB", "BTG PACTUAL",
            "BANCO PAN", "BANCO ORIGINAL", "XP INVESTIMENTOS", "BANCO SAFRA",
            "SICREDI", "MERCADO PAGO", "PICPAY", "PAGBANK", "BANCO NEXT", "BANCO NEON"
        ]
        tipos = ["Faixa 1", "Faixa 2", "Faixa 3", "Faixa 4"]
        areas = ["Crédito Pessoal", "Cartão de Crédito", "Cheque Especial", "Financiamento"]
        
        uf_weights = np.array([0.25, 0.15, 0.10, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.03,
                               0.03, 0.02, 0.02, 0.02, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01])
        banco_weights = np.array([0.18, 0.15, 0.12, 0.10, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03,
                                  0.02, 0.02, 0.02, 0.02, 0.01, 0.01, 0.01, 0.005, 0.003, 0.002])
        uf_weights = uf_weights / uf_weights.sum()
        banco_weights = banco_weights / banco_weights.sum()
        
        data = []
        for _ in range(n_records):
            base_volume = np.random.lognormal(mean=14, sigma=1.2)
            ops = max(1, int(base_volume / np.random.lognormal(mean=3, sigma=0.5)))
            volume = base_volume * np.random.uniform(0.8, 1.2)
            date_idx = np.random.choice(len(dates), p=np.linspace(0.02, 0.08, len(dates)))
            date = dates[date_idx]
            trend_factor = 1 + (date_idx / len(dates)) * 0.5
            
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


class DataLoader:
    """Handle data loading, validation, and preprocessing."""

    REQUIRED_COLUMNS = ["data_base", "unidade_federacao", "nome_conglomerado_financeiro", 
                        "numero_operacoes", "volume_operacoes"]

    @staticmethod
    def validate_schema(df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Check if dataframe contains all required columns.

        Returns:
            Tuple[bool, List[str]]: (is_valid, list_of_missing_columns)
        """
        missing = [col for col in DataLoader.REQUIRED_COLUMNS if col not in df.columns]
        return len(missing) == 0, missing

    @staticmethod
    def clean_numeric(series: pd.Series) -> pd.Series:
        """
        Clean and convert a series to numeric, handling Brazilian decimal and thousand separators.

        Parameters:
            series (pd.Series): Series containing numeric strings.

        Returns:
            pd.Series: Cleaned numeric series.
        """
        return pd.to_numeric(
            series.astype(str)
            .str.replace(".", "", regex=False)
            .str.replace(",", ".", regex=False)
            .str.strip(),
            errors="coerce"
        )

    @staticmethod
    @st.cache_data(show_spinner="🔄 Processando dados...")
    def process_raw_data(df: pd.DataFrame) -> pd.DataFrame:
        """
        Process and enrich raw dataframe: clean columns, derive features, drop invalid rows.

        Parameters:
            df (pd.DataFrame): Raw input dataframe.

        Returns:
            pd.DataFrame: Processed dataframe.
        """
        df = df.copy()
        df.columns = df.columns.str.lower().str.strip()
        
        for col in ["numero_operacoes", "volume_operacoes"]:
            if col in df.columns:
                df[col] = DataLoader.clean_numeric(df[col])
        
        if "data_base" in df.columns:
            df["data_base"] = pd.to_datetime(df["data_base"].astype(str), format="%Y%m", errors="coerce")
        
        if "nome_conglomerado_financeiro" in df.columns:
            df["tipo_banco"] = df["nome_conglomerado_financeiro"].apply(classify_bank)
        if "unidade_federacao" in df.columns:
            df["regiao"] = df["unidade_federacao"].apply(classify_region)
            df["uf_codigo"] = df["unidade_federacao"].str.upper().str.strip()
        
        if "volume_operacoes" in df.columns and "numero_operacoes" in df.columns:
            df["ticket_medio"] = df["volume_operacoes"] / df["numero_operacoes"].replace(0, np.nan)
        
        df = df.dropna(subset=["volume_operacoes", "numero_operacoes", "data_base"])
        df = df[(df["numero_operacoes"] > 0) & (df["volume_operacoes"] > 0)]
        
        logger.info(f"Processed data: {len(df)} valid records")
        return df

    @staticmethod
    def load_from_file(uploaded_file) -> Optional[pd.DataFrame]:
        """
        Load data from an uploaded CSV or Excel file, trying multiple encodings.

        Parameters:
            uploaded_file (UploadedFile): Streamlit uploaded file object.

        Returns:
            Optional[pd.DataFrame]: Loaded dataframe or None if failed.
        """
        try:
            file_name = uploaded_file.name.lower()
            if file_name.endswith('.csv'):
                for enc in ["utf-8", "latin1", "cp1252", "iso-8859-1"]:
                    try:
                        uploaded_file.seek(0)
                        df = pd.read_csv(uploaded_file, sep=None, encoding=enc, 
                                        engine='python', low_memory=False)
                        return df
                    except (UnicodeDecodeError, pd.errors.ParserError):
                        continue
            elif file_name.endswith(('.xls', '.xlsx')):
                return pd.read_excel(uploaded_file)
        except Exception as e:
            logger.error(f"Failed to load file: {e}")
            st.error(f"❌ Erro ao carregar arquivo: {e}")
        return None


# ============================================================
# GEOJSON CACHE
# ============================================================
@st.cache_data(ttl=86400)
def get_brazil_geojson() -> Optional[Dict]:
    """Download and cache official Brazil states GeoJSON."""
    try:
        import urllib.request
        response = urllib.request.urlopen(CONFIG.geojson_url, timeout=10)
        geojson = json.loads(response.read())
        logger.info("GeoJSON loaded successfully")
        return geojson
    except Exception as e:
        logger.warning(f"Failed to load GeoJSON: {e}")
        return None


# ============================================================
# ANALYTICS ENGINE
# ============================================================
class AnalyticsEngine:
    """Core analytics computations: KPIs, outlier detection, correlations, decomposition, forecasting."""

    @staticmethod
    @st.cache_data
    def compute_kpis(df: pd.DataFrame) -> Dict[str, Any]:
        """Compute all key performance indicators from filtered dataframe."""
        vol_tot = df["volume_operacoes"].sum()
        ops_tot = df["numero_operacoes"].sum()
        ticket = safe_divide(vol_tot, ops_tot)
        
        evol_g = df.groupby("data_base")["volume_operacoes"].sum().reset_index().sort_values("data_base")
        evol_g["mom"] = evol_g["volume_operacoes"].pct_change() * 100
        evol_g["yoy"] = evol_g["volume_operacoes"].pct_change(periods=12) * 100
        evol_g["ma3"] = evol_g["volume_operacoes"].rolling(3, min_periods=1).mean()
        
        mom_last = evol_g["mom"].dropna().iloc[-1] if not evol_g["mom"].dropna().empty else 0.0
        yoy_last = evol_g["yoy"].dropna().iloc[-1] if not evol_g["yoy"].dropna().empty else 0.0
        
        b_agg = df.groupby("nome_conglomerado_financeiro")["volume_operacoes"].sum()
        hhi_val = hhi(b_agg)
        cr3 = concentration_ratio(b_agg, 3)
        cr5 = concentration_ratio(b_agg, 5)
        
        reg_v = df.groupby("regiao")["volume_operacoes"].sum()
        gini_r = gini(reg_v)
        
        return {
            "vol_tot": vol_tot, "ops_tot": ops_tot, "ticket": ticket,
            "evol_g": evol_g, "mom_last": mom_last, "yoy_last": yoy_last,
            "hhi": hhi_val, "cr3": cr3, "cr5": cr5, "gini_r": gini_r,
            "n_banks": df["nome_conglomerado_financeiro"].nunique(),
            "n_uf": df["unidade_federacao"].nunique(),
            "reg_df": reg_v.reset_index(),
            "b_agg": b_agg.reset_index(),
            "avg_ticket": df["ticket_medio"].mean() if "ticket_medio" in df.columns else ticket,
        }

    @staticmethod
    @st.cache_data
    def detect_outliers(df: pd.DataFrame, column: str = "volume_operacoes") -> Tuple[pd.DataFrame, int]:
        """
        Detect outliers using Isolation Forest and IQR methods.

        Returns:
            Tuple[pd.DataFrame, int]: DataFrame with outlier flags and count of outliers.
        """
        try:
            features = df[[column, "numero_operacoes"]].fillna(0)
            iso_forest = IsolationForest(
                contamination=CONFIG.outlier_contamination, 
                random_state=42
            )
            df = df.copy()
            df["outlier_score"] = iso_forest.fit_predict(features)
            df["is_outlier"] = df["outlier_score"] == -1
            
            q1 = df[column].quantile(0.25)
            q3 = df[column].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            df["outlier_iqr"] = (df[column] < lower_bound) | (df[column] > upper_bound)
            df["is_outlier"] = df["is_outlier"] | df["outlier_iqr"]
            n_outliers = df["is_outlier"].sum()
            return df, int(n_outliers)
        except Exception as e:
            logger.warning(f"Outlier detection failed: {e}")
            df = df.copy()
            df["is_outlier"] = False
            return df, 0

    @staticmethod
    @st.cache_data
    def compute_correlations(df: pd.DataFrame) -> pd.DataFrame:
        """Compute Spearman correlation matrix for numeric columns."""
        numeric_cols = ["volume_operacoes", "numero_operacoes", "ticket_medio"]
        available_cols = [c for c in numeric_cols if c in df.columns]
        if len(available_cols) < 2:
            return pd.DataFrame()
        return df[available_cols].corr(method="spearman")

    @staticmethod
    @st.cache_data
    def seasonal_decomposition(evol_g: pd.DataFrame) -> Optional[Dict]:
        """Decompose time series into trend, seasonal, and residual components."""
        try:
            series = evol_g.set_index("data_base")["volume_operacoes"]
            series = series.asfreq("MS")
            if len(series) < 12:
                return None
            result = seasonal_decompose(series, model="additive", period=12)
            return {
                "trend": result.trend,
                "seasonal": result.seasonal,
                "resid": result.resid,
                "observed": result.observed
            }
        except Exception as e:
            logger.warning(f"Seasonal decomposition failed: {e}")
            return None

    @staticmethod
    @st.cache_data
    def forecast_series(evol_g: pd.DataFrame, periods: int = 3) -> Optional[Dict]:
        """Forecast future values using Exponential Smoothing with trend."""
        try:
            serie = evol_g["volume_operacoes"]
            if len(serie) < 4:
                return None
            hw = ExponentialSmoothing(
                serie.values, trend="add", seasonal=None,
                initialization_method="estimated"
            ).fit()
            prev = hw.forecast(periods)
            sigma = float(np.std(hw.resid))
            dt_fut = pd.date_range(evol_g["data_base"].max(), periods=periods+1, freq="MS")[1:]
            return {
                "forecast": prev,
                "dates": dt_fut,
                "lower": prev - 1.96 * sigma,
                "upper": prev + 1.96 * sigma,
                "sigma": sigma
            }
        except Exception as e:
            logger.warning(f"Forecast failed: {e}")
            return None


# ============================================================
# CHART FACTORY
# ============================================================
class ChartFactory:
    """Factory class to create all Plotly charts with cached results."""

    @staticmethod
    @st.cache_data
    def create_timeline_chart(df: pd.DataFrame, evol_g: pd.DataFrame) -> go.Figure:
        """Create timeline chart with tranche breakdown and forecast."""
        fig = go.Figure()
        tipos = sorted(df["tipo_desenrola"].unique())
        for i, tp in enumerate(tipos):
            g = df[df["tipo_desenrola"] == tp].groupby("data_base")["volume_operacoes"].sum().reset_index()
            cor = CHART_COLORS[i % len(CHART_COLORS)]
            fig.add_trace(go.Scatter(
                x=g["data_base"], y=g["volume_operacoes"],
                mode="lines+markers", name=f"Faixa {tp}",
                line=dict(color=cor, width=2.5),
                marker=dict(size=6, color=cor, line=dict(color="white", width=1)),
                hovertemplate=f"<b>Faixa {tp}</b><br>%{{x|%b/%Y}}<br>R$ %{{y:,.0f}}<extra></extra>"
            ))
        forecast = AnalyticsEngine.forecast_series(evol_g, CONFIG.forecast_periods)
        if forecast:
            xband = list(forecast["dates"]) + list(forecast["dates"][::-1])
            yband = list(forecast["upper"]) + list(forecast["lower"][::-1])
            fig.add_trace(go.Scatter(
                x=xband, y=yband, fill="toself",
                fillcolor=rgba(COLORS["AMBER"], 0.15), 
                line=dict(color="rgba(0,0,0,0)"),
                name="IC 95%", hoverinfo="skip"
            ))
            fig.add_trace(go.Scatter(
                x=forecast["dates"], y=forecast["forecast"],
                mode="lines+markers", name="Projeção",
                line=dict(color=COLORS["AMBER"], width=2, dash="dash"),
                marker=dict(size=7, symbol="diamond", color=COLORS["AMBER"])
            ))
        fig.update_layout(title=TEXT["program_evolution"])
        base_layout(fig, h=420)
        fig.update_yaxes(tickprefix="R$ ", tickformat=",.0f")
        return fig

    @staticmethod
    @st.cache_data
    def create_mom_chart(evol_g: pd.DataFrame) -> go.Figure:
        """Create month-over-month growth bar chart."""
        colors = [COLORS["VERDE"] if v >= 0 else COLORS["VERM"] for v in evol_g["mom"]]
        fig = go.Figure(go.Bar(
            x=evol_g["data_base"], y=evol_g["mom"],
            marker_color=colors, marker_line_width=0,
            hovertemplate="%{x|%b/%Y}<br>MoM: %{y:.1f}%<extra></extra>"
        ))
        fig.add_hline(y=0, line_color=COLORS["BORDA"], line_width=1.5)
        fig.update_layout(title=TEXT["monthly_growth"])
        base_layout(fig, h=420, leg=False)
        fig.update_yaxes(ticksuffix="%")
        return fig

    @staticmethod
    @st.cache_data
    def create_heatmap(df: pd.DataFrame) -> Optional[go.Figure]:
        """Create heatmap of volume by tranche and month."""
        try:
            heat_data = df.pivot_table(
                index="tipo_desenrola", columns="data_base",
                values="volume_operacoes", aggfunc="sum"
            )
            if heat_data.empty:
                return None
            heat_data.columns = [pd.Timestamp(c).strftime("%b/%y") for c in heat_data.columns]
            fig = go.Figure(go.Heatmap(
                z=heat_data.values, x=heat_data.columns, y=heat_data.index,
                colorscale=[[0, COLORS["BG"]], [0.5, COLORS["P2"]], [1, COLORS["P1"]]],
                hovertemplate="Faixa: %{y}<br>Mês: %{x}<br>R$ %{z:,.0f}<extra></extra>",
                showscale=True,
                colorbar=dict(title="Volume (R$)", tickprefix="R$ ")
            ))
            base_layout(fig, h=320, leg=False)
            fig.update_layout(title=TEXT["ren_intensity"])
            return fig
        except Exception as e:
            logger.warning(f"Heatmap creation failed: {e}")
            return None

    @staticmethod
    @st.cache_data
    def create_concentration_chart(df: pd.DataFrame, top_n: int) -> go.Figure:
        """Create horizontal bar chart of top N banks by volume."""
        banco_agg = df.groupby("nome_conglomerado_financeiro").agg(
            volume=("volume_operacoes", "sum"),
            ops=("numero_operacoes", "sum")
        ).reset_index()
        banco_agg["ticket"] = safe_divide(banco_agg["volume"], banco_agg["ops"])
        banco_agg["seg"] = banco_agg["nome_conglomerado_financeiro"].apply(classify_bank)
        banco_agg = banco_agg.nlargest(top_n, "volume")
        
        segment_colors = {
            "Tradicional": COLORS["P1"],
            "Digital": COLORS["AZUL"],
            "Investimento": COLORS["ROXO"],
            "Cooperativa": COLORS["VERDE"],
            "Fintech": COLORS["AMBER"],
            "Outros": COLORS["CINZA"]
        }
        colors = [segment_colors.get(seg, COLORS["CINZA"]) for seg in banco_agg["seg"]]
        
        fig = go.Figure(go.Bar(
            x=banco_agg["volume"], y=banco_agg["nome_conglomerado_financeiro"].str[:30],
            orientation="h",
            marker=dict(color=colors, line=dict(width=0)),
            text=banco_agg["volume"].apply(fmt_brl),
            textposition="outside",
            textfont=dict(size=10, color=COLORS["TXT"]),
            hovertemplate="<b>%{y}</b><br>Volume: R$ %{x:,.0f}<br>Ops: %{customdata[0]:,.0f}<br>Ticket: R$ %{customdata[1]:,.2f}<extra></extra>",
            customdata=banco_agg[["ops", "ticket"]].values
        ))
        fig.update_layout(title=f"{TEXT['top_n_institutions']} ({top_n})")
        base_layout(fig, h=max(400, top_n * 30), leg=False)
        fig.update_xaxes(tickprefix="R$ ", tickformat=".2s")
        return fig

    @staticmethod
    @st.cache_data
    def create_treemap(df: pd.DataFrame, top_n: int) -> go.Figure:
        """Create treemap by segment."""
        banco_agg = df.groupby("nome_conglomerado_financeiro").agg(
            volume=("volume_operacoes", "sum"),
            ops=("numero_operacoes", "sum")
        ).reset_index()
        banco_agg["ticket"] = safe_divide(banco_agg["volume"], banco_agg["ops"])
        banco_agg["seg"] = banco_agg["nome_conglomerado_financeiro"].apply(classify_bank)
        banco_agg = banco_agg.nlargest(top_n, "volume")
        
        fig = px.treemap(
            banco_agg, path=["seg", "nome_conglomerado_financeiro"],
            values="volume", color="ticket",
            color_continuous_scale=[COLORS["P1"], COLORS["P2"], COLORS["P3"]],
            title=TEXT["treemap_title"]
        )
        fig.update_layout(
            template=COLORS["TPLOTE"], height=450,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=COLORS["TXT"]),
            margin=dict(l=10, r=10, t=50, b=10)
        )
        return fig

    @staticmethod
    @st.cache_data
    def create_brazil_map(df: pd.DataFrame) -> go.Figure:
        """Create choropleth map of Brazil."""
        uf_data = df.groupby("uf_codigo")["volume_operacoes"].sum().reset_index()
        uf_data.columns = ["uf", "volume"]
        geojson = get_brazil_geojson()
        
        if geojson:
            props = geojson["features"][0]["properties"]
            featureidkey = "properties.sigla" if "sigla" in props else "properties.name"
            fig = px.choropleth(
                uf_data, geojson=geojson, locations="uf",
                featureidkey=featureidkey,
                color="volume", color_continuous_scale="Blues",
                title=TEXT["regional_distribution"],
                hover_name="uf",
                hover_data={"volume": ":,.0f"},
                labels={"volume": "Volume (R$)"}
            )
            fig.update_geos(fitbounds="locations", visible=False, bgcolor="rgba(0,0,0,0)")
            fig.update_layout(
                height=500, margin=dict(l=0, r=0, t=40, b=0),
                template=COLORS["TPLOTE"], 
                paper_bgcolor="rgba(0,0,0,0)",
                geo=dict(bgcolor="rgba(0,0,0,0)")
            )
            return fig
        else:
            fig = go.Figure()
            fig.add_annotation(
                text=TEXT["map_unavailable"],
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color=COLORS["TXT2"])
            )
            base_layout(fig, h=500)
            return fig

    @staticmethod
    @st.cache_data
    def create_regional_bar(df: pd.DataFrame) -> go.Figure:
        """Create bar + line chart for regional analysis."""
        reg_data = df.groupby("regiao").agg(
            volume=("volume_operacoes", "sum"),
            ops=("numero_operacoes", "sum")
        ).reset_index()
        reg_data["ticket"] = safe_divide(reg_data["volume"], reg_data["ops"])
        reg_data = reg_data.sort_values("volume", ascending=True)
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(
            x=reg_data["volume"], y=reg_data["regiao"],
            name="Volume", orientation="h",
            marker_color=CHART_COLORS[:len(reg_data)],
            text=reg_data["volume"].apply(fmt_brl),
            textposition="outside"
        ), secondary_y=False)
        fig.add_trace(go.Scatter(
            x=reg_data["ticket"], y=reg_data["regiao"],
            name="Ticket Médio", mode="lines+markers",
            line=dict(color=COLORS["AMBER"], width=2.5),
            marker=dict(size=8, symbol="diamond", color=COLORS["AMBER"]),
            xaxis="x2"
        ), secondary_y=True)
        fig.update_layout(
            title=TEXT["volume_and_ticket_by_region"],
            xaxis2=dict(overlaying="x", side="top", showgrid=False),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        base_layout(fig, h=450)
        fig.update_xaxes(title_text="Volume (R$)", tickprefix="R$ ", tickformat=".2s", secondary_y=False)
        fig.update_xaxes(title_text="Ticket Médio (R$)", tickprefix="R$ ", secondary_y=True, showgrid=False)
        return fig

    @staticmethod
    @st.cache_data
    def create_regional_donut(df: pd.DataFrame) -> go.Figure:
        """Create donut chart for regional share."""
        reg_data = df.groupby("regiao")["volume_operacoes"].sum().reset_index()
        fig = go.Figure(go.Pie(
            labels=reg_data["regiao"], values=reg_data["volume_operacoes"],
            hole=0.55,
            marker=dict(colors=CHART_COLORS[:len(reg_data)], 
                       line=dict(color=COLORS["BG"], width=2)),
            textinfo="label+percent",
            hovertemplate="<b>%{label}</b><br>R$ %{value:,.0f}<br>%{percent}<extra></extra>",
            textposition="outside",
            textfont=dict(size=11)
        ))
        fig.update_layout(title=TEXT["regional_share"], height=400)
        base_layout(fig, h=400)
        return fig

    @staticmethod
    @st.cache_data
    def create_cluster_chart(df: pd.DataFrame) -> Optional[go.Figure]:
        """Create clustering scatter plot."""
        try:
            cl_df = df.groupby("nome_conglomerado_financeiro").agg(
                ops=("numero_operacoes", "sum"),
                vol=("volume_operacoes", "sum")
            ).reset_index()
            cl_df["ticket"] = safe_divide(cl_df["vol"], cl_df["ops"])
            cl_df = cl_df[cl_df["ops"] > CONFIG.min_ops_for_cluster].dropna()
            if len(cl_df) < CONFIG.min_cluster_samples:
                return None
            
            scaler = StandardScaler()
            feat = scaler.fit_transform(cl_df[["ops", "ticket"]])
            n_clusters = min(4, len(cl_df))
            km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cl_df["cluster"] = km.fit_predict(feat)
            
            fig = go.Figure()
            for c in cl_df["cluster"].unique():
                grp = cl_df[cl_df["cluster"] == c]
                sz = np.log1p(grp["vol"] / grp["vol"].max() + 0.01) * 25 + 9
                fig.add_trace(go.Scatter(
                    x=grp["ops"], y=grp["ticket"], mode="markers+text",
                    name=f"Cluster {c+1} ({len(grp)} bancos)",
                    marker=dict(
                        size=sz, color=CHART_COLORS[c % len(CHART_COLORS)],
                        opacity=0.8, line=dict(width=1, color=COLORS["BORDA"])
                    ),
                    text=grp["nome_conglomerado_financeiro"].str[:15],
                    textposition="top center",
                    textfont=dict(size=8, color=COLORS["TXT2"]),
                    hovertemplate="<b>%{text}</b><br>Ops: %{x:,.0f}<br>Ticket: R$ %{y:,.2f}<extra></extra>"
                ))
            fig.update_layout(
                title=TEXT["clustering_title"],
                xaxis_title="Operações", yaxis_title="Ticket Médio (R$)"
            )
            base_layout(fig, h=500)
            return fig
        except Exception as e:
            logger.warning(f"Clustering failed: {e}")
            return None

    @staticmethod
    @st.cache_data
    def create_radar_chart(hhi_val: float, cr3: float, cr5: float, gini_r: float, ticket: float) -> go.Figure:
        """Create radar chart of concentration indices."""
        radar_data = pd.DataFrame({
            "Métrica": ["HHI", "CR3", "CR5", "Gini Regional", "Ticket Médio"],
            "Valor": [
                min(hhi_val / 3000, 1),
                cr3 / 100,
                cr5 / 100,
                min(gini_r, 1),
                min(ticket / 15000, 1)
            ]
        })
        fig = px.line_polar(
            radar_data, r="Valor", theta="Métrica",
            line_close=True,
            color_discrete_sequence=[COLORS["P1"]],
            title=TEXT["concentration_radar"]
        )
        fig.update_traces(fill="toself", fillcolor=rgba(COLORS["P1"], 0.2))
        fig.update_layout(
            template=COLORS["TPLOTE"], height=450,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            polar=dict(
                bgcolor="rgba(0,0,0,0)",
                radialaxis=dict(visible=True, range=[0, 1], gridcolor=COLORS["GRID"]),
                angularaxis=dict(gridcolor=COLORS["GRID"])
            )
        )
        return fig

    @staticmethod
    @st.cache_data
    def create_scatter_chart(df: pd.DataFrame) -> go.Figure:
        """Create scatter plot of avg ticket vs market share."""
        sc_df = df.groupby("nome_conglomerado_financeiro").agg(
            vol=("volume_operacoes", "sum"),
            ops=("numero_operacoes", "sum")
        ).reset_index()
        sc_df["ticket"] = safe_divide(sc_df["vol"], sc_df["ops"])
        sc_df["ms"] = sc_df["ops"] / sc_df["ops"].sum() * 100
        sc_df = sc_df.dropna().query("ops > 50")
        
        fig = px.scatter(
            sc_df, x="ms", y="ticket", size="vol", color="ticket",
            hover_name="nome_conglomerado_financeiro",
            title=TEXT["scatter_title"],
            labels={"ms": "Market Share (%)", "ticket": "Ticket Médio (R$)"},
            color_continuous_scale="Viridis",
            size_max=50
        )
        fig.update_layout(template=COLORS["TPLOTE"], height=450)
        fig.update_traces(marker=dict(line=dict(width=1, color=COLORS["BORDA"])))
        return fig

    @staticmethod
    @st.cache_data
    def create_outlier_chart(df: pd.DataFrame) -> Optional[go.Figure]:
        """Create outlier visualization."""
        if "is_outlier" not in df.columns:
            return None
        fig = go.Figure()
        normal = df[~df["is_outlier"]]
        fig.add_trace(go.Scatter(
            x=normal["numero_operacoes"], y=normal["volume_operacoes"],
            mode="markers", name="Normal",
            marker=dict(size=8, color=CHART_COLORS[0], opacity=0.6),
            hovertemplate="<b>%{text}</b><br>Ops: %{x:,.0f}<br>Vol: R$ %{y:,.0f}<extra></extra>",
            text=normal["nome_conglomerado_financeiro"]
        ))
        outliers = df[df["is_outlier"]]
        if not outliers.empty:
            fig.add_trace(go.Scatter(
                x=outliers["numero_operacoes"], y=outliers["volume_operacoes"],
                mode="markers", name="Outlier",
                marker=dict(size=12, color=COLORS["VERM"], symbol="x",
                           line=dict(width=2, color="white")),
                hovertemplate="<b>%{text}</b><br>Ops: %{x:,.0f}<br>Vol: R$ %{y:,.0f}<br>⚠️ OUTLIER<extra></extra>",
                text=outliers["nome_conglomerado_financeiro"]
            ))
        fig.update_layout(
            title=TEXT["outlier_detection"],
            xaxis_title="Número de Operações",
            yaxis_title="Volume (R$)"
        )
        base_layout(fig, h=450)
        return fig

    @staticmethod
    @st.cache_data
    def create_correlation_heatmap(corr_matrix: pd.DataFrame) -> Optional[go.Figure]:
        """Create correlation heatmap."""
        if corr_matrix.empty:
            return None
        fig = go.Figure(go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            colorscale=[[0, COLORS["VERM"]], [0.5, COLORS["BG"]], [1, COLORS["VERDE"]]],
            zmid=0,
            text=corr_matrix.values,
            texttemplate="%{text:.2f}",
            textfont=dict(size=12),
            hovertemplate="%{y} × %{x}<br>Correlação: %{z:.2f}<extra></extra>",
            showscale=True
        ))
        base_layout(fig, h=400, leg=False)
        fig.update_layout(title=TEXT["correlation_matrix"])
        return fig

    @staticmethod
    @st.cache_data
    def create_pareto_chart(df: pd.DataFrame) -> Tuple[go.Figure, int]:
        """Create Pareto chart and return figure and count for 80%."""
        pareto_data = df.groupby("nome_conglomerado_financeiro")["volume_operacoes"].sum()
        pareto_data = pareto_data.sort_values(ascending=False).reset_index()
        pareto_data["acum"] = pareto_data["volume_operacoes"].cumsum() / pareto_data["volume_operacoes"].sum() * 100
        p80 = (pareto_data["acum"] <= 80).sum()
        pareto_display = pareto_data.head(30)
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(
            x=pareto_display["nome_conglomerado_financeiro"].str[:20],
            y=pareto_display["volume_operacoes"],
            name="Volume", marker_color=COLORS["P1"],
            text=pareto_display["volume_operacoes"].apply(fmt_brl),
            textposition="outside"
        ), secondary_y=False)
        fig.add_trace(go.Scatter(
            x=pareto_display["nome_conglomerado_financeiro"].str[:20],
            y=pareto_display["acum"],
            name="% Acumulado", mode="lines+markers",
            line=dict(color=COLORS["AMBER"], width=2.5),
            marker=dict(size=8, color=COLORS["AMBER"])
        ), secondary_y=True)
        fig.add_hline(y=80, line_dash="dot", line_color=COLORS["VERM"], 
                     secondary_y=True, annotation_text="80%", annotation_position="top right")
        fig.update_layout(title=f"Pareto: {p80} instituições = 80% do volume")
        base_layout(fig, h=450)
        fig.update_yaxes(title_text="Volume (R$)", tickprefix="R$ ", secondary_y=False)
        fig.update_yaxes(title_text="% Acumulado", ticksuffix="%", secondary_y=True, range=[0, 105])
        return fig, p80

    @staticmethod
    @st.cache_data
    def create_seasonal_chart(decomposition: Dict) -> Optional[go.Figure]:
        """Create seasonal decomposition chart."""
        if decomposition is None:
            return None
        fig = make_subplots(
            rows=4, cols=1,
            subplot_titles=("Observado", "Tendência", "Sazonalidade", "Resíduo"),
            vertical_spacing=0.08
        )
        fig.add_trace(go.Scatter(
            x=decomposition["observed"].index, y=decomposition["observed"],
            mode="lines", name="Observado", line=dict(color=COLORS["P1"], width=1.5)
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=decomposition["trend"].index, y=decomposition["trend"],
            mode="lines", name="Tendência", line=dict(color=COLORS["AMBER"], width=2)
        ), row=2, col=1)
        fig.add_trace(go.Scatter(
            x=decomposition["seasonal"].index, y=decomposition["seasonal"],
            mode="lines", name="Sazonalidade", line=dict(color=COLORS["AZUL"], width=1.5),
            fill="tozeroy", fillcolor=rgba(COLORS["AZUL"], 0.2)
        ), row=3, col=1)
        fig.add_trace(go.Scatter(
            x=decomposition["resid"].index, y=decomposition["resid"],
            mode="lines", name="Resíduo", line=dict(color=COLORS["VERM"], width=1)
        ), row=4, col=1)
        fig.update_layout(
            height=600,
            title=TEXT["seasonal_decomposition"],
            showlegend=False
        )
        base_layout(fig, h=600, leg=False)
        fig.update_yaxes(tickprefix="R$ ", tickformat=",.0f")
        return fig


# ============================================================
# EXPORT MANAGER
# ============================================================
class ExportManager:
    """Handle data and report exports to various formats."""

    @staticmethod
    def create_excel_export(df: pd.DataFrame, kpis: Dict) -> bytes:
        """Create multi-sheet Excel workbook."""
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Dados Filtrados', index=False)
            kpis["evol_g"].to_excel(writer, sheet_name='Evolução', index=False)
            kpis["b_agg"].to_excel(writer, sheet_name='Bancos', index=False)
            kpis["reg_df"].to_excel(writer, sheet_name='Regional', index=False)
            summary = pd.DataFrame({
                "Métrica": ["Volume Total", "Operações", "Ticket Médio", "HHI", "CR3", "CR5", "Gini Regional"],
                "Valor": [
                    fmt_brl(kpis["vol_tot"]),
                    fmt_num(kpis["ops_tot"]),
                    fmt_brl(kpis["ticket"]),
                    f"{kpis['hhi']:.0f}",
                    f"{kpis['cr3']:.1f}%",
                    f"{kpis['cr5']:.1f}%",
                    f"{kpis['gini_r']:.3f}"
                ]
            })
            summary.to_excel(writer, sheet_name='Resumo', index=False)
        output.seek(0)
        return output.getvalue()

    @staticmethod
    def create_txt_report(kpis: Dict) -> str:
        """Create text summary report."""
        return f"""DESENROLA BRASIL - RELATÓRIO EXECUTIVO
{'='*50}
Gerado em: {datetime.now().strftime('%d/%m/%Y %H:%M')}

RESUMO GERAL
{'-'*50}
Volume Total Renegociado: {fmt_brl(kpis['vol_tot'])}
Total de Contratos: {fmt_num(kpis['ops_tot'])}
Ticket Médio: {fmt_brl(kpis['ticket'])}
Variação Mensal: {kpis['mom_last']:+.1f}%
Variação Anual: {kpis['yoy_last']:+.1f}%

MERCADO
{'-'*50}
Número de Instituições: {kpis['n_banks']}
Número de Estados: {kpis['n_uf']}
HHI (Concentração): {kpis['hhi']:.0f}
CR3 (Top 3): {kpis['cr3']:.1f}%
CR5 (Top 5): {kpis['cr5']:.1f}%
Gini Regional: {kpis['gini_r']:.3f}

INTERPRETAÇÃO
{'-'*50}
""" + (
    "Mercado altamente concentrado." if kpis['hhi'] > 2500 else
    "Mercado moderadamente concentrado." if kpis['hhi'] > 1500 else
    "Mercado competitivo."
) + "\n\n" + (
    "Alta desigualdade regional." if kpis['gini_r'] > 0.7 else
    "Desigualdade regional moderada." if kpis['gini_r'] > 0.5 else
    "Distribuição regional equilibrada."
) + f"""

FONTE
{'-'*50}
Banco Central do Brasil - SCR (Sistema de Informações de Crédito)
"""


# ============================================================
# MAIN APPLICATION
# ============================================================
def main():
    """Main application entry point."""
    
    # ============================================================
    # SIDEBAR
    # ============================================================
    with st.sidebar:
        st.markdown(f"""
        <div style="text-align:center; padding:0.5rem 0 1rem; border-bottom:1px solid {COLORS['BORDA']}; margin-bottom:1rem;">
            <div style="font-family:'Playfair Display',serif; font-size:1.3rem; font-weight:700; color:{COLORS['P1']};">🏦 {TEXT['app_title']}</div>
            <div style="font-size:0.7rem; color:{COLORS['TXT2']};">{TEXT['app_subtitle']}</div>
        </div>
        """, unsafe_allow_html=True)
        
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            if st.button("☀️ Light" if T == "dark" else "🌙 Dark", use_container_width=True):
                st.session_state.tema = "light" if T == "dark" else "dark"
                st.rerun()
        with col_t2:
            lang_options = {"pt": "🇧🇷 PT", "en": "🇺🇸 EN"}
            new_lang = st.selectbox(
                "Idioma", 
                options=["pt", "en"], 
                index=["pt", "en"].index(LANG),
                format_func=lambda x: lang_options[x],
                key="lang_selector"
            )
            if new_lang != LANG:
                st.session_state.lang = new_lang
                st.rerun()
        
        st.markdown("---")
        st.markdown(f"### 📂 {TEXT['data_source']}")
        
        data_source = st.radio(
            "Fonte",
            options=["demo", "upload", "local"],
            format_func=lambda x: {
                "demo": f"📊 {TEXT['use_demo_data']}",
                "upload": f"📤 {TEXT['upload_csv']}",
                "local": "📁 Arquivo local"
            }.get(x, x),
            horizontal=False,
            label_visibility="collapsed"
        )
        
        uploaded_file = None
        if data_source == "upload":
            uploaded_file = st.file_uploader(
                "Carregar arquivo",
                type=["csv", "xlsx", "xls"],
                label_visibility="collapsed"
            )
        
        st.markdown("---")
        
        with st.spinner(TEXT["loading_data"]):
            if data_source == "demo":
                raw_df = DataGenerator.generate_sample_data()
                st.session_state.demo_mode = True
            elif data_source == "upload" and uploaded_file:
                raw_df = DataLoader.load_from_file(uploaded_file)
                st.session_state.demo_mode = False
            else:
                try:
                    raw_df = pd.read_csv("dados_desenrola.csv", sep=";", encoding="utf-8")
                    st.session_state.demo_mode = False
                except FileNotFoundError:
                    raw_df = DataGenerator.generate_sample_data()
                    st.session_state.demo_mode = True
                    st.info("ℹ️ Arquivo local não encontrado. Usando dados de demonstração.")
        
        if raw_df is None:
            st.error("❌ Falha ao carregar dados.")
            st.stop()
        
        is_valid, missing_cols = DataLoader.validate_schema(raw_df)
        if not is_valid:
            st.error(f"❌ Colunas ausentes: {', '.join(missing_cols)}")
            st.stop()
        
        df = DataLoader.process_raw_data(raw_df)
        
        source_name = "Dados de Demonstração" if st.session_state.demo_mode else (
            uploaded_file.name if uploaded_file else "dados_desenrola.csv"
        )
        st.markdown(f"""
        <div class="source-card">
            <span class="source-badge">{"DEMO" if st.session_state.demo_mode else "FILE"}</span>
            <span style="font-size:0.75rem; color:{COLORS['TXT2']};">{source_name}</span>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"### 🔍 {TEXT['filters']}")
        
        if "filters_initialized" not in st.session_state:
            st.session_state.sel_tip = sorted(df["tipo_desenrola"].unique())
            st.session_state.sel_reg = sorted(df["regiao"].unique())
            st.session_state.sel_seg = sorted(df["tipo_banco"].unique())
            st.session_state.filters_initialized = True
        
        tipos = sorted(df["tipo_desenrola"].unique())
        sel_tip = st.multiselect(
            "Faixa", tipos, 
            default=st.session_state.sel_tip,
            key="sel_tip"
        )
        
        regioes = sorted(df["regiao"].unique())
        sel_reg = st.multiselect(
            "Região", regioes,
            default=st.session_state.sel_reg,
            key="sel_reg"
        )
        
        segmentos = sorted(df["tipo_banco"].unique())
        sel_seg = st.multiselect(
            "Segmento", segmentos,
            default=st.session_state.sel_seg,
            key="sel_seg"
        )
        
        datas = sorted(df["data_base"].unique())
        if len(datas) > 1:
            date_labels = [pd.Timestamp(d).strftime("%m/%Y") for d in datas]
            i0, i1 = st.select_slider(
                "Período", 
                options=list(range(len(datas))),
                value=(0, len(datas) - 1),
                format_func=lambda i: date_labels[i]
            )
            d_ini, d_fim = datas[i0], datas[i1]
        else:
            d_ini = d_fim = datas[0]
        
        if st.button(f"🔄 {TEXT['reset_filters']}", use_container_width=True):
            for key in ["sel_tip", "sel_reg", "sel_seg", "filters_initialized"]:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()
        
        st.markdown("---")
        st.markdown(f"### 📋 {TEXT['data_quality']}")
        st.markdown(f"""
        <div style="background:{COLORS['CARD_GLASS']}; border-radius:12px; padding:0.8rem; border:1px solid {COLORS['BORDA']};">
            <div style="font-size:0.65rem; color:{COLORS['TXT2']}; margin-bottom:0.3rem;">Registros</div>
            <div style="font-size:1.1rem; font-weight:700; color:{COLORS['TXT']};">{len(df):,}</div>
            <div style="font-size:0.65rem; color:{COLORS['TXT2']}; margin-top:0.5rem;">Período</div>
            <div style="font-size:0.75rem; color:{COLORS['TXT']};">{d_ini.strftime('%m/%Y')} → {d_fim.strftime('%m/%Y')}</div>
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
    ].copy()
    
    if dff.empty:
        st.warning(TEXT["no_data"])
        st.info(TEXT["no_data_suggestion"])
        st.stop()
    
    # ============================================================
    # COMPUTE ANALYTICS
    # ============================================================
    with st.spinner(TEXT["processing"]):
        kpis = AnalyticsEngine.compute_kpis(dff)
        dff_with_outliers, n_outliers = AnalyticsEngine.detect_outliers(dff)
        corr_matrix = AnalyticsEngine.compute_correlations(dff)
        seasonal = AnalyticsEngine.seasonal_decomposition(kpis["evol_g"])
    
    vol_tot = kpis["vol_tot"]
    ops_tot = kpis["ops_tot"]
    ticket = kpis["ticket"]
    evol_g = kpis["evol_g"]
    mom_last = kpis["mom_last"]
    hhi_val = kpis["hhi"]
    cr3 = kpis["cr3"]
    cr5 = kpis["cr5"]
    gini_r = kpis["gini_r"]
    n_banks = kpis["n_banks"]
    n_uf = kpis["n_uf"]
    
    # ============================================================
    # HERO SECTION
    # ============================================================
    st.markdown(f"""
    <div class="hero">
        <h1>🏦 {TEXT['hero_title']}</h1>
        <p>{TEXT['hero_subtitle']}</p>
        <div>
            <span class="hero-badge">📊 {fmt_num(ops_tot)} contratos</span>
            <span class="hero-badge">💰 {fmt_brl(vol_tot)} renegociados</span>
            <span class="hero-badge">🏛️ {n_banks} instituições</span>
            <span class="hero-badge">🗺️ {n_uf} estados</span>
            {f'<span class="hero-badge">🧪 Modo Demo</span>' if st.session_state.demo_mode else ''}
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
    
    if mom_last < CONFIG.mom_drop_sharp:
        alertas.append(("er", TEXT["alert_sharp_drop"].format(mom_last)))
    elif mom_last < CONFIG.mom_drop_moderate:
        alertas.append(("wa", TEXT["alert_slowdown"].format(mom_last)))
    elif mom_last > CONFIG.mom_growth_strong:
        alertas.append(("ok", TEXT["alert_strong_accel"].format(mom_last)))
    elif mom_last > 0:
        alertas.append(("ok", TEXT["alert_stable_growth"].format(mom_last)))
    
    if hhi_val > CONFIG.hhi_threshold_high:
        alertas.append(("er", TEXT["alert_high_concentration"].format(CONFIG.hhi_threshold_high)))
    elif hhi_val > CONFIG.hhi_threshold_moderate:
        alertas.append(("wa", TEXT["alert_moderate_concentration"].format(
            CONFIG.hhi_threshold_moderate, CONFIG.hhi_threshold_high)))
    else:
        alertas.append(("ok", TEXT["alert_competitive"].format(CONFIG.hhi_threshold_moderate)))
    
    if gini_r > CONFIG.gini_threshold_high:
        alertas.append(("er", TEXT["alert_high_regional_inequality"].format(gini_r)))
    elif gini_r > CONFIG.gini_threshold_moderate:
        alertas.append(("wa", TEXT["alert_regional_inequality"].format(gini_r)))
    
    if n_outliers > 0:
        alertas.append(("wa", TEXT["alert_outliers_detected"].format(n_outliers)))
    
    if seasonal:
        alertas.append(("in", TEXT["alert_seasonal_pattern"]))
    
    if alertas:
        st.markdown(f"### ⚡ {TEXT['smart_alerts']}")
        cols = st.columns(min(len(alertas), 4))
        for i, (cls, msg) in enumerate(alertas):
            with cols[i % len(cols)]:
                st.markdown(f'<div class="al {cls}">{msg}</div>', unsafe_allow_html=True)
    
    # ============================================================
    # INSIGHTS
    # ============================================================
    st.markdown(f"### 🔍 {TEXT['automated_insights']}")
    
    reg_leader = kpis["reg_df"].loc[kpis["reg_df"]["volume_operacoes"].idxmax()]
    reg_name = reg_leader["regiao"]
    reg_vol = reg_leader["volume_operacoes"]
    reg_pct = reg_vol / kpis["reg_df"]["volume_operacoes"].sum() * 100
    
    if "grande_area" in dff.columns:
        area_df = dff.groupby("grande_area")["volume_operacoes"].sum().reset_index()
        area_leader = area_df.loc[area_df["volume_operacoes"].idxmax()] if not area_df.empty else None
    else:
        area_leader = None
    
    cresc_med = evol_g["mom"].dropna().mean() if not evol_g["mom"].dropna().empty else 0.0
    
    if not corr_matrix.empty and "volume_operacoes" in corr_matrix.columns and "ticket_medio" in corr_matrix.columns:
        corr_val = corr_matrix.loc["volume_operacoes", "ticket_medio"]
    else:
        corr_val = 0.0
    
    insight_cols = st.columns(3)
    with insight_cols[0]:
        st.markdown(f"""
        <div class="insight-card">
            <div class="insight-title">📍 {TEXT['regional_concentration']}</div>
            <div class="insight-text">{TEXT['insight_regional'].format(reg_name, reg_pct)}</div>
            <div class="insight-value">{fmt_brl(reg_vol)}</div>
        </div>
        """, unsafe_allow_html=True)
    with insight_cols[1]:
        if area_leader is not None:
            st.markdown(f"""
            <div class="insight-card">
                <div class="insight-title">🧬 {TEXT['leading_area']}</div>
                <div class="insight-text">{TEXT['insight_area'].format(area_leader['grande_area'])}</div>
                <div class="insight-value">{fmt_brl(area_leader['volume_operacoes'])}</div>
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
        st.markdown(f"""
        <div class="insight-card">
            <div class="insight-title">🔗 {TEXT['insight_correlation'].split(':')[0]}</div>
            <div class="insight-text">{TEXT['insight_correlation'].format(corr_val)}</div>
            <div class="insight-value">{TEXT['insight_top_banks'].format(cr3)}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # ============================================================
    # TABS
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
            fig1 = ChartFactory.create_timeline_chart(dff, evol_g)
            st.plotly_chart(fig1, use_container_width=True)
        with col_ev2:
            fig2 = ChartFactory.create_mom_chart(evol_g)
            st.plotly_chart(fig2, use_container_width=True)
        
        st.markdown(f"**🔥 {TEXT['heatmap_title']}**")
        fig3 = ChartFactory.create_heatmap(dff)
        if fig3:
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.info(TEXT["heatmap_unavailable"])
        
        if seasonal:
            st.markdown(f"**📊 {TEXT['seasonal_decomposition']}**")
            fig_seasonal = ChartFactory.create_seasonal_chart(seasonal)
            if fig_seasonal:
                st.plotly_chart(fig_seasonal, use_container_width=True)
    
    # ---------- TAB 2: BANK CONCENTRATION ----------
    with tab2:
        st.markdown(f"### 🏦 {TEXT['bank_concentration']}")
        top_n = st.slider(TEXT['top_n_institutions'], 5, 30, CONFIG.default_top_n, key="top_n_bancos")
        fig4 = ChartFactory.create_concentration_chart(dff, top_n)
        st.plotly_chart(fig4, use_container_width=True)
        fig5 = ChartFactory.create_treemap(dff, top_n)
        st.plotly_chart(fig5, use_container_width=True)
        
        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            st.metric("CR3", f"{cr3:.1f}%", help="Concentração dos 3 maiores bancos")
        with col_m2:
            st.metric("CR5", f"{cr5:.1f}%", help="Concentração dos 5 maiores bancos")
        with col_m3:
            gini_b = gini(kpis["b_agg"]["volume_operacoes"]) if not kpis["b_agg"].empty else 0
            st.metric("Gini Bancário", f"{gini_b:.3f}", help="Desigualdade entre bancos")
    
    # ---------- TAB 3: REGIONAL ANALYSIS ----------
    with tab3:
        st.markdown(f"### 🗺️ {TEXT['regional_distribution']}")
        fig_map = ChartFactory.create_brazil_map(dff)
        st.plotly_chart(fig_map, use_container_width=True)
        col_r1, col_r2 = st.columns([2, 1])
        with col_r1:
            fig_bar = ChartFactory.create_regional_bar(dff)
            st.plotly_chart(fig_bar, use_container_width=True)
        with col_r2:
            fig_donut = ChartFactory.create_regional_donut(dff)
            st.plotly_chart(fig_donut, use_container_width=True)
    
    # ---------- TAB 4: ADVANCED ANALYTICS ----------
    with tab4:
        st.markdown(f"### 🔬 {TEXT['advanced_analytics_title']}")
        fig_cluster = ChartFactory.create_cluster_chart(dff)
        if fig_cluster:
            st.plotly_chart(fig_cluster, use_container_width=True)
        else:
            st.info(TEXT["clustering_unavailable"])
        
        col_a1, col_a2 = st.columns(2)
        with col_a1:
            fig_radar = ChartFactory.create_radar_chart(hhi_val, cr3, cr5, gini_r, ticket)
            st.plotly_chart(fig_radar, use_container_width=True)
        with col_a2:
            fig_corr = ChartFactory.create_correlation_heatmap(corr_matrix)
            if fig_corr:
                st.plotly_chart(fig_corr, use_container_width=True)
        
        fig_scatter = ChartFactory.create_scatter_chart(dff)
        st.plotly_chart(fig_scatter, use_container_width=True)
        fig_outlier = ChartFactory.create_outlier_chart(dff_with_outliers)
        if fig_outlier:
            st.plotly_chart(fig_outlier, use_container_width=True)
        else:
            st.info(TEXT["outliers_unavailable"])
    
    # ---------- TAB 5: DISTRIBUTION ----------
    with tab5:
        st.markdown(f"### 📊 {TEXT['pareto_title']}")
        fig_pareto, p80 = ChartFactory.create_pareto_chart(dff)
        st.plotly_chart(fig_pareto, use_container_width=True)
        st.markdown(f"""
        <div class="insight-card">
            <div class="insight-title">📐 {TEXT['pareto_interpretation']}</div>
            <div class="insight-text"><b>{p80}</b> {TEXT['pareto_text']}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # ============================================================
    # EXPORT SECTION
    # ============================================================
    st.markdown("---")
    st.markdown(f"### 📥 {TEXT['export_section']}")
    
    col_exp1, col_exp2, col_exp3, col_exp4 = st.columns(4)
    with col_exp1:
        csv_data = dff.to_csv(index=False).encode("utf-8")
        st.download_button(
            f"📄 {TEXT['csv_download']}",
            csv_data,
            file_name=f"desenrola_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    with col_exp2:
        excel_data = ExportManager.create_excel_export(dff, kpis)
        st.download_button(
            f"📊 {TEXT['excel_download']}",
            excel_data,
            file_name=f"desenrola_{datetime.now().strftime('%Y%m%d')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
    with col_exp3:
        txt_report = ExportManager.create_txt_report(kpis)
        st.download_button(
            f"📝 {TEXT['report_download']}",
            txt_report,
            file_name=f"relatorio_{datetime.now().strftime('%Y%m%d')}.txt",
            mime="text/plain",
            use_container_width=True
        )
    with col_exp4:
        json_data = json.dumps({
            "generated_at": datetime.now().isoformat(),
            "filters": {
                "tipos": sel_tip,
                "regioes": sel_reg,
                "segmentos": sel_seg,
                "periodo": f"{d_ini.strftime('%Y%m')} - {d_fim.strftime('%Y%m')}"
            },
            "kpis": {
                "vol_tot": vol_tot,
                "ops_tot": ops_tot,
                "ticket": ticket,
                "hhi": hhi_val,
                "cr3": cr3,
                "cr5": cr5,
                "gini_regional": gini_r
            }
        }, indent=2, ensure_ascii=False)
        st.download_button(
            "🔧 JSON (API)",
            json_data,
            file_name=f"desenrola_{datetime.now().strftime('%Y%m%d')}.json",
            mime="application/json",
            use_container_width=True
        )
    
    # ============================================================
    # FOOTER
    # ============================================================
    st.markdown(f"""
    <div class="footer">
        🏦 {TEXT['footer_text']}<br>
        {TEXT['footer_source']}: <a href="https://www.bcb.gov.br/estatisticas/scr" target="_blank" style="color:{COLORS['P1']};">Banco Central do Brasil (SCR)</a><br>
        <span style="font-size:0.55rem; color:{COLORS['TXT2']};">v2.0 final | Built with Streamlit & Plotly</span>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
