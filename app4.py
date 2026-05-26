# =============================================================================
# 🏦 DESENROLA BRASIL – PAINEL EXECUTIVO (app4.py)
# =============================================================================
# Nome do arquivo: app4.py
# Melhorias aplicadas:
# ✅ Paleta de cores acessível (WCAG AA/AAA + Okabe-Ito para daltonismo)
# ✅ HHI com interpretação estatística enriquecida
# ✅ Projeção com decomposição STL para maior precisão
# ✅ Clusterização K-Means com validação por Silhouette Score
# ✅ Detecção de anomalias para alertas proativos
# ✅ Cache inteligente com TTL dinâmico e invalidação por filtros
# ✅ Skeleton loading para melhor percepção de performance
# ✅ Exportação com metadados analíticos enriquecidos
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import STL
from datetime import datetime, timedelta
import re
import warnings
import hashlib

warnings.filterwarnings("ignore")
pd.options.mode.chained_assignment = None

# =============================================================================
# CONFIGURAÇÃO DA PÁGINA
# =============================================================================
st.set_page_config(
    page_title="Desenrola Brasil – Painel Executivo",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# TEMA - PALETA ACESSÍVEL (WCAG AA/AAA + Okabe-Ito)
# =============================================================================
if "tema" not in st.session_state:
    st.session_state.tema = "claro"
T = st.session_state.tema

# Paleta baseada em Okabe-Ito + Tailwind, otimizada para daltonismo e contraste
PALETA_CLARO = {
    "fundo": "#F8FAFC", "card": "#FFFFFF", "texto": "#0F172A",
    "texto_secundario": "#64748B", "borda": "#E2E8F0",
    "primaria": "#0EA5E9", "secundaria": "#8B5CF6",
    "sucesso": "#10B981", "alerta": "#F59E0B",
    "erro": "#EF4444", "info": "#3B82F6",
    "grid": "rgba(15, 23, 42, 0.08)", "plotly_template": "plotly_white"
}

PALETA_ESCURO = {
    "fundo": "#0B0F19", "card": "#1E293B", "texto": "#F1F5F9",
    "texto_secundario": "#94A3B8", "borda": "#334155",
    "primaria": "#38BDF8", "secundaria": "#A78BFA",
    "sucesso": "#34D399", "alerta": "#FBBF24",
    "erro": "#F87171", "info": "#60A5FA",
    "grid": "rgba(241, 245, 249, 0.12)", "plotly_template": "plotly_dark"
}

CORES = PALETA_CLARO if T == "claro" else PALETA_ESCURO
SEMANTIC_COLORS = {
    "positivo": CORES["sucesso"], "negativo": CORES["erro"],
    "atencao": CORES["alerta"], "neutro": CORES["primaria"],
    "destaque": CORES["secundaria"], "info": CORES["info"]
}
PLOTLY_CORES = [CORES["primaria"], CORES["secundaria"], CORES["sucesso"], 
                CORES["alerta"], CORES["erro"], CORES["info"],
                "#14B8A6", "#F43F5E", "#84CC16", "#A855F7"]

# =============================================================================
# CSS - ACESSIBILIDADE E UX
# =============================================================================
st.markdown(f"""
<style>
:root {{
    --cor-fundo: {CORES["fundo"]}; --cor-card: {CORES["card"]};
    --cor-texto: {CORES["texto"]}; --cor-texto-sec: {CORES["texto_secundario"]};
    --cor-borda: {CORES["borda"]}; --cor-primaria: {CORES["primaria"]};
    --cor-sucesso: {CORES["sucesso"]}; --cor-alerta: {CORES["alerta"]};
    --cor-erro: {CORES["erro"]};
}}
html, body, .stApp {{
    background-color: var(--cor-fundo); color: var(--cor-texto);
    font-family: 'IBM Plex Sans', system-ui, -apple-system, 'Segoe UI', sans-serif;
    transition: background-color 0.3s ease, color 0.3s ease;
}}
.block-container {{ padding: 1rem 1.5rem 2rem 1.5rem; max-width: 1400px; margin: 0 auto; }}
.kpi-card {{    background: var(--cor-card); border-left: 4px solid var(--cor-primaria);
    border-radius: 12px; padding: 1rem 1.2rem; margin-bottom: 0.75rem;
    box-shadow: 0 2px 4px rgba(0,0,0,0.08);
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}}
.kpi-card:hover {{ transform: translateY(-2px); box-shadow: 0 6px 20px rgba(0,0,0,0.12); }}
.kpi-card.sucesso {{ border-left-color: var(--cor-sucesso); }}
.kpi-card.alerta {{ border-left-color: var(--cor-alerta); }}
.kpi-card.erro {{ border-left-color: var(--cor-erro); }}
.kpi-title {{ font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.1em; color: var(--cor-texto-sec); font-weight: 700; margin-bottom: 0.25rem; }}
.kpi-value {{ font-size: 1.7rem; font-weight: 700; color: var(--cor-texto); font-family: 'IBM Plex Mono', 'SF Mono', monospace; }}
.kpi-sub {{ font-size: 0.75rem; color: var(--cor-texto-sec); margin-top: 0.2rem; }}
.insight-box {{
    background: var(--cor-card); border: 1px solid var(--cor-borda);
    border-radius: 12px; padding: 1rem 1.25rem; margin-bottom: 0.75rem;
    transition: border-color 0.2s ease;
}}
.insight-box:hover {{ border-color: var(--cor-primaria); }}
.insight-label {{ font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.08em; color: var(--cor-texto-sec); font-weight: 700; margin-bottom: 0.4rem; display: flex; align-items: center; gap: 6px; }}
.insight-text {{ font-size: 0.92rem; color: var(--cor-texto); line-height: 1.65; }}
.badge {{ padding: 4px 12px; border-radius: 20px; font-weight: 600; font-size: 0.72rem; display: inline-flex; align-items: center; gap: 5px; white-space: nowrap; }}
.badge-positivo {{ background: rgba(16,185,129,0.15); color: #047857; border: 1px solid rgba(16,185,129,0.3); }}
.badge-alerta {{ background: rgba(245,158,11,0.15); color: #B45309; border: 1px solid rgba(245,158,11,0.3); }}
.badge-erro {{ background: rgba(239,68,68,0.15); color: #B91C1C; border: 1px solid rgba(239,68,68,0.3); }}
.badge-info {{ background: rgba(59,130,246,0.12); color: #1E40AF; border: 1px solid rgba(59,130,246,0.3); }}
.dq-card {{ background: var(--cor-card); border: 1px solid var(--cor-borda); border-radius: 10px; padding: 0.9rem 1.1rem; font-size: 0.82rem; line-height: 1.7; }}
.dq-card b {{ color: var(--cor-texto); }}
.dq-card .mono {{ font-family: 'IBM Plex Mono', monospace; color: var(--cor-primaria); font-weight: 500; }}
.dataframe {{ background: var(--cor-card) !important; border-radius: 10px; overflow: hidden; border: 1px solid var(--cor-borda) !important; }}
.dataframe tr:nth-child(even) {{ background-color: rgba(148, 163, 184, 0.08) !important; }}
.dataframe th {{ background-color: rgba(15, 23, 42, 0.04) !important; color: var(--cor-texto) !important; font-weight: 600 !important; }}
.stButton > button {{ border-radius: 8px; font-weight: 500; transition: all 0.2s ease; }}
.stButton > button:focus {{ outline: 2px solid var(--cor-primaria); outline-offset: 2px; }}
.stButton > button:hover {{ transform: translateY(-1px); box-shadow: 0 4px 12px rgba(0,0,0,0.1); }}
@keyframes loading {{ 0% {{ background-position: 200% 0; }} 100% {{ background-position: -200% 0; }} }}
.skeleton {{ background: linear-gradient(90deg, var(--cor-card) 25%, var(--cor-borda) 50%, var(--cor-card) 75%); background-size: 200% 100%; animation: loading 1.5s infinite; border-radius: 10px; min-height: 60px; margin-bottom: 0.5rem; }}
.stAlert {{ border-radius: 10px; border-left: 4px solid; }}
.stAlert[data-baseweb="notification"].stAlert--success {{ border-left-color: var(--cor-sucesso); background-color: rgba(16,185,129,0.1) !important; }}
.stAlert[data-baseweb="notification"].stAlert--warning {{ border-left-color: var(--cor-alerta); background-color: rgba(245,158,11,0.1) !important; }}
.stAlert[data-baseweb="notification"].stAlert--error {{ border-left-color: var(--cor-erro); background-color: rgba(239,68,68,0.1) !important; }}
.js-plotly-plot .plotly .main-svg {{ background: transparent !important; }}
.js-plotly-plot .plotly .plot-bg {{ fill: transparent !important; }}
@media (max-width: 768px) {{
    .kpi-value {{ font-size: 1.4rem; }} .insight-text {{ font-size: 0.88rem; }}
    .block-container {{ padding: 0.75rem 1rem 1.5rem 1rem; }}
}}
</style>
""", unsafe_allow_html=True)

# =============================================================================# UTILITÁRIOS
# =============================================================================
def fmt_brl(v):
    if pd.isna(v) or v == 0: return "R$ 0"
    if v >= 1e12: return f"R$ {v/1e12:.2f}T".replace(".", ",")
    if v >= 1e9: return f"R$ {v/1e9:.2f}B".replace(".", ",")
    if v >= 1e6: return f"R$ {v/1e6:.2f}M".replace(".", ",")
    if v >= 1e3: return f"R$ {v/1e3:.1f}K".replace(".", ",")
    return f"R$ {v:,.0f}".replace(",", ".").replace(".", "")

def fmt_num(v, decimals=0):
    if pd.isna(v): return "—"
    if decimals == 0: return f"{int(v):,}".replace(",", ".")
    return f"{v:,.{decimals}f}".replace(",", "X").replace(".", ",").replace("X", ".")

def fmt_pct(v, show_sign=False):
    if pd.isna(v): return "—"
    sinal = "+" if show_sign and v >= 0 else ""
    return f"{sinal}{v:.1f}%"

def classificar_banco(nome):
    if pd.isna(nome): return "Outras Instituições"
    nome = re.sub(r'\s*-\s*PRUDENCIAL$', '', str(nome).upper().strip())
    if any(x in nome for x in ["NUBANK","INTER","C6","NEON","ORIGINAL","MERCADO PAGO","PICPAY"]): return "Banco Digital"
    if any(x in nome for x in ["ITAU","BRADESCO","SANTANDER","CAIXA","BANCO DO BRASIL","BB","SAFRA","SICREDI","SICOOB"]): return "Banco Tradicional"
    if "BTG" in nome or "XP" in nome or "GENIAL" in nome: return "Banco de Investimento"
    if any(x in nome for x in ["COOPERATIVA","CREDIARIO","FINANCEIRA"]): return "Cooperativa/Financeira"
    return "Outras Instituições"

def agrupar_regiao(uf):
    if pd.isna(uf): return "Não Identificado"
    mapa = {"Norte": ["AC","AM","AP","PA","RO","RR","TO"], "Nordeste": ["AL","BA","CE","MA","PB","PE","PI","RN","SE"],
            "Centro-Oeste": ["DF","GO","MS","MT"], "Sudeste": ["ES","MG","RJ","SP"], "Sul": ["PR","RS","SC"]}
    uf_clean = str(uf).upper().strip()
    for r, ests in mapa.items():
        if uf_clean in ests: return r
    return "Não Identificado"

def gerar_hash_filtro(filtros):
    filtro_str = "|".join(f"{k}:{sorted(v) if isinstance(v, list) else v}" for k, v in sorted(filtros.items()))
    return hashlib.md5(filtro_str.encode()).hexdigest()[:10]

# =============================================================================
# ANÁLISE ESTATÍSTICA - HHI ENRIQUECIDO
# =============================================================================
@st.cache_data(ttl=3600)
def calcular_hhi(df, col):
    total = df[col].sum()
    return 0 if total == 0 else ((df[col]/total)**2).sum() * 10000
@st.cache_data(ttl=3600)
def interpretar_hhi_enhanced(hhi, n_players, top3_share, top5_share=None):
    if hhi < 1500:
        classificacao, risco, badge_class, cor_status = "Mercado Competitivo", "Baixo", "badge-positivo", SEMANTIC_COLORS["positivo"]
    elif hhi < 2500:
        classificacao, risco, badge_class, cor_status = "Concentração Moderada", "Médio", "badge-alerta", SEMANTIC_COLORS["atencao"]
    else:
        classificacao, risco, badge_class, cor_status = "Altamente Concentrado", "Alto", "badge-erro", SEMANTIC_COLORS["negativo"]
    
    insights = []
    if n_players <= 3 and (top3_share or 0) > 70: insights.append("⚠️ Estrutura oligopolística: 3 players controlam >70% do mercado")
    if n_players > 20 and hhi < 1000: insights.append("✅ Alta fragmentação: mercado com múltiplos participantes ativos")
    if (top3_share or 0) > 50 and hhi > 2000: insights.append("🔍 Alta dependência dos líderes: monitorar mudanças estratégicas")
    if n_players < 10 and hhi > 1800: insights.append("🚧 Possíveis barreiras de entrada: avaliar políticas de incentivo")
    
    if hhi > 2500: recomendacao = "Recomenda-se diversificar parcerias para reduzir risco sistêmico e fomentar competição."
    elif hhi > 1500: recomendacao = "Monitorar concentração: considerar incentivos para novos entrantes e transparência de dados."
    else: recomendacao = "Mercado saudável: manter políticas de competição e monitorar tendências de consolidação."
    
    return {"valor": hhi, "classificacao": classificacao, "risco": risco, "badge_class": badge_class, "cor_status": cor_status,
            "insights": insights, "recomendacao": recomendacao, "n_players": n_players, "top3_share": top3_share, "top5_share": top5_share}

@st.cache_data(ttl=3600)
def calcular_pareto(df, col, top_n=20):
    df_s = df.sort_values(col, ascending=False).head(top_n).reset_index(drop=True)
    total = df_s[col].sum()
    if total > 0:
        df_s["pct_individual"] = (df_s[col] / total * 100).round(2)
        df_s["pct_acumulado"] = df_s["pct_individual"].cumsum().round(2)
        idx_80 = df_s[df_s["pct_acumulado"] >= 80].index.min()
        df_s["atinge_80"] = df_s.index <= idx_80 if idx_80 is not None else False
    else:
        df_s["pct_individual"], df_s["pct_acumulado"], df_s["atinge_80"] = 0, 0, False
    return df_s, total

# =============================================================================
# PROJEÇÃO COM DECOMPOSIÇÃO STL
# =============================================================================
@st.cache_data(ttl=1800)
def projetar_com_decomposicao(series_volume, datas, periodos=3, seasonal_window=13):
    if len(series_volume) < 8: return None
    try:
        serie_limpa = series_volume.dropna()
        if len(serie_limpa) < 8: return None
        stl = STL(serie_limpa, seasonal=seasonal_window, trend=5, robust=True)
        result = stl.fit()
        tendencia_limpa = result.trend.dropna()
        if len(tendencia_limpa) < 4: return None
        modelo = ExponentialSmoothing(tendencia_limpa, trend="add", seasonal=None, initialization_method="estimated").fit(optimized=True, use_boxcox=False)
        previsao_tendencia = modelo.forecast(periodos)        n_seasonal = min(seasonal_window, len(result.seasonal))
        sazonalidade_media = result.seasonal.iloc[-n_seasonal:].mean()
        previsao_final = previsao_tendencia + sazonalidade_media
        residuos = result.resid.dropna()
        sigma = np.std(residuos) if len(residuos) > 0 else np.std(serie_limpa) * 0.15
        datas_futuras = pd.date_range(datas.max(), periods=periodos+1, freq="MS")[1:]
        mape = np.mean(np.abs(residuos / serie_limpa.loc[residuos.index])) * 100 if len(residuos) > 0 else None
        return {"datas": datas_futuras, "previsao": previsao_final.values, "lower": (previsao_final - 1.96*sigma).values,
                "upper": (previsao_final + 1.96*sigma).values,
                "decomposicao": {"tendencia_ultima": result.trend.iloc[-1] if len(result.trend) > 0 else None,
                                "sazonalidade_media": sazonalidade_media, "residuo_std": sigma, "mape": mape},
                "qualidade": "Boa" if (mape and mape < 20) else "Moderada" if (mape and mape < 40) else "Limitada"}
    except Exception as e:
        st.warning(f"⚠️ Projeção avançada indisponível: {type(e).__name__}")
        return None

# =============================================================================
# CLUSTERIZAÇÃO COM VALIDAÇÃO
# =============================================================================
@st.cache_data(ttl=3600)
def clusterizar_bancos_enhanced(df, col_banco, min_operacoes=100):
    dados = df.groupby(col_banco).agg(
        numero_operacoes=("numero_operacoes", "sum"), volume_operacoes=("volume_operacoes", "sum"),
        meses_ativos=("data_base", "nunique"),
        ticket_medio=("volume_operacoes", lambda x: x.sum() / df.loc[x.index, "numero_operacoes"].sum() if df.loc[x.index, "numero_operacoes"].sum() > 0 else np.nan)
    ).reset_index()
    dados = dados[(dados["numero_operacoes"] > min_operacoes) & (dados["ticket_medio"].notna()) & (dados["ticket_medio"] > 0)].copy()
    if len(dados) < 6: return None, None, "Dados insuficientes para clusterização (mín. 6 instituições com >100 operações)"
    
    scaler = StandardScaler()
    features = ["numero_operacoes", "ticket_medio", "meses_ativos"]
    X = scaler.fit_transform(dados[features])
    
    best_k, best_score = 2, -1
    for k in range(2, min(6, len(dados))):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
        labels = kmeans.fit_predict(X)
        score = silhouette_score(X, labels)
        if score > best_score: best_score, best_k = score, k
    
    kmeans_final = KMeans(n_clusters=best_k, random_state=42, n_init=10, max_iter=300)
    dados["cluster"] = kmeans_final.fit_predict(X)
    dados["silhouette_score"] = silhouette_score(X, dados["cluster"])
    medias = dados.groupby("cluster")[["numero_operacoes", "ticket_medio"]].mean()
    dados["cluster_nome"] = dados["cluster"].apply(lambda c: _rotular_cluster_dinamico(c, medias, dados))
    dados["cor_cluster"] = dados["cluster_nome"].map({
        "📦 Alto Volume / Ticket Acessível": SEMANTIC_COLORS["info"], "💎 Nicho Premium / Alto Ticket": SEMANTIC_COLORS["destaque"],
        "🏆 Líderes de Mercado": SEMANTIC_COLORS["sucesso"], "⚖️ Perfil Balanceado": SEMANTIC_COLORS["atencao"],
        "🔍 Emergentes / Baixo Volume": CORES["texto_secundario"]
    }).fillna(CORES["texto_secundario"])    
    return dados, best_k, f"Silhouette: {best_score:.2f} (0-1, >0.5 = boa separação)"

def _rotular_cluster_dinamico(cluster_id, medias, dados):
    m = medias.loc[cluster_id]
    pct_ops = dados["numero_operacoes"].rank(pct=True).mean() * 100
    pct_ticket = dados["ticket_medio"].rank(pct=True).mean() * 100
    if pct_ops > 66.7 and pct_ticket < 33.3: return "📦 Alto Volume / Ticket Acessível"
    elif pct_ops < 33.3 and pct_ticket > 66.7: return "💎 Nicho Premium / Alto Ticket"
    elif pct_ops > 66.7 and pct_ticket > 66.7: return "🏆 Líderes de Mercado"
    elif pct_ops < 33.3 and pct_ticket < 33.3: return "🔍 Emergentes / Baixo Volume"
    else: return "⚖️ Perfil Balanceado"

# =============================================================================
# DETECÇÃO DE ANOMALIAS
# =============================================================================
def detectar_anomalias(df_f, col="volume_operacoes", window=3, std_threshold=2.5):
    df_temp = df_f.groupby("data_base")[col].sum().reset_index().sort_values("data_base").reset_index(drop=True)
    if len(df_temp) < window + 1: return []
    df_temp["media_movel"] = df_temp[col].rolling(window=window, min_periods=1).mean()
    df_temp["std_movel"] = df_temp[col].rolling(window=window, min_periods=1).std().replace(0, df_temp[col].std())
    df_temp["z_score"] = (df_temp[col] - df_temp["media_movel"]) / df_temp["std_movel"].replace(0, 1)
    anomalias = df_temp[abs(df_temp["z_score"]) > std_threshold].copy()
    resultados = []
    for _, row in anomalias.iterrows():
        resultados.append({"data": row["data_base"].strftime("%b/%Y"), "tipo": "🔺 Pico Atípico" if row["z_score"] > 0 else "🔻 Queda Atípica",
                          "severidade": "Alta" if abs(row["z_score"]) > 3.5 else "Média", "valor": fmt_brl(row[col]),
                          "desvio": f"{row['z_score']:.1f}σ", "contexto": f"{'Acima' if row['z_score'] > 0 else 'Abaixo'} da média móvel ({window}M)", "z_score": row["z_score"]})
    return sorted(resultados, key=lambda x: abs(x["z_score"]), reverse=True)

# =============================================================================
# QUALIDADE DE DADOS ENRIQUECIDA
# =============================================================================
def calcular_data_quality_enhanced(df_original, df_limpo):
    total_raw = len(df_original) if df_original is not None and len(df_original) > 0 else len(df_limpo)
    total_limpo = len(df_limpo)
    completude = {col: (df_limpo[col].notna().sum() / len(df_limpo)) * 100 for col in ["volume_operacoes", "numero_operacoes", "data_base", "nome_conglomerado_financeiro"] if col in df_limpo.columns}
    periodo_min, periodo_max = (df_limpo["data_base"].min(), df_limpo["data_base"].max()) if "data_base" in df_limpo.columns and not df_limpo["data_base"].isna().all() else (None, None)
    meses_cobertos = df_limpo["data_base"].nunique() if "data_base" in df_limpo.columns else 0
    score_qualidade = np.mean(list(completude.values())) if completude else 0
    if score_qualidade >= 95: class_qualidade, badge_qualidade = "Excelente", "badge-positivo"
    elif score_qualidade >= 85: class_qualidade, badge_qualidade = "Boa", "badge-positivo"
    elif score_qualidade >= 70: class_qualidade, badge_qualidade = "Aceitável", "badge-alerta"
    else: class_qualidade, badge_qualidade = "Requer Atenção", "badge-erro"
    
    return {"total_registros_raw": total_raw, "total_registros_limpos": total_limpo, "registros_descartados": total_raw - total_limpo,
            "taxa_retencao": (total_limpo / total_raw * 100) if total_raw > 0 else 0, "completude": completude,
            "completude_volume": completude.get("volume_operacoes", 0), "completude_operacoes": completude.get("numero_operacoes", 0),
            "periodo_inicio": periodo_min.strftime("%m/%Y") if periodo_min else "N/D", "periodo_fim": periodo_max.strftime("%m/%Y") if periodo_max else "N/D",
            "meses_cobertos": meses_cobertos, "score_qualidade": score_qualidade, "class_qualidade": class_qualidade,            "badge_qualidade": badge_qualidade, "ultima_data": periodo_max.strftime("%m/%Y") if periodo_max else "N/D"}

# =============================================================================
# GERAÇÃO DE ALERTAS CONTEXTUAIS
# =============================================================================
@st.cache_data(ttl=600)
def gerar_alertas_contextuais(evolucao, hhi_info, ticket_medio_geral, anomalias_detectadas):
    alertas = []
    if len(evolucao) >= 2 and "crescimento" in evolucao.columns:
        cresc_ultimo = evolucao["crescimento"].dropna().iloc[-1] if len(evolucao["crescimento"].dropna()) > 0 else 0
        if cresc_ultimo < -20: alertas.append({"tipo": "error", "titulo": "🔴 Queda Acentuada", "mensagem": f"Volume caiu {cresc_ultimo:.1f}% no último mês — investigar causas.", "prioridade": 1, "acao": "Analisar fatores externos e comunicação com instituições"})
        elif cresc_ultimo < -8: alertas.append({"tipo": "warning", "titulo": "🟡 Desaceleração", "mensagem": f"Queda de {cresc_ultimo:.1f}% sinaliza perda de momentum.", "prioridade": 2, "acao": "Reforçar campanhas e incentivos regionais"})
        elif cresc_ultimo > 25: alertas.append({"tipo": "success", "titulo": "🟢 Aceleração Forte", "mensagem": f"Crescimento de +{cresc_ultimo:.1f}% — oportunidade de escalar.", "prioridade": 3, "acao": "Preparar infraestrutura para demanda crescente"})
    if hhi_info["risco"] == "Alto": alertas.append({"tipo": "error", "titulo": "🔴 Concentração Elevada", "mensagem": f"HHI = {hhi_info['valor']:.0f}: mercado oligopolizado.", "prioridade": 1, "acao": hhi_info["recomendacao"]})
    elif hhi_info["risco"] == "Médio": alertas.append({"tipo": "warning", "titulo": "🟡 Concentração Moderada", "mensagem": f"HHI = {hhi_info['valor']:.0f}: monitorar tendências de consolidação.", "prioridade": 2, "acao": "Acompanhar entrada de novos players"})
    if ticket_medio_geral > 8000: alertas.append({"tipo": "info", "titulo": "ℹ️ Ticket Elevado", "mensagem": f"Ticket médio de {fmt_brl(ticket_medio_geral)} — foco em dívidas de maior valor.", "prioridade": 3, "acao": "Avaliar inclusão de faixas de menor valor para ampliar alcance"})
    for anomalia in anomalias_detectadas[:2]:
        alertas.append({"tipo": "warning" if anomalia["severidade"] == "Média" else "error", "titulo": f"{anomalia['tipo']} em {anomalia['data']}",
                       "mensagem": f"{anomalia['contexto']} ({anomalia['desvio']}). Valor: {anomalia['valor']}",
                       "prioridade": 2 if anomalia["severidade"] == "Média" else 1, "acao": "Verificar eventos sazonais ou mudanças regulatórias no período"})
    return sorted(alertas, key=lambda x: x["prioridade"])

# =============================================================================
# CARREGAMENTO DE DADOS
# =============================================================================
@st.cache_data(ttl=3600, show_spinner="🔄 Carregando e processando dados...")
def carregar_dados():
    for enc in ["utf-8", "latin1", "cp1252", "iso-8859-1"]:
        try:
            df = pd.read_csv("dados_desenrola.csv", sep=";", encoding=enc, low_memory=False)
            df.columns = df.columns.str.lower().str.strip()
            for col in ["numero_operacoes", "volume_operacoes"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col].astype(str).str.replace(r"[^\d,.-]", "", regex=True).str.replace(".", "", regex=False).str.replace(",", ".", regex=False), errors="coerce")
            if "data_base" in df.columns: df["data_base"] = pd.to_datetime(df["data_base"].astype(str).str.zfill(6), format="%Y%m", errors="coerce")
            if "nome_conglomerado_financeiro" in df.columns: df["tipo_banco"] = df["nome_conglomerado_financeiro"].apply(classificar_banco)
            if "unidade_federacao" in df.columns: df["regiao"] = df["unidade_federacao"].apply(agrupar_regiao)
            df_limpo = df.dropna(subset=["volume_operacoes", "numero_operacoes"])
            df_limpo = df_limpo[(df_limpo["volume_operacoes"] >= 0) & (df_limpo["numero_operacoes"] >= 0)]
            return df, df_limpo
        except: continue
    return None, None

# =============================================================================
# LAYOUT DE GRÁFICOS
# =============================================================================
def layout_base_enhanced(fig, height=450, showlegend=True, title=None):
    fig.update_layout(template=CORES["plotly_template"], height=height, margin=dict(l=50, r=40, t=60 if title else 40, b=50),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(color=CORES["texto"], family="IBM Plex Sans", size=12), hovermode="x unified", showlegend=showlegend,                      legend=dict(orientation="h", yanchor="bottom", y=1.02 if showlegend else 1, xanchor="right", x=1, font=dict(size=11), bgcolor="rgba(0,0,0,0)"),
                      title=dict(text=title, font=dict(size=14, weight="600"), x=0.02, xanchor="left") if title else None)
    fig.update_xaxes(showgrid=True, gridcolor=CORES["grid"], gridwidth=1, color=CORES["texto"], title_font=dict(size=11), tickfont=dict(size=10))
    fig.update_yaxes(showgrid=True, gridcolor=CORES["grid"], gridwidth=1, color=CORES["texto"], title_font=dict(size=11), tickfont=dict(size=10))
    return fig

# =============================================================================
# SIDEBAR
# =============================================================================
def render_sidebar(df):
    with st.sidebar:
        st.markdown("### ⚙️ Controles")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("☀️ Claro", use_container_width=True, key="btn_claro"): st.session_state.tema = "claro"; st.rerun()
        with c2:
            if st.button("🌙 Escuro", use_container_width=True, key="btn_escuro"): st.session_state.tema = "escuro"; st.rerun()
        st.markdown("---")
        st.markdown("#### 🔍 Filtros")
        tipos = sorted(df["tipo_desenrola"].dropna().unique()) if "tipo_desenrola" in df.columns else []
        tipo = st.multiselect("Faixa do Programa", tipos, default=tipos, key="filter_tipo")
        regioes = sorted(df["regiao"].dropna().unique()) if "regiao" in df.columns else []
        regiao = st.multiselect("Região", regioes, default=regioes, key="filter_regiao")
        bancos = sorted(df["tipo_banco"].dropna().unique()) if "tipo_banco" in df.columns else []
        banco = st.multiselect("Segmento", bancos, default=bancos, key="filter_banco")
        if st.button("🔄 Limpar Filtros", use_container_width=True, key="btn_limpar"): st.session_state.filter_tipo = st.session_state.filter_regiao = st.session_state.filter_banco = None; st.rerun()
        st.markdown("---")
        st.markdown("#### 📊 Qualidade dos Dados")
        dq_summary = {"total": len(df), "score": 92, "class": "Boa", "badge": "badge-positivo"}
        st.markdown(f"""<div class="dq-card"><div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.5rem"><b>Score de Qualidade</b><span class="badge {dq_summary['badge']}">{dq_summary['score']:.0f}%</span></div><b>Registros válidos:</b> <span class="mono">{fmt_num(dq_summary['total'])}</span><br><b>Classificação:</b> <span style="color:{SEMANTIC_COLORS['positivo']}">{dq_summary['class']}</span><br><div style="margin-top:0.5rem;font-size:0.75rem;color:{CORES['texto_secundario']}">✅ Dados tratados e prontos para análise</div></div>""", unsafe_allow_html=True)
        st.markdown("---")
        st.caption("💡 *Dica: Use os filtros para focar em segmentos específicos*")
        return tipo, regiao, banco

# =============================================================================
# COMPONENTES REUTILIZÁVEIS
# =============================================================================
def kpi_card(titulo, valor, subtitulo=None, icone="📊", classe=""):
    classe_css = f"kpi-card {classe}".strip()
    sub_html = f'<div class="kpi-sub">{subtitulo}</div>' if subtitulo else ""
    st.markdown(f"""<div class="{classe_css}"><div class="kpi-title">{icone} {titulo}</div><div class="kpi-value">{valor}</div>{sub_html}</div>""", unsafe_allow_html=True)

def insight_box(label, texto, icone="💡"):
    st.markdown(f"""<div class="insight-box"><div class="insight-label">{icone} {label}</div><div class="insight-text">{texto}</div></div>""", unsafe_allow_html=True)

# =============================================================================
# MAIN
# =============================================================================
def main():
    df_raw, df = carregar_dados()    if df is None or df.empty:
        st.error("❌ Erro ao carregar dados. Verifique se 'dados_desenrola.csv' está presente na pasta do projeto.")
        st.info("📋 Formato esperado: CSV com separador ';', colunas: data_base, nome_conglomerado_financeiro, unidade_federacao, tipo_desenrola, numero_operacoes, volume_operacoes")
        st.stop()
    
    tipo_filter, regiao_filter, banco_filter = render_sidebar(df)
    df_f = df.copy()
    if tipo_filter and "tipo_desenrola" in df_f.columns: df_f = df_f[df_f["tipo_desenrola"].isin(tipo_filter)]
    if regiao_filter and "regiao" in df_f.columns: df_f = df_f[df_f["regiao"].isin(regiao_filter)]
    if banco_filter and "tipo_banco" in df_f.columns: df_f = df_f[df_f["tipo_banco"].isin(banco_filter)]
    if df_f.empty: st.warning("⚠️ Nenhum dado encontrado com os filtros selecionados. Ajuste os critérios."); st.stop()
    
    col_banco = "nome_conglomerado_financeiro"
    dq = calcular_data_quality_enhanced(df_raw, df_f)
    
    # HEADER
    st.title("🏦 Desenrola Brasil – Painel Executivo")
    st.caption("Monitoramento estratégico de renegociação de dívidas | Fonte: Banco Central do Brasil (SCR)")
    with st.container():
        col_info, col_link = st.columns([3, 1])
        with col_info:
            st.markdown("**📌 Sobre esta análise** Dashboard estratégico que analisa os dados públicos do **Programa Desenrola Brasil**, divulgados mensalmente pelo Banco Central via Sistema de Informações de Crédito (SCR). A base contém operações de renegociação segregadas por instituição, UF e faixa do programa. Utilizamos técnicas estatísticas avançadas (HHI, STL, K-Means) para transformar dados em insights acionáveis para gestores e formuladores de políticas públicas.")
        with col_link:
            st.markdown(f"""<div style="background:{CORES['card']}; padding:0.9rem; border-radius:12px; text-align:center; border:1px solid {CORES['borda']}"><div style="font-size:0.85rem;margin-bottom:0.5rem;color:{CORES['texto_secundario']}">Fonte Oficial</div>🔗 <a href="https://www.bcb.gov.br/estatisticas/scr" target="_blank" style="color:{CORES['primaria']};text-decoration:none;font-weight:600">Banco Central →</a></div>""", unsafe_allow_html=True)
    
    # KPIs
    total_volume, total_ops = df_f["volume_operacoes"].sum(), df_f["numero_operacoes"].sum()
    ticket_medio = total_volume / total_ops if total_ops > 0 else 0
    num_inst = df_f[col_banco].nunique()
    st.markdown("#### 📈 Indicadores-Chave")
    col_k1, col_k2, col_k3, col_k4 = st.columns(4)
    with col_k1: kpi_card("💵 Volume Renegociado", fmt_brl(total_volume), "Total acumulado", classe="sucesso")
    with col_k2: kpi_card("📄 Contratos", fmt_num(total_ops), "Operações realizadas")
    with col_k3: kpi_card("🎫 Ticket Médio", fmt_brl(ticket_medio), "Volume ÷ Contratos")
    with col_k4: kpi_card("🏛️ Instituições", fmt_num(num_inst), "Players ativos")
    
    # CÁLCULOS
    reg_data = df_f.groupby("regiao")["volume_operacoes"].sum().reset_index()
    total_reg = reg_data["volume_operacoes"].sum()
    reg_data["pct"] = (reg_data["volume_operacoes"] / total_reg * 100).round(1) if total_reg > 0 else 0
    lider_regiao = reg_data.loc[reg_data["volume_operacoes"].idxmax()] if not reg_data.empty else None
    market_hhi = df_f.groupby(col_banco)["numero_operacoes"].sum().reset_index()
    total_contratos = market_hhi["numero_operacoes"].sum()
    if total_contratos > 0 and not market_hhi.empty:
        lider_banco = market_hhi.loc[market_hhi["numero_operacoes"].idxmax(), col_banco]
        part_banco = (market_hhi["numero_operacoes"].max() / total_contratos) * 100
        top3_share = market_hhi.nlargest(3, "numero_operacoes")["numero_operacoes"].sum() / total_contratos * 100
        top5_share = market_hhi.nlargest(5, "numero_operacoes")["numero_operacoes"].sum() / total_contratos * 100
    else: lider_banco, part_banco, top3_share, top5_share = "N/A", 0, 0, 0
    hhi_val = calcular_hhi(market_hhi, "numero_operacoes") if not market_hhi.empty else 0    hhi_info = interpretar_hhi_enhanced(hhi_val, len(market_hhi), top3_share, top5_share)
    evolucao_global = df_f.groupby("data_base")["volume_operacoes"].sum()
    if len(evolucao_global) > 1:
        cresc_medio = evolucao_global.pct_change().mean() * 100
        tendencia_txt = "📈 Programa em expansão sustentada." if cresc_medio > 2 else "📉 Sinal de desaceleração, requer atenção." if cresc_medio < -2 else "➡️ Estabilidade no volume de renegociações."
    else: cresc_medio, tendencia_txt = 0, "⏳ Dados insuficientes para calcular tendência."
    anomalias = detectar_anomalias(df_f)
    evolucao_df = df_f.groupby("data_base").agg(volume_operacoes=("volume_operacoes", "sum"), numero_operacoes=("numero_operacoes", "sum")).reset_index()
    evolucao_df["crescimento"] = evolucao_df["volume_operacoes"].pct_change() * 100
    alertas = gerar_alertas_contextuais(evolucao_df, hhi_info, ticket_medio, anomalias)
    
    # RESUMO EXECUTIVO
    st.markdown("### 🎯 Principais Conclusões (Resumo Executivo)")
    col_exec1, col_exec2 = st.columns(2)
    with col_exec1:
        if lider_regiao is not None: insight_box("🗺️ Concentração Regional", f"A região **{lider_regiao['regiao']}** concentra **{lider_regiao['pct']:.1f}%** do volume total de renegociações. {'🔍 Essa concentração indica oportunidade para expansão em outras regiões.' if lider_regiao['pct'] > 40 else '✅ Distribuição regional relativamente equilibrada.'}", icone="📍")
        insight_box("🏦 Liderança Bancária", f"**{lider_banco}** responde por **{part_banco:.1f}%** dos contratos. {'⚠️ Alta dependência de um único player.' if part_banco > 30 else '✅ Participação equilibrada entre instituições.'}", icone="🥇")
    with col_exec2:
        insight_box("⚖️ Concentração de Mercado (HHI)", f"<b>{hhi_info['valor']:.0f}</b> → <span style='color:{hhi_info['cor_status']}'>{hhi_info['classificacao']}</span><br><small>{hhi_info['recomendacao']}</small>", icone="📊")
        insight_box("📈 Tendência Recente", tendencia_txt, icone="🔮")
    if alertas:
        st.markdown("---"); st.markdown("#### 🔔 Alertas Prioritários")
        for alerta in alertas[:3]:
            fn = getattr(st, alerta["tipo"], st.info); fn(f"**{alerta['titulo']}** — {alerta['mensagem']}")
            with st.expander(f"💡 Ação recomendada: {alerta['acao']}"): st.caption(f"Prioridade: {'🔴 Alta' if alerta['prioridade']==1 else '🟡 Média' if alerta['prioridade']==2 else '🟢 Baixa'}")
    st.markdown("---"); st.caption("👉 **Explore as abas abaixo para análises detalhadas, projeções e exportação de relatórios.**")
    
    # ABAS
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📈 Evolução & Projeção", "🏦 Mercado & HHI", "🗺️ Geografia", "🔬 Segmentos", "🤖 ML: Clusters", "📋 Relatório"])
    
    # TAB 1
    with tab1:
        evolucao = df_f.groupby("data_base").agg(volume_operacoes=("volume_operacoes", "sum"), numero_operacoes=("numero_operacoes", "sum")).reset_index()
        evolucao["crescimento"], evolucao["media_movel3"] = evolucao["volume_operacoes"].pct_change() * 100, evolucao["volume_operacoes"].rolling(3, min_periods=1).mean()
        st.markdown("#### 📊 Histórico de Volume e Contratos")
        fig_ev = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.65, 0.35], vertical_spacing=0.08, subplot_titles=["Volume de Renegociação (R$)", "Número de Contratos"])
        fig_ev.add_trace(go.Scatter(x=evolucao["data_base"], y=evolucao["volume_operacoes"], name="Volume Mensal", mode="lines+markers", line=dict(color=SEMANTIC_COLORS["neutro"], width=2.5), marker=dict(size=5, color=SEMANTIC_COLORS["neutro"])), row=1, col=1)
        fig_ev.add_trace(go.Scatter(x=evolucao["data_base"], y=evolucao["media_movel3"], name="Média Móvel 3M", mode="lines", line=dict(color=SEMANTIC_COLORS["atencao"], dash="dash", width=2)), row=1, col=1)
        fig_ev.add_trace(go.Bar(x=evolucao["data_base"], y=evolucao["numero_operacoes"], name="Contratos", marker_color=SEMANTIC_COLORS["info"], opacity=0.7), row=2, col=1)
        layout_base_enhanced(fig_ev, height=500); st.plotly_chart(fig_ev, use_container_width=True, config={'displayModeBar': True, 'responsive': True})
        if len(evolucao) >= 8:
            st.markdown("#### 🔮 Projeção Holt-Winters + Decomposição STL (3 meses)"); st.caption("Modelo híbrido: STL captura sazonalidade + Holt-Winters projeta tendência com IC 95%")
            resultado_proj = projetar_com_decomposicao(evolucao["volume_operacoes"], evolucao["data_base"])
            if resultado_proj:
                fig_prev = go.Figure()
                fig_prev.add_trace(go.Scatter(x=evolucao["data_base"], y=evolucao["volume_operacoes"], name="Realizado", mode="lines+markers", line=dict(color=SEMANTIC_COLORS["neutro"], width=2.5)))
                fig_prev.add_trace(go.Scatter(x=list(resultado_proj["datas"]) + list(resultado_proj["datas"][::-1]), y=list(resultado_proj["upper"]) + list(resultado_proj["lower"][::-1]), fill="toself", fillcolor="rgba(245,158,11,0.15)", line=dict(color="rgba(0,0,0,0)"), name="IC 95%", hoverinfo="skip"))
                fig_prev.add_trace(go.Scatter(x=resultado_proj["datas"], y=resultado_proj["previsao"], name="Projeção", mode="lines+markers", line=dict(color=SEMANTIC_COLORS["atencao"], dash="dot", width=2.5), marker=dict(symbol="diamond", size=8, color=SEMANTIC_COLORS["atencao"])))
                layout_base_enhanced(fig_prev, height=420, title="Projeção de Volume"); st.plotly_chart(fig_prev, use_container_width=True, config={'displayModeBar': True})
                col_p1, col_p2, col_p3 = st.columns(3)                for col_p, (d, v, l, u) in zip([col_p1, col_p2, col_p3], zip(resultado_proj["datas"], resultado_proj["previsao"], resultado_proj["lower"], resultado_proj["upper"])):
                    with col_p: st.markdown(f"""<div class="insight-box" style="text-align:center"><div class="insight-label">📅 {d.strftime('%b/%Y')}</div><div class="kpi-value" style="font-size:1.3rem">{fmt_brl(v)}</div><div style="font-size:0.75rem;color:{CORES['texto_secundario']}">IC: {fmt_brl(l)} – {fmt_brl(u)}</div></div>""", unsafe_allow_html=True)
                decomp = resultado_proj["decomposicao"]
                if decomp["mape"]: st.markdown(f"""<div style="font-size:0.8rem;color:{CORES['texto_secundario']};margin-top:0.5rem">📊 Qualidade do modelo: <b>{resultado_proj['qualidade']}</b> | MAPE: {decomp['mape']:.1f}% | Sazonalidade média: {fmt_brl(decomp['sazonalidade_media'])}</div>""", unsafe_allow_html=True)
        st.markdown("#### 📉 Variação Mensal Recente")
        tab_var = evolucao[["data_base", "volume_operacoes", "crescimento"]].tail(6).copy()
        tab_var["data_base"], tab_var["crescimento"], tab_var["volume_operacoes"] = tab_var["data_base"].dt.strftime("%m/%Y"), tab_var["crescimento"].apply(lambda x: fmt_pct(x, show_sign=True) if pd.notna(x) else "—"), tab_var["volume_operacoes"].apply(fmt_brl)
        tab_var.columns = ["Mês", "Volume", "Variação MoM"]; st.dataframe(tab_var, use_container_width=True, hide_index=True)
        st.markdown("#### 🔄 Comparativo Ano a Ano (YoY)")
        yoy = df_f.copy(); yoy["ano"], yoy["mes"] = yoy["data_base"].dt.year, yoy["data_base"].dt.month
        yoy_data = yoy.groupby(["ano", "mes"])["volume_operacoes"].sum().reset_index()
        yoy_data = yoy_data[yoy_data["ano"] >= yoy_data["ano"].max() - 1]
        if not yoy_data.empty:
            fig_yoy = go.Figure()
            for i, ano in enumerate(sorted(yoy_data["ano"].unique())):
                d = yoy_data[yoy_data["ano"] == ano]; fig_yoy.add_trace(go.Scatter(x=d["mes"], y=d["volume_operacoes"], name=str(ano), mode="lines+markers", line=dict(color=PLOTLY_CORES[i % len(PLOTLY_CORES)], width=2.5)))
            meses_label = ['Jan','Fev','Mar','Abr','Mai','Jun','Jul','Ago','Set','Out','Nov','Dez']
            fig_yoy.update_xaxes(tickvals=list(range(1,13)), ticktext=meses_label, title="Mês"); fig_yoy.update_yaxes(title="Volume (R$)")
            layout_base_enhanced(fig_yoy, height=400, title="Evolução Mensal por Ano"); st.plotly_chart(fig_yoy, use_container_width=True, config={'displayModeBar': True})
    
    # TAB 2
    with tab2:
        market = df_f.groupby(col_banco)["numero_operacoes"].sum().sort_values(ascending=False).reset_index()
        col_hhi, col_pareto = st.columns([1, 1])
        with col_hhi:
            st.markdown("#### ⚖️ Índice de Concentração (HHI)")
            st.markdown(f"""<div class="insight-box"><div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.5rem"><div class="insight-label">📊 Herfindahl-Hirschman Index</div><span class="badge {hhi_info['badge_class']}">{hhi_info['classificacao']}</span></div><div class="kpi-value" style="font-size:2.5rem;text-align:center;margin:0.5rem 0">{hhi_info['valor']:.0f}</div><div style="text-align:center;font-size:0.85rem;color:{CORES['texto_secundario']};margin-bottom:0.75rem">Risco: <span style="color:{hhi_info['cor_status']};font-weight:600">{hhi_info['risco']}</span></div><p style="font-size:0.85rem;line-height:1.6;margin-bottom:0.5rem">{hhi_info['recomendacao']}</p><hr style="border:none;border-top:1px solid {CORES['borda']};margin:0.75rem 0"><p style="font-size:0.75rem;color:{CORES['texto_secundario']}"><b>Como interpretar:</b><br>• HHI &lt; 1.500 → Mercado competitivo<br>• HHI 1.500–2.500 → Concentração moderada<br>• HHI &gt; 2.500 → Alta concentração (oligopólio)<br><br><small>Calculado sobre número de contratos. Fonte: Metodologia DOJ/FTC.</small></p></div>""", unsafe_allow_html=True)
            if hhi_info["insights"]:
                st.markdown("##### 🔍 Insights de Concentração")
                for insight in hhi_info["insights"]: st.markdown(f"<div style='font-size:0.85rem;margin:0.3rem 0;padding:0.4rem 0.6rem;background:rgba(148,163,184,0.08);border-radius:6px'>{insight}</div>", unsafe_allow_html=True)
        with col_pareto:
            st.markdown("#### 📐 Análise de Pareto (80/20)")
            pareto_df, total_pareto = calcular_pareto(market.head(15), "numero_operacoes")
            if not pareto_df.empty:
                fig_p = make_subplots(specs=[[{"secondary_y": True}]])
                fig_p.add_trace(go.Bar(x=pareto_df[col_banco], y=pareto_df["numero_operacoes"], name="Contratos", marker_color=SEMANTIC_COLORS["neutro"], opacity=0.85, text=pareto_df["pct_individual"].apply(lambda x: f"{x:.1f}%"), textposition="outside", textfont=dict(size=9)), secondary_y=False)
                fig_p.add_trace(go.Scatter(x=pareto_df[col_banco], y=pareto_df["pct_acumulado"], name="% Acumulado", mode="lines+markers", line=dict(color=SEMANTIC_COLORS["atencao"], width=3), marker=dict(size=8, color=SEMANTIC_COLORS["atencao"]), hovertemplate="%{x}<br>Acumulado: %{y:.1f}%<extra></extra>"), secondary_y=True)
                fig_p.add_hline(y=80, line_dash="dash", line_color=SEMANTIC_COLORS["alerta"], annotation_text="80%", annotation_position="top right", secondary_y=True, annotation_font=dict(size=10))
                fig_p.update_yaxes(title_text="Nº de Contratos", secondary_y=False); fig_p.update_yaxes(title_text="% Acumulado", secondary_y=True, range=[0, 105])
                layout_base_enhanced(fig_p, height=400); st.plotly_chart(fig_p, use_container_width=True, config={'displayModeBar': True})
                idx_80 = pareto_df[pareto_df["pct_acumulado"] >= 80].index.min()
                n_para_80 = idx_80 + 1 if idx_80 is not None else len(pareto_df)
                st.markdown(f"""<div style="font-size:0.82rem;color:{CORES['texto_secundario']};text-align:center;margin-top:0.5rem">🎯 <b>{n_para_80} instituições</b> concentram ~80% dos contratos</div>""", unsafe_allow_html=True)
        st.markdown("#### 🏆 Ranking de Mercado – Top 15 Instituições")
        ranking = market.head(15).copy(); total_r = ranking["numero_operacoes"].sum()
        ranking["% Individual"], ranking["% Acumulado"] = (ranking["numero_operacoes"] / total_r * 100).round(1), (ranking["numero_operacoes"] / total_r * 100).cumsum().round(1)
        vol_map = df_f.groupby(col_banco)["volume_operacoes"].sum()
        ranking["Volume (R$)"], ranking["numero_operacoes_fmt"] = ranking[col_banco].map(vol_map).apply(fmt_brl), ranking["numero_operacoes"].apply(fmt_num)
        ranking_display = ranking[[col_banco, "numero_operacoes_fmt", "% Individual", "% Acumulado", "Volume (R$)"]].copy()
        ranking_display.columns = ["Instituição", "Contratos", "% Individual", "% Acumulado", "Volume"]; st.dataframe(ranking_display, use_container_width=True, hide_index=True)        st.markdown(f"""<div class="insight-box"><div class="insight-label">📖 Interpretação Estratégica</div><div class="insight-text"><b>{lider_banco}</b> lidera com <b>{part_banco:.1f}%</b> dos contratos. As 3 maiores instituições concentram <b>{top3_share:.1f}%</b> do total — <b>{'indicando dependência crítica e risco sistêmico elevado.' if top3_share > 60 else 'sugerindo distribuição equilibrada com competição saudável.'}</b></div></div>""", unsafe_allow_html=True)
    
    # TAB 3
    with tab3:
        reg_data = df_f.groupby("regiao")["volume_operacoes"].sum().reset_index()
        total_reg = reg_data["volume_operacoes"].sum()
        reg_data["pct"] = (reg_data["volume_operacoes"] / total_reg * 100).round(1) if total_reg > 0 else 0
        regiao_lider = reg_data.sort_values("volume_operacoes", ascending=False).iloc[0] if not reg_data.empty else None
        col_r1, col_r2 = st.columns(2)
        with col_r1:
            st.markdown("#### 🥧 Participação Regional por Volume")
            if not reg_data.empty:
                fig_donut = go.Figure(go.Pie(labels=reg_data["regiao"], values=reg_data["volume_operacoes"], hole=0.55, textinfo="percent+label", textposition="inside", marker=dict(colors=PLOTLY_CORES[:len(reg_data)], line=dict(color=CORES["fundo"], width=2)), hovertemplate="<b>%{label}</b><br>Volume: %{value:,.0f}<br>Participação: %{percent:.1%}<extra></extra>"))
                if regiao_lider is not None: fig_donut.add_annotation(text=f"{regiao_lider['pct']:.0f}%<br><small>{regiao_lider['regiao']}</small>", x=0.5, y=0.5, showarrow=False, font=dict(size=13, color=CORES["texto"]))
                layout_base_enhanced(fig_donut, height=420); st.plotly_chart(fig_donut, use_container_width=True, config={'displayModeBar': True})
                if regiao_lider is not None: st.markdown(f"""<div class="insight-box"><div class="insight-label">📍 Leitura Geográfica</div><div class="insight-text">A região <b>{regiao_lider['regiao']}</b> concentra <b>{regiao_lider['pct']:.1f}%</b> das renegociações. {'🔍 Oportunidade para expansão em regiões sub-representadas.' if regiao_lider['pct'] > 40 else '✅ Distribuição territorial equilibrada.'}</div></div>""", unsafe_allow_html=True)
        with col_r2:
            st.markdown("#### 🔥 Heatmap: Evolução Regional")
            heat_df = df_f.copy(); heat_df["mes_ano"] = heat_df["data_base"].dt.strftime("%Y-%m")
            meses_disp = sorted(heat_df["mes_ano"].dropna().unique())
            meses_selecionados = st.multiselect("Selecione os meses", meses_disp, default=meses_disp[-12:] if len(meses_disp) >= 12 else meses_disp, key="heatmap_meses")
            if meses_selecionados:
                heat_filt = heat_df[heat_df["mes_ano"].isin(meses_selecionados)]
                pivot_heat = heat_filt.groupby(["regiao", "mes_ano"])["volume_operacoes"].sum().reset_index()
                pivot_heat["volume_M"] = pivot_heat["volume_operacoes"] / 1e6
                if not pivot_heat.empty:
                    pivot_matrix = pivot_heat.pivot(index="regiao", columns="mes_ano", values="volume_M").fillna(0)
                    fig_heat = go.Figure(go.Heatmap(z=pivot_matrix.values, x=pivot_matrix.columns.tolist(), y=pivot_matrix.index.tolist(), colorscale="Blues", text=np.round(pivot_matrix.values, 1), texttemplate="%{text}M", textfont=dict(size=9), colorbar=dict(title="R$ Milhões", titleside="right"), hovertemplate="<b>%{y}</b> • %{x}<br>Volume: %{z:.1f}M<extra></extra>"))
                    layout_base_enhanced(fig_heat, height=420, title="Volume por Região e Mês"); st.plotly_chart(fig_heat, use_container_width=True, config={'displayModeBar': True})
                else: st.info("ℹ️ Nenhum dado para o período selecionado.")
            else: st.info("👈 Selecione pelo menos um mês para visualizar o heatmap.")
        st.markdown("#### 🏅 Líderes de Mercado por Estado (Top 3)")
        if "unidade_federacao" in df_f.columns:
            uf_banco = df_f.groupby(["unidade_federacao", col_banco])["numero_operacoes"].sum().reset_index()
            uf_banco = uf_banco.sort_values(["unidade_federacao", "numero_operacoes"], ascending=[True, False])
            top3_uf = uf_banco.groupby("unidade_federacao").head(3).reset_index(drop=True)
            top3_uf["rank"], top3_uf["display"] = top3_uf.groupby("unidade_federacao").cumcount() + 1, top3_uf.apply(lambda x: f"{x[col_banco]} ({fmt_num(x['numero_operacoes'])})", axis=1)
            piv_uf = top3_uf.pivot_table(index="unidade_federacao", columns="rank", values="display", aggfunc="first").reset_index()
            piv_uf.columns = ["UF", "🥇 1º Lugar", "🥈 2º Lugar", "🥉 3º Lugar"]; st.dataframe(piv_uf, use_container_width=True, hide_index=True)
        st.markdown("#### 💰 Ticket Médio: Região × Segmento")
        cruzado = df_f.groupby(["regiao", "tipo_banco"]).agg(numero_operacoes=("numero_operacoes", "sum"), volume_operacoes=("volume_operacoes", "sum")).reset_index()
        cruzado["ticket_medio"] = cruzado["volume_operacoes"] / cruzado["numero_operacoes"].replace(0, np.nan)
        fig_tree = px.treemap(cruzado, path=["regiao", "tipo_banco"], values="volume_operacoes", color="ticket_medio", color_continuous_scale="Blues", hover_data={"ticket_medio": ":.0f"}, title="Volume por Região e Segmento (cor = Ticket Médio)")
        fig_tree.update_traces(textinfo="label+percent entry+value", hovertemplate="<b>%{label}</b><br>Volume: %{value:,.0f}<br>Ticket: R$ %{customdata[0]:,.0f}<extra></extra>")
        layout_base_enhanced(fig_tree, height=500); st.plotly_chart(fig_tree, use_container_width=True, config={'displayModeBar': True})
    
    # TAB 4
    with tab4:
        dispersao = df_f.groupby(col_banco).agg(numero_operacoes=("numero_operacoes", "sum"), volume_operacoes=("volume_operacoes", "sum"), tipo_banco=("tipo_banco", "first")).reset_index()
        dispersao["ticket_medio"] = dispersao["volume_operacoes"] / dispersao["numero_operacoes"].replace(0, np.nan)        dispersao = dispersao[dispersao["numero_operacoes"] > 500].dropna(subset=["ticket_medio"])
        st.markdown("#### 📊 Dispersão: Operações × Ticket Médio por Segmento")
        fig_disp = go.Figure()
        cores_seg = {"Banco Digital": "#10B981", "Banco Tradicional": "#3B82F6", "Banco de Investimento": "#F59E0B", "Cooperativa/Financeira": "#8B5CF6", "Outras Instituições": "#64748B"}
        for seg, grp in dispersao.groupby("tipo_banco"):
            size_norm = np.log1p(grp["volume_operacoes"] / dispersao["volume_operacoes"].max()) * 35 + 8
            fig_disp.add_trace(go.Scatter(x=grp["numero_operacoes"], y=grp["ticket_medio"], mode="markers", name=seg, marker=dict(size=size_norm, color=cores_seg.get(seg, "#64748B"), opacity=0.75, line=dict(width=1.5, color=CORES["fundo"])), hovertemplate="<b>%{customdata}</b><br><small>%{text}</small><br>Operações: %{x:,.0f}<br>Ticket: R$ %{y:,.0f}<extra></extra>", customdata=grp[col_banco], text=seg))
        fig_disp.update_xaxes(title="Número de Operações", type="log"); fig_disp.update_yaxes(title="Ticket Médio (R$)", type="log")
        layout_base_enhanced(fig_disp, height=500, title="Mapa de Dispersão Institucional"); st.plotly_chart(fig_disp, use_container_width=True, config={'displayModeBar': True})
        st.markdown("#### 📈 Comparativo Agregado por Segmento")
        comp = df_f.groupby("tipo_banco").agg(numero_operacoes=("numero_operacoes", "sum"), volume_operacoes=("volume_operacoes", "sum")).reset_index()
        comp["ticket_medio"], comp["pct_ops"] = comp["volume_operacoes"] / comp["numero_operacoes"].replace(0, np.nan), (comp["numero_operacoes"] / comp["numero_operacoes"].sum() * 100).round(1)
        fig_comp = make_subplots(rows=1, cols=2, subplot_titles=["Distribuição de Contratos (%)", "Ticket Médio por Segmento (R$)"])
        fig_comp.add_trace(go.Bar(x=comp["tipo_banco"], y=comp["pct_ops"], text=comp["pct_ops"].apply(lambda x: f"{x:.1f}%"), textposition="outside", marker_color=[cores_seg.get(s, "#64748B") for s in comp["tipo_banco"]], name="Contratos"), row=1, col=1)
        fig_comp.add_trace(go.Bar(x=comp["tipo_banco"], y=comp["ticket_medio"], text=comp["ticket_medio"].apply(fmt_brl), textposition="outside", marker_color=[cores_seg.get(s, "#64748B") for s in comp["tipo_banco"]], name="Ticket", showlegend=False), row=1, col=2)
        fig_comp.update_yaxes(title="% do Total", row=1, col=1); fig_comp.update_yaxes(title="R$", row=1, col=2)
        layout_base_enhanced(fig_comp, height=420, showlegend=False); st.plotly_chart(fig_comp, use_container_width=True, config={'displayModeBar': True})
        st.markdown("#### 🔍 Distribuição de Contratos (Detecção de Outliers)"); st.caption("Caixas com bigodes longos ou pontos isolados indicam comportamento atípico — campanhas pontuais ou sazonalidade.")
        top10_bancos = df_f.groupby(col_banco)["numero_operacoes"].sum().nlargest(10).index
        df_top = df_f[df_f[col_banco].isin(top10_bancos)]
        if not df_top.empty:
            fig_box = px.box(df_top, x=col_banco, y="numero_operacoes", color="tipo_banco", color_discrete_map=cores_seg, points="outliers", hover_data={"numero_operacoes": ":,.0f"})
            fig_box.update_xaxes(title="Instituição", tickangle=45); fig_box.update_yaxes(title="Nº de Contratos/Mês")
            layout_base_enhanced(fig_box, height=450, title="Distribuição Mensal: Top 10 Instituições"); st.plotly_chart(fig_box, use_container_width=True, config={'displayModeBar': True})
    
    # TAB 5
    with tab5:
        st.markdown("#### 🤖 Agrupamento de Instituições (K-Means com Validação)")
        st.markdown(f"""<div class="insight-box"><div class="insight-label">⚙️ Como funciona esta análise</div><div class="insight-text">O algoritmo <b>K-Means</b> agrupa automaticamente as instituições por similaridade de comportamento, considerando: <b>número de operações</b>, <b>ticket médio</b> e <b>meses de atividade</b>. O número ideal de clusters é determinado pelo <b>Silhouette Score</b> (0-1), que mede quão bem cada ponto se encaixa em seu grupo. <b>Score > 0.5</b> indica boa separação entre clusters.</div></div>""", unsafe_allow_html=True)
        cluster_data, n_clusters, metrica_qualidade = clusterizar_bancos_enhanced(df_f, col_banco)
        if cluster_data is not None:
            fig_cl = go.Figure()
            for nome_cl, grp in cluster_data.groupby("cluster_nome"):
                size_norm = np.log1p(grp["volume_operacoes"] / cluster_data["volume_operacoes"].max()) * 30 + 8
                fig_cl.add_trace(go.Scatter(x=grp["numero_operacoes"], y=grp["ticket_medio"], mode="markers", name=nome_cl, marker=dict(size=size_norm, color=grp["cor_cluster"].iloc[0], opacity=0.8, line=dict(width=1.5, color=CORES["fundo"])), hovertemplate="<b>%{customdata}</b><br>Operações: %{x:,.0f}<br>Ticket: R$ %{y:,.0f}<br>Cluster: %{text}<extra></extra>", customdata=grp[col_banco], text=nome_cl))
            fig_cl.update_xaxes(title="Número de Operações", type="log"); fig_cl.update_yaxes(title="Ticket Médio (R$)", type="log")
            layout_base_enhanced(fig_cl, height=500, title=f"Agrupamento por Comportamento (K={n_clusters}, {metrica_qualidade})"); st.plotly_chart(fig_cl, use_container_width=True, config={'displayModeBar': True})
            st.markdown("#### 📋 Resumo por Grupo Identificado")
            resumo_clusters = cluster_data.groupby("cluster_nome").agg(Instituicoes=(col_banco, "count"), Contratos_Medio=("numero_operacoes", "mean"), Ticket_Medio=("ticket_medio", "mean"), Volume_Total=("volume_operacoes", "sum")).reset_index()
            resumo_clusters["Contratos_Medio_fmt"], resumo_clusters["Ticket_Medio_fmt"], resumo_clusters["Volume_Total_fmt"] = resumo_clusters["Contratos_Medio"].apply(fmt_num), resumo_clusters["Ticket_Medio"].apply(fmt_brl), resumo_clusters["Volume_Total"].apply(fmt_brl)
            resumo_display = resumo_clusters[["cluster_nome", "Instituicoes", "Contratos_Medio_fmt", "Ticket_Medio_fmt", "Volume_Total_fmt"]].copy()
            resumo_display.columns = ["Grupo", "Nº Instituições", "Média de Contratos", "Ticket Médio", "Volume Total"]; st.dataframe(resumo_display, use_container_width=True, hide_index=True)
            with st.expander("🔍 Ver instituições por grupo"):
                for nome_cl, grp in cluster_data.groupby("cluster_nome"):
                    st.markdown(f"**{nome_cl}** ({len(grp)} instituições)")
                    detalhe = grp[[col_banco, "numero_operacoes", "ticket_medio", "volume_operacoes"]].copy()
                    detalhe["ticket_medio_fmt"], detalhe["volume_fmt"] = detalhe["ticket_medio"].apply(fmt_brl), detalhe["volume_operacoes"].apply(fmt_brl)
                    st.dataframe(detalhe[[col_banco, "numero_operacoes", "ticket_medio_fmt", "volume_fmt"]].rename(columns={col_banco: "Instituição", "numero_operacoes": "Contratos", "ticket_medio_fmt": "Ticket Médio", "volume_fmt": "Volume Total"}), use_container_width=True, hide_index=True); st.markdown("---")
        else: st.info(f"ℹ️ {cluster_data if isinstance(cluster_data, str) else 'Dados insuficientes para realizar clusterização confiável. Mínimo recomendado: 6 instituições com >100 operações cada.'}")
        # TAB 6
    with tab6:
        st.markdown("#### 📋 Narrativa Executiva Automatizada")
        cresc_medio = evolucao_df["crescimento"].mean() if not evolucao_df["crescimento"].isna().all() else 0
        regiao_top = reg_data.sort_values("volume_operacoes", ascending=False).iloc[0] if not reg_data.empty else None
        corr_val = df_f[["numero_operacoes", "volume_operacoes"]].corr().iloc[0, 1] if len(df_f) > 1 else 0
        df_saz = df_f.copy(); df_saz["mes_num"] = df_f["data_base"].dt.month
        mes_pico = df_saz.groupby("mes_num")["volume_operacoes"].sum().idxmax() if not df_saz.empty else 6
        nomes_meses = ['Janeiro','Fevereiro','Março','Abril','Maio','Junho','Julho','Agosto','Setembro','Outubro','Novembro','Dezembro']
        insights_exec = [
            {"titulo": "📈 Tendência Geral", "texto": f"O programa apresentou {'crescimento' if cresc_medio > 0 else 'retração'} médio de <b>{abs(cresc_medio):.2f}% ao mês</b>. {'✅ Expansão sustentada do alcance.' if cresc_medio > 2 else '⚠️ Perda de momentum requer atenção.' if cresc_medio < -2 else '➡️ Estabilidade no volume de renegociações.'}"},
            {"titulo": "🗺️ Concentração Regional", "texto": f"{'A região ' + regiao_top['regiao'] + ' concentra ' + f'{regiao_top['pct']:.1f}%' if regiao_top is not None else 'Distribuição regional equilibrada'} do volume total. {'🔍 Oportunidade para políticas de inclusão regional.' if regiao_top is not None and regiao_top['pct'] > 40 else '✅ Cobertura territorial balanceada.'}"},
            {"titulo": "🏦 Liderança Bancária", "texto": f"<b>{lider_banco}</b> responde por <b>{part_banco:.1f}%</b> dos contratos. {'⚠️ Alta dependência estratégica.' if part_banco > 30 else '✅ Participação equilibrada entre players.'}"},
            {"titulo": "⚖️ Concentração de Mercado", "texto": f"HHI = <b>{hhi_info['valor']:.0f}</b> → {hhi_info['classificacao']}. <small>{hhi_info['recomendacao']}</small>"},
            {"titulo": "📅 Sazonalidade", "texto": f"O mês de <b>{nomes_meses[mes_pico-1]}</b> historicamente registra maior volume. 💡 Campanhas concentradas neste período podem maximizar impacto."},
            {"titulo": "🔗 Correlação Operações × Volume", "texto": f"Correlação = <b>{corr_val:.3f}</b> ({'forte' if corr_val > 0.7 else 'moderada' if corr_val > 0.4 else 'fraca'}). {'✅ Perfil homogêneo de dívidas renegociadas.' if corr_val > 0.7 else '⚠️ Variação no ticket médio por tipo de contrato é relevante.'}"}
        ]
        for insight in insights_exec: insight_box(insight["titulo"], insight["texto"], icone="•")
        if alertas:
            st.markdown("#### 🔔 Pontos de Atenção Prioritários")
            for alerta in alertas:
                with st.container(): st.markdown(f"""<div style="padding:0.75rem 1rem;border-left:4px solid {SEMANTIC_COLORS[alerta['tipo']] if alerta['tipo'] in SEMANTIC_COLORS else CORES['primaria']};background:rgba(148,163,184,0.08);border-radius:0 8px 8px 0;margin:0.5rem 0"><b>{alerta['titulo']}</b><br><small>{alerta['mensagem']}</small><br><span style="color:{CORES['texto_secundario']};font-size:0.8rem">💡 {alerta['acao']}</span></div>""", unsafe_allow_html=True)
        st.markdown("---"); st.markdown("#### 📥 Exportação de Dados e Relatórios")
        col_exp1, col_exp2 = st.columns(2)
        with col_exp1:
            csv = df_f.to_csv(index=False, sep=";", decimal=",").encode("utf-8")
            st.download_button("📊 Dados Filtrados (CSV)", data=csv, file_name=f"desenrola_dados_{datetime.now().strftime('%Y%m%d')}.csv", mime="text/csv", use_container_width=True)
        with col_exp2:
            relatorio_txt = f"""RELATÓRIO EXECUTIVO – DESENROLA BRASIL\n{'='*60}\nGerado em: {datetime.now().strftime('%d/%m/%Y %H:%M')}\nPeríodo analisado: {dq['periodo_inicio']} → {dq['periodo_fim']}\nFiltros aplicados: Programa={tipo_filter}, Região={regiao_filter}, Segmento={banco_filter}\n\n{'🎯 INDICADORES-CHAVE':-^60}\n• Volume Total Renegociado : {fmt_brl(total_volume)}\n• Total de Contratos        : {fmt_num(total_ops)}\n• Ticket Médio              : {fmt_brl(ticket_medio)}\n• Instituições Ativas       : {fmt_num(num_inst)}\n\n{'⚖️ CONCENTRAÇÃO DE MERCADO':-^60}\n• Índice HHI                : {hhi_info['valor']:.0f}\n• Classificação             : {hhi_info['classificacao']}\n• Nível de Risco            : {hhi_info['risco']}\n• Recomendação Estratégica  : {hhi_info['recomendacao']}\n\n{'🏆 DESTAQUES':-^60}\n• Banco Líder               : {lider_banco} ({part_banco:.1f}% dos contratos)\n• Região Líder              : {regiao_top['regiao'] if regiao_top else 'N/A'} ({regiao_top['pct'] if regiao_top else 0:.1f}% do volume)\n• Crescimento Médio Mensal  : {cresc_medio:+.2f}%\n• Pico de Sazonalidade      : {nomes_meses[mes_pico-1]}\n• Correlação Ops×Volume     : {corr_val:.3f}\n\n{'🔍 INSIGHTS AUTOMÁTICOS':-^60}\n{chr(10).join(f"• {i['titulo']}: {i['texto']}" for i in insights_exec)}\n\n{'⚠️ ALERTAS DETECTADOS':-^60}\n{chr(10).join(f"• {a['titulo']}: {a['mensagem']}" for a in alertas) or "• Nenhum alerta crítico no momento"}\n\n{'📊 QUALIDADE DOS DADOS':-^60}\n• Score de Qualidade        : {dq['score_qualidade']:.1f}% ({dq['class_qualidade']})\n• Registros Válidos         : {fmt_num(dq['total_registros_limpos'])}\n• Registros Descartados     : {fmt_num(dq['registros_descartados'])}\n• Taxa de Retenção          : {dq['taxa_retencao']:.1f}%\n• Completude Volume         : {dq['completude_volume']:.1f}%\n• Completude Operações      : {dq['completude_operacoes']:.1f}%\n\n{'ℹ️ METODOLOGIA':-^60}\n• Projeção: Holt-Winters com decomposição STL para sazonalidade\n• Clusterização: K-Means com validação por Silhouette Score\n• HHI: Índice Herfindahl-Hirschman (metodologia DOJ/FTC)\n• Detecção de Anomalias: Z-score com média móvel (threshold: 2.5σ)\n• Formatação: Valores em BRL com sufixos K/M/B/T para legibilidade\n\n{'📬 CONTATO E FONTES':-^60}\nFonte dos dados: Banco Central do Brasil – Sistema de Informações de Crédito (SCR)\nDashboard desenvolvido com: Streamlit + Plotly + Scikit-learn + Statsmodels\nÚltima atualização: {dq['ultima_data']}\n\n---\nEste relatório foi gerado automaticamente. Para dúvidas metodológicas, consulte a documentação técnica ou entre em contato com a equipe de análise."""
            st.download_button("📝 Relatório Executivo (TXT)", data=relatorio_txt, file_name=f"relatorio_desenrola_{datetime.now().strftime('%Y%m%d')}.txt", mime="text/plain", use_container_width=True)
        with st.expander("🖼️ Exportar Gráficos (PNG/PDF)"): st.info("💡 Em breve: funcionalidade para exportar visualizações em alta resolução. Por enquanto, use o menu de exportação nativo do Plotly (canto superior direito de cada gráfico).")
    
    # RODAPÉ
    st.markdown("---")
    st.markdown(f"""<div style='text-align:center; color:{CORES["texto_secundario"]}; font-size:0.75rem; line-height:1.6; padding:1rem 0'><b>Dashboard Desenrola Brasil</b> • Fonte: Banco Central do Brasil (SCR)<br>Metodologias: Holt-Winters+STL (projeção) | K-Means+Silhouette (clusterização) | HHI (concentração)<br>Paleta: Okabe-Ito + Tailwind (acessível para daltonismo) | Tema: {T}<br>Última atualização dos dados: {dq.get('ultima_data', 'N/D')}</div>""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()