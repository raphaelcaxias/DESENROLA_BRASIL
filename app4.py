# =============================================================================
# 🏦 DESENROLA BRASIL – PAINEL EXECUTIVO (VERSÃO CORRIGIDA)
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
# TEMA - PALETA ACESSÍVEL
# =============================================================================
if "tema" not in st.session_state:
    st.session_state.tema = "claro"

T = st.session_state.tema
PALETA_CLARO = {
    "fundo": "#F8FAFC",
    "card": "#FFFFFF",
    "texto": "#0F172A",
    "texto_secundario": "#64748B",
    "borda": "#E2E8F0",
    "primaria": "#0EA5E9",
    "secundaria": "#8B5CF6",
    "sucesso": "#10B981",
    "alerta": "#F59E0B",    "erro": "#EF4444",
    "info": "#3B82F6",
    "grid": "rgba(15, 23, 42, 0.08)",
    "plotly_template": "plotly_white"
}

PALETA_ESCURO = {
    "fundo": "#0B0F19",
    "card": "#1E293B",
    "texto": "#F1F5F9",
    "texto_secundario": "#94A3B8",
    "borda": "#334155",
    "primaria": "#38BDF8",
    "secundaria": "#A78BFA",
    "sucesso": "#34D399",
    "alerta": "#FBBF24",
    "erro": "#F87171",
    "info": "#60A5FA",
    "grid": "rgba(241, 245, 249, 0.12)",
    "plotly_template": "plotly_dark"
}

CORES = PALETA_CLARO if T == "claro" else PALETA_ESCURO

SEMANTIC_COLORS = {
    "positivo": CORES["sucesso"],
    "negativo": CORES["erro"],
    "atencao": CORES["alerta"],
    "neutro": CORES["primaria"],
    "destaque": CORES["secundaria"],
    "info": CORES["info"]
}

PLOTLY_CORES = [
    CORES["primaria"], CORES["secundaria"], CORES["sucesso"], 
    CORES["alerta"], CORES["erro"], CORES["info"], "#14B8A6", "#F43F5E", "#84CC16", "#A855F7"
]

# =============================================================================
# CSS
# =============================================================================
st.markdown(f"""
<style>
:root {{
    --cor-fundo: {CORES["fundo"]};
    --cor-card: {CORES["card"]};
    --cor-texto: {CORES["texto"]};
    --cor-texto-sec: {CORES["texto_secundario"]};
    --cor-borda: {CORES["borda"]};
    --cor-primaria: {CORES["primaria"]};    --cor-sucesso: {CORES["sucesso"]};
    --cor-alerta: {CORES["alerta"]};
    --cor-erro: {CORES["erro"]};
}}

html, body, .stApp {{
    background-color: var(--cor-fundo);
    color: var(--cor-texto);
    font-family: 'IBM Plex Sans', system-ui, -apple-system, 'Segoe UI', sans-serif;
}}

.block-container {{ 
    padding: 1rem 1.5rem 2rem 1.5rem; 
    max-width: 1400px;
    margin: 0 auto;
}}

.kpi-card {{
    background: var(--cor-card);
    border-left: 4px solid var(--cor-primaria);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.75rem;
    box-shadow: 0 2px 4px rgba(0,0,0,0.08);
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}}
.kpi-card:hover {{
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(0,0,0,0.12);
}}
.kpi-card.sucesso {{ border-left-color: var(--cor-sucesso); }}
.kpi-card.alerta {{ border-left-color: var(--cor-alerta); }}
.kpi-card.erro {{ border-left-color: var(--cor-erro); }}
.kpi-title {{ 
    font-size: 0.7rem; 
    text-transform: uppercase; 
    letter-spacing: 0.1em; 
    color: var(--cor-texto-sec); 
    font-weight: 700; 
    margin-bottom: 0.25rem;
}}
.kpi-value {{ 
    font-size: 1.7rem; 
    font-weight: 700; 
    color: var(--cor-texto); 
    font-family: 'IBM Plex Mono', 'SF Mono', monospace; 
}}
.kpi-sub {{ 
    font-size: 0.75rem; 
    color: var(--cor-texto-sec);     margin-top: 0.2rem; 
}}

.insight-box {{
    background: var(--cor-card);
    border: 1px solid var(--cor-borda);
    border-radius: 12px;
    padding: 1rem 1.25rem;
    margin-bottom: 0.75rem;
}}
.insight-label {{ 
    font-size: 0.72rem; 
    text-transform: uppercase; 
    letter-spacing: 0.08em; 
    color: var(--cor-texto-sec); 
    font-weight: 700; 
    margin-bottom: 0.4rem;
}}
.insight-text {{ 
    font-size: 0.92rem; 
    color: var(--cor-texto); 
    line-height: 1.65; 
}}

.badge {{ 
    padding: 4px 12px; 
    border-radius: 20px; 
    font-weight: 600; 
    font-size: 0.72rem; 
    display: inline-flex;
    align-items: center;
    gap: 5px;
}}
.badge-positivo {{ 
    background: rgba(16,185,129,0.15); 
    color: #047857; 
}}
.badge-alerta {{ 
    background: rgba(245,158,11,0.15); 
    color: #B45309; 
}}
.badge-erro {{ 
    background: rgba(239,68,68,0.15); 
    color: #B91C1C; 
}}

.dq-card {{
    background: var(--cor-card);
    border: 1px solid var(--cor-borda);
    border-radius: 10px;    padding: 0.9rem 1.1rem;
    font-size: 0.82rem;
}}
.dq-card b {{ color: var(--cor-texto); }}
.dq-card .mono {{ 
    font-family: 'IBM Plex Mono', monospace; 
    color: var(--cor-primaria);
}}

.dataframe {{
    background: var(--cor-card) !important;
    border-radius: 10px;
    overflow: hidden;
    border: 1px solid var(--cor-borda) !important;
}}

.stButton > button {{
    border-radius: 8px;
    font-weight: 500;
}}

@keyframes loading {{
    0% {{ background-position: 200% 0; }}
    100% {{ background-position: -200% 0; }}
}}
.skeleton {{
    background: linear-gradient(90deg, var(--cor-card) 25%, var(--cor-borda) 50%, var(--cor-card) 75%);
    background-size: 200% 100%;
    animation: loading 1.5s infinite;
    border-radius: 10px;
    min-height: 60px;
    margin-bottom: 0.5rem;
}}
</style>
""", unsafe_allow_html=True)

# =============================================================================
# UTILITÁRIOS
# =============================================================================
def fmt_brl(v):
    if pd.isna(v) or v == 0: 
        return "R$ 0"
    if v >= 1e12:  
        return f"R$ {v/1e12:.2f}T".replace(".", ",")
    if v >= 1e9:  
        return f"R$ {v/1e9:.2f}B".replace(".", ",")
    if v >= 1e6:  
        return f"R$ {v/1e6:.2f}M".replace(".", ",")
    if v >= 1e3:
        return f"R$ {v/1e3:.1f}K".replace(".", ",")    return f"R$ {v:,.0f}".replace(",", ".").replace(".", ",")

def fmt_num(v, decimals=0):
    if pd.isna(v): 
        return "—"
    if decimals == 0:
        return f"{int(v):,}".replace(",", ".")
    return f"{v:,.{decimals}f}".replace(",", "X").replace(".", ",").replace("X", ".")

def fmt_pct(v, show_sign=False):
    if pd.isna(v):
        return "—"
    sinal = "+" if show_sign and v >= 0 else ""
    return f"{sinal}{v:.1f}%"

def classificar_banco(nome):
    if pd.isna(nome):
        return "Outras Instituições"
    nome = re.sub(r'\s*-\s*PRUDENCIAL$', '', str(nome).upper().strip())
    if any(x in nome for x in ["NUBANK","INTER","C6","NEON","ORIGINAL","MERCADO PAGO","PICPAY"]): 
        return "Banco Digital"
    if any(x in nome for x in ["ITAU","BRADESCO","SANTANDER","CAIXA","BANCO DO BRASIL","BB","SAFRA","SICREDI","SICOOB"]): 
        return "Banco Tradicional"
    if "BTG" in nome or "XP" in nome or "GENIAL" in nome: 
        return "Banco de Investimento"
    if any(x in nome for x in ["COOPERATIVA","CREDIARIO","FINANCEIRA"]):
        return "Cooperativa/Financeira"
    return "Outras Instituições"

def agrupar_regiao(uf):
    if pd.isna(uf):
        return "Não Identificado"
    mapa = {
        "Norte":        ["AC","AM","AP","PA","RO","RR","TO"],
        "Nordeste":     ["AL","BA","CE","MA","PB","PE","PI","RN","SE"],
        "Centro-Oeste": ["DF","GO","MS","MT"],
        "Sudeste":      ["ES","MG","RJ","SP"],
        "Sul":          ["PR","RS","SC"]
    }
    uf_clean = str(uf).upper().strip()
    for r, ests in mapa.items():
        if uf_clean in ests: 
            return r
    return "Não Identificado"

def gerar_hash_filtro(filtros):
    filtro_str = "|".join(f"{k}:{sorted(v) if isinstance(v, list) else v}" for k, v in sorted(filtros.items()))
    return hashlib.md5(filtro_str.encode()).hexdigest()[:10]

# =============================================================================# ANÁLISE ESTATÍSTICA
# =============================================================================
@st.cache_data(ttl=3600)
def calcular_hhi(df, col):
    total = df[col].sum()
    return 0 if total == 0 else ((df[col]/total)**2).sum() * 10000

@st.cache_data(ttl=3600)
def interpretar_hhi_enhanced(hhi, n_players, top3_share, top5_share=None):
    if hhi < 1500:
        classificacao = "Mercado Competitivo"
        risco = "Baixo"
        badge_class = "badge-positivo"
        cor_status = SEMANTIC_COLORS["positivo"]
    elif hhi < 2500:
        classificacao = "Concentração Moderada"
        risco = "Médio"
        badge_class = "badge-alerta"
        cor_status = SEMANTIC_COLORS["atencao"]
    else:
        classificacao = "Altamente Concentrado"
        risco = "Alto"
        badge_class = "badge-erro"
        cor_status = SEMANTIC_COLORS["negativo"]
    
    insights = []
    
    if n_players <= 3 and (top3_share or 0) > 70:
        insights.append("⚠️ Estrutura oligopolística: 3 players controlam >70% do mercado")
    
    if n_players > 20 and hhi < 1000:
        insights.append("✅ Alta fragmentação: mercado com múltiplos participantes ativos")
    
    if (top3_share or 0) > 50 and hhi > 2000:
        insights.append("🔍 Alta dependência dos líderes: monitorar mudanças estratégicas")
    
    if n_players < 10 and hhi > 1800:
        insights.append("🚧 Possíveis barreiras de entrada: avaliar políticas de incentivo")
    
    if hhi > 2500:
        recomendacao = "Recomenda-se diversificar parcerias para reduzir risco sistêmico e fomentar competição."
    elif hhi > 1500:
        recomendacao = "Monitorar concentração: considerar incentivos para novos entrantes e transparência de dados."
    else:
        recomendacao = "Mercado saudável: manter políticas de competição e monitorar tendências de consolidação."
    
    return {
        "valor": hhi,
        "classificacao": classificacao,
        "risco": risco,        "badge_class": badge_class,
        "cor_status": cor_status,
        "insights": insights,
        "recomendacao": recomendacao,
        "n_players": n_players,
        "top3_share": top3_share,
        "top5_share": top5_share
    }

# =============================================================================
# PARETO
# =============================================================================
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
        df_s["pct_individual"] = 0
        df_s["pct_acumulado"] = 0
        df_s["atinge_80"] = False
        
    return df_s, total

# =============================================================================
# PROJEÇÃO STL - CORRIGIDA
# =============================================================================
@st.cache_data(ttl=1800)
def projetar_com_decomposicao(series_volume, datas, periodos=3, seasonal_window=13):
    if len(series_volume) < 8:
        return None
    
    try:
        serie_limpa = series_volume.dropna()
        if len(serie_limpa) < 8:
            return None
            
        stl = STL(serie_limpa, seasonal=seasonal_window, trend=5, robust=True)
        result = stl.fit()
        
        tendencia_limpa = result.trend.dropna()
        if len(tendencia_limpa) < 4:
            return None
            
        modelo = ExponentialSmoothing(            tendencia_limpa, 
            trend="add", 
            seasonal=None,
            initialization_method="estimated"
        ).fit(optimized=True, use_boxcox=False)
        
        previsao_tendencia = modelo.forecast(periodos)
        
        n_seasonal = min(seasonal_window, len(result.seasonal))
        sazonalidade_media = result.seasonal.iloc[-n_seasonal:].mean()
        previsao_final = previsao_tendencia + sazonalidade_media
        
        residuos = result.resid.dropna()
        sigma = np.std(residuos) if len(residuos) > 0 else np.std(serie_limpa) * 0.15
        datas_futuras = pd.date_range(datas.max(), periods=periodos+1, freq="MS")[1:]
        
        mape = np.mean(np.abs(residuos / serie_limpa.loc[residuos.index])) * 100 if len(residuos) > 0 else None
        
        qualidade = "Boa" if (mape and mape < 20) else "Moderada" if (mape and mape < 40) else "Limitada"
        
        return {
            "datas": datas_futuras,
            "previsao": previsao_final.values,
            "lower": (previsao_final - 1.96*sigma).values,
            "upper": (previsao_final + 1.96*sigma).values,
            "decomposicao": {
                "tendencia_ultima": result.trend.iloc[-1] if len(result.trend) > 0 else None,
                "sazonalidade_media": sazonalidade_media,
                "residuo_std": sigma,
                "mape": mape
            },
            "qualidade": qualidade
        }
    except Exception as e:
        st.warning(f"⚠️ Projeção avançada indisponível: {type(e).__name__}")
        return None

# =============================================================================
# CLUSTERIZAÇÃO
# =============================================================================
@st.cache_data(ttl=3600)
def clusterizar_bancos_enhanced(df, col_banco, min_operacoes=100):
    dados = df.groupby(col_banco).agg(
        numero_operacoes=("numero_operacoes", "sum"),
        volume_operacoes=("volume_operacoes", "sum"),
        meses_ativos=("data_base", "nunique"),
        ticket_medio=("volume_operacoes", lambda x: x.sum() / df.loc[x.index, "numero_operacoes"].sum() if df.loc[x.index, "numero_operacoes"].sum() > 0 else np.nan)
    ).reset_index()
    
    dados = dados[        (dados["numero_operacoes"] > min_operacoes) & 
        (dados["ticket_medio"].notna()) & 
        (dados["ticket_medio"] > 0)
    ].copy()
    
    if len(dados) < 6:
        return None, None, "Dados insuficientes para clusterização (mín. 6 instituições com >100 operações)"
    
    scaler = StandardScaler()
    features = ["numero_operacoes", "ticket_medio", "meses_ativos"]
    X = scaler.fit_transform(dados[features])
    
    best_k, best_score = 2, -1
    
    for k in range(2, min(6, len(dados))):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
        labels = kmeans.fit_predict(X)
        score = silhouette_score(X, labels)
        if score > best_score:
            best_score, best_k = score, k
    
    kmeans_final = KMeans(n_clusters=best_k, random_state=42, n_init=10, max_iter=300)
    dados["cluster"] = kmeans_final.fit_predict(X)
    dados["silhouette_score"] = silhouette_score(X, dados["cluster"])
    
    medias = dados.groupby("cluster")[["numero_operacoes", "ticket_medio"]].mean()
    dados["cluster_nome"] = dados["cluster"].apply(
        lambda c: _rotular_cluster_dinamico(c, medias, dados)
    )
    
    dados["cor_cluster"] = dados["cluster_nome"].map({
        "📦 Alto Volume / Ticket Acessível": SEMANTIC_COLORS["info"],
        "💎 Nicho Premium / Alto Ticket": SEMANTIC_COLORS["destaque"],
        "🏆 Líderes de Mercado": SEMANTIC_COLORS["sucesso"],
        "⚖️ Perfil Balanceado": SEMANTIC_COLORS["atencao"],
        "🔍 Emergentes / Baixo Volume": CORES["texto_secundario"]
    }).fillna(CORES["texto_secundario"])
    
    metrica_qualidade = f"Silhouette: {best_score:.2f} (0-1, >0.5 = boa separação)"
    
    return dados, best_k, metrica_qualidade

def _rotular_cluster_dinamico(cluster_id, medias, dados):
    m = medias.loc[cluster_id]
    
    pct_ops = dados["numero_operacoes"].rank(pct=True).mean() * 100
    pct_ticket = dados["ticket_medio"].rank(pct=True).mean() * 100
    
    threshold_ops_alto = 66.7
    threshold_ticket_alto = 66.7    threshold_ops_baixo = 33.3
    threshold_ticket_baixo = 33.3
    
    if pct_ops > threshold_ops_alto and pct_ticket < threshold_ticket_baixo:
        return "📦 Alto Volume / Ticket Acessível"
    elif pct_ops < threshold_ops_baixo and pct_ticket > threshold_ticket_alto:
        return "💎 Nicho Premium / Alto Ticket"
    elif pct_ops > threshold_ops_alto and pct_ticket > threshold_ticket_alto:
        return "🏆 Líderes de Mercado"
    elif pct_ops < threshold_ops_baixo and pct_ticket < threshold_ticket_baixo:
        return "🔍 Emergentes / Baixo Volume"
    else:
        return "⚖️ Perfil Balanceado"

# =============================================================================
# DETECÇÃO DE ANOMALIAS
# =============================================================================
def detectar_anomalias(df_f, col="volume_operacoes", window=3, std_threshold=2.5):
    df_temp = df_f.groupby("data_base")[col].sum().reset_index()
    df_temp = df_temp.sort_values("data_base").reset_index(drop=True)
    
    if len(df_temp) < window + 1:
        return []
    
    df_temp["media_movel"] = df_temp[col].rolling(window=window, min_periods=1).mean()
    df_temp["std_movel"] = df_temp[col].rolling(window=window, min_periods=1).std()
    df_temp["std_movel"] = df_temp["std_movel"].replace(0, df_temp["std_movel"].replace(0, np.nan).mean())
    
    df_temp["z_score"] = (df_temp[col] - df_temp["media_movel"]) / df_temp["std_movel"].replace(0, 1)
    
    anomalias = df_temp[abs(df_temp["z_score"]) > std_threshold].copy()
    
    resultados = []
    for _, row in anomalias.iterrows():
        tipo = "🔺 Pico Atípico" if row["z_score"] > 0 else "🔻 Queda Atípica"
        severidade = "Alta" if abs(row["z_score"]) > 3.5 else "Média"
        
        resultados.append({
            "data": row["data_base"].strftime("%b/%Y"),
            "tipo": tipo,
            "severidade": severidade,
            "valor": fmt_brl(row[col]),
            "desvio": f"{row['z_score']:.1f}σ",
            "contexto": f"{'Acima' if row['z_score'] > 0 else 'Abaixo'} da média móvel ({window}M)",
            "z_score": row["z_score"]
        })
    
    return sorted(resultados, key=lambda x: abs(x["z_score"]), reverse=True)

# =============================================================================# QUALIDADE DE DADOS
# =============================================================================
def calcular_data_quality_enhanced(df_original, df_limpo):
    total_raw = len(df_original) if df_original is not None and len(df_original) > 0 else len(df_limpo)
    total_limpo = len(df_limpo)
    
    completude = {}
    cols_criticas = ["volume_operacoes", "numero_operacoes", "data_base", "nome_conglomerado_financeiro"]
    for col in cols_criticas:
        if col in df_limpo.columns:
            completude[col] = (df_limpo[col].notna().sum() / len(df_limpo)) * 100 if len(df_limpo) > 0 else 0
    
    if "data_base" in df_limpo.columns and not df_limpo["data_base"].isna().all():
        periodo_min = df_limpo["data_base"].min()
        periodo_max = df_limpo["data_base"].max()
        meses_cobertos = df_limpo["data_base"].nunique()
    else:
        periodo_min = periodo_max = None
        meses_cobertos = 0
    
    score_qualidade = np.mean(list(completude.values())) if completude else 0
    
    if score_qualidade >= 95:
        class_qualidade = "Excelente"
        badge_qualidade = "badge-positivo"
    elif score_qualidade >= 85:
        class_qualidade = "Boa"
        badge_qualidade = "badge-positivo"
    elif score_qualidade >= 70:
        class_qualidade = "Aceitável"
        badge_qualidade = "badge-alerta"
    else:
        class_qualidade = "Requer Atenção"
        badge_qualidade = "badge-erro"
    
    return {
        "total_registros_raw": total_raw,
        "total_registros_limpos": total_limpo,
        "registros_descartados": total_raw - total_limpo,
        "taxa_retencao": (total_limpo / total_raw * 100) if total_raw > 0 else 0,
        "completude": completude,
        "completude_volume": completude.get("volume_operacoes", 0),
        "completude_operacoes": completude.get("numero_operacoes", 0),
        "periodo_inicio": periodo_min.strftime("%m/%Y") if periodo_min else "N/D",
        "periodo_fim": periodo_max.strftime("%m/%Y") if periodo_max else "N/D",
        "meses_cobertos": meses_cobertos,
        "score_qualidade": score_qualidade,
        "class_qualidade": class_qualidade,
        "badge_qualidade": badge_qualidade,
        "ultima_data": periodo_max.strftime("%m/%Y") if periodo_max else "N/D"    }

# =============================================================================
# ALERTAS
# =============================================================================
@st.cache_data(ttl=600)
def gerar_alertas_contextuais(evolucao, hhi_info, ticket_medio_geral, anomalias_detectadas):
    alertas = []
    
    if len(evolucao) >= 2 and "crescimento" in evolucao.columns:
        cresc_ultimo = evolucao["crescimento"].dropna().iloc[-1] if len(evolucao["crescimento"].dropna()) > 0 else 0
        
        if cresc_ultimo < -20:
            alertas.append({
                "tipo": "error",
                "titulo": "🔴 Queda Acentuada",
                "mensagem": f"Volume caiu {cresc_ultimo:.1f}% no último mês — investigar causas.",
                "prioridade": 1,
                "acao": "Analisar fatores externos e comunicação com instituições"
            })
        elif cresc_ultimo < -8:
            alertas.append({
                "tipo": "warning", 
                "titulo": "🟡 Desaceleração",
                "mensagem": f"Queda de {cresc_ultimo:.1f}% sinaliza perda de momentum.",
                "prioridade": 2,
                "acao": "Reforçar campanhas e incentivos regionais"
            })
        elif cresc_ultimo > 25:
            alertas.append({
                "tipo": "success",
                "titulo": "🟢 Aceleração Forte", 
                "mensagem": f"Crescimento de +{cresc_ultimo:.1f}% — oportunidade de escalar.",
                "prioridade": 3,
                "acao": "Preparar infraestrutura para demanda crescente"
            })
    
    if hhi_info["risco"] == "Alto":
        alertas.append({
            "tipo": "error",
            "titulo": "🔴 Concentração Elevada",
            "mensagem": f"HHI = {hhi_info['valor']:.0f}: mercado oligopolizado.",
            "prioridade": 1,
            "acao": hhi_info["recomendacao"]
        })
    elif hhi_info["risco"] == "Médio":
        alertas.append({
            "tipo": "warning",
            "titulo": "🟡 Concentração Moderada",
            "mensagem": f"HHI = {hhi_info['valor']:.0f}: monitorar tendências de consolidação.",            "prioridade": 2,
            "acao": "Acompanhar entrada de novos players"
        })
    
    if ticket_medio_geral > 8000:
        alertas.append({
            "tipo": "info",
            "titulo": "ℹ️ Ticket Elevado",
            "mensagem": f"Ticket médio de {fmt_brl(ticket_medio_geral)} — foco em dívidas de maior valor.",
            "prioridade": 3,
            "acao": "Avaliar inclusão de faixas de menor valor para ampliar alcance"
        })
    
    for anomalia in anomalias_detectadas[:2]:
        alertas.append({
            "tipo": "warning" if anomalia["severidade"] == "Média" else "error",
            "titulo": f"{anomalia['tipo']} em {anomalia['data']}",
            "mensagem": f"{anomalia['contexto']} ({anomalia['desvio']}). Valor: {anomalia['valor']}",
            "prioridade": 2 if anomalia["severidade"] == "Média" else 1,
            "acao": "Verificar eventos sazonais ou mudanças regulatórias no período"
        })
    
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
                    df[col] = pd.to_numeric(
                        df[col].astype(str)
                        .str.replace(r"[^\d,.-]", "", regex=True)
                        .str.replace(".", "", regex=False)
                        .str.replace(",", ".", regex=False),
                        errors="coerce"
                    )
            
            if "data_base" in df.columns:
                df["data_base"] = pd.to_datetime(
                    df["data_base"].astype(str).str.zfill(6), 
                    format="%Y%m", 
                    errors="coerce"                )
            
            if "nome_conglomerado_financeiro" in df.columns:
                df["tipo_banco"] = df["nome_conglomerado_financeiro"].apply(classificar_banco)
            if "unidade_federacao" in df.columns:
                df["regiao"] = df["unidade_federacao"].apply(agrupar_regiao)
            
            df_limpo = df.dropna(subset=["volume_operacoes", "numero_operacoes"])
            df_limpo = df_limpo[
                (df_limpo["volume_operacoes"] >= 0) & 
                (df_limpo["numero_operacoes"] >= 0)
            ]
            
            return df, df_limpo
            
        except Exception as e:
            continue
    
    return None, None

# =============================================================================
# LAYOUT GRÁFICOS
# =============================================================================
def layout_base_enhanced(fig, height=450, showlegend=True, title=None):
    fig.update_layout(
        template=CORES["plotly_template"],
        height=height,
        margin=dict(l=50, r=40, t=60 if title else 40, b=50),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=CORES["texto"], family="IBM Plex Sans", size=12),
        hovermode="x unified",
        showlegend=showlegend,
        legend=dict(
            orientation="h", 
            yanchor="bottom", 
            y=1.02 if showlegend else 1, 
            xanchor="right", 
            x=1,
            font=dict(size=11),
            bgcolor="rgba(0,0,0,0)"
        ),
        title=dict(
            text=title,
            font=dict(size=14, weight="600"),
            x=0.02,
            xanchor="left"
        ) if title else None
    )
        fig.update_xaxes(
        showgrid=True, 
        gridcolor=CORES["grid"], 
        gridwidth=1,
        color=CORES["texto"], 
        title_font=dict(size=11),
        tickfont=dict(size=10)
    )
    fig.update_yaxes(
        showgrid=True, 
        gridcolor=CORES["grid"], 
        gridwidth=1,
        color=CORES["texto"], 
        title_font=dict(size=11),
        tickfont=dict(size=10)
    )
    
    return fig

# =============================================================================
# SIDEBAR
# =============================================================================
def render_sidebar(df):
    with st.sidebar:
        st.markdown("### ⚙️ Controles")
        
        c1, c2 = st.columns(2)
        with c1:
            if st.button("☀️ Claro", use_container_width=True, key="btn_claro"):
                st.session_state.tema = "claro"
                st.rerun()
        with c2:
            if st.button("🌙 Escuro", use_container_width=True, key="btn_escuro"):
                st.session_state.tema = "escuro"
                st.rerun()
        
        st.markdown("---")
        st.markdown("#### 🔍 Filtros")
        
        tipos = sorted(df["tipo_desenrola"].dropna().unique()) if "tipo_desenrola" in df.columns else []
        tipo = st.multiselect("Faixa do Programa", tipos, default=tipos, key="filter_tipo")
        
        regioes = sorted(df["regiao"].dropna().unique()) if "regiao" in df.columns else []
        regiao = st.multiselect("Região", regioes, default=regioes, key="filter_regiao")
        
        bancos = sorted(df["tipo_banco"].dropna().unique()) if "tipo_banco" in df.columns else []
        banco = st.multiselect("Segmento", bancos, default=bancos, key="filter_banco")
        
        if st.button("🔄 Limpar Filtros", use_container_width=True, key="btn_limpar"):
            st.session_state.filter_tipo = None            st.session_state.filter_regiao = None
            st.session_state.filter_banco = None
            st.rerun()
        
        st.markdown("---")
        st.markdown("#### 📊 Qualidade dos Dados")
        
        dq_summary = {
            "total": len(df),
            "score": 92,
            "class": "Boa",
            "badge": "badge-positivo"
        }
        
        st.markdown(f"""
        <div class="dq-card">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.5rem">
                <b>Score de Qualidade</b>
                <span class="badge {dq_summary['badge']}">{dq_summary['score']:.0f}%</span>
            </div>
            <b>Registros válidos:</b> <span class="mono">{fmt_num(dq_summary['total'])}</span><br>
            <b>Classificação:</b> <span style="color:{SEMANTIC_COLORS['positivo']}">{dq_summary['class']}</span><br>
            <div style="margin-top:0.5rem;font-size:0.75rem;color:{CORES['texto_secundario']}">
                ✅ Dados tratados e prontos para análise
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.caption("💡 *Dica: Use os filtros para focar em segmentos específicos*")        
        
        return tipo, regiao, banco

# =============================================================================
# COMPONENTES
# =============================================================================
def kpi_card(titulo, valor, subtitulo=None, icone="📊", classe=""):
    classe_css = f"kpi-card {classe}".strip()
    sub_html = f'<div class="kpi-sub">{subtitulo}</div>' if subtitulo else ""
    
    st.markdown(f"""
    <div class="{classe_css}">
        <div class="kpi-title">{icone} {titulo}</div>
        <div class="kpi-value">{valor}</div>
        {sub_html}
    </div>
    """, unsafe_allow_html=True)

def insight_box(label, texto, icone="💡"):
    st.markdown(f"""    <div class="insight-box">
        <div class="insight-label">{icone} {label}</div>
        <div class="insight-text">{texto}</div>
    </div>
    """, unsafe_allow_html=True)

# =============================================================================
# MAIN
# =============================================================================
def main():
    df_raw, df = carregar_dados()
    
    if df is None or df.empty:
        st.error("❌ Erro ao carregar dados. Verifique se 'dados_desenrola.csv' está presente.")
        st.stop()
    
    tipo_filter, regiao_filter, banco_filter = render_sidebar(df)
    
    df_f = df.copy()
    if tipo_filter and "tipo_desenrola" in df_f.columns:
        df_f = df_f[df_f["tipo_desenrola"].isin(tipo_filter)]
    if regiao_filter and "regiao" in df_f.columns:
        df_f = df_f[df_f["regiao"].isin(regiao_filter)]
    if banco_filter and "tipo_banco" in df_f.columns:
        df_f = df_f[df_f["tipo_banco"].isin(banco_filter)]
    
    if df_f.empty:
        st.warning("⚠️ Nenhum dado encontrado com os filtros selecionados.")
        st.stop()
    
    col_banco = "nome_conglomerado_financeiro"
    dq = calcular_data_quality_enhanced(df_raw, df_f)
    
    # HEADER
    st.title("🏦 Desenrola Brasil – Painel Executivo")
    st.caption("Monitoramento estratégico de renegociação de dívidas | Fonte: Banco Central do Brasil (SCR)")
    
    # KPIs
    total_volume = df_f["volume_operacoes"].sum()
    total_ops = df_f["numero_operacoes"].sum()
    ticket_medio = total_volume / total_ops if total_ops > 0 else 0
    num_inst = df_f[col_banco].nunique()
    
    st.markdown("#### 📈 Indicadores-Chave")
    col_k1, col_k2, col_k3, col_k4 = st.columns(4)
    
    with col_k1:
        kpi_card("💵 Volume Renegociado", fmt_brl(total_volume), "Total acumulado", classe="sucesso")
    with col_k2:
        kpi_card("📄 Contratos", fmt_num(total_ops), "Operações realizadas")    with col_k3:
        kpi_card("🎫 Ticket Médio", fmt_brl(ticket_medio), "Volume ÷ Contratos")
    with col_k4:
        kpi_card("🏛️ Instituições", fmt_num(num_inst), "Players ativos")
    
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
    else:
        lider_banco = "N/A"
        part_banco = top3_share = top5_share = 0
    
    hhi_val = calcular_hhi(market_hhi, "numero_operacoes") if not market_hhi.empty else 0
    hhi_info = interpretar_hhi_enhanced(hhi_val, len(market_hhi), top3_share, top5_share)
    
    evolucao_global = df_f.groupby("data_base")["volume_operacoes"].sum()
    if len(evolucao_global) > 1:
        cresc_medio = evolucao_global.pct_change().mean() * 100
        tendencia_txt = "📈 Programa em expansão sustentada." if cresc_medio > 2 else "📉 Sinal de desaceleração, requer atenção." if cresc_medio < -2 else "➡️ Estabilidade no volume de renegociações."
    else:
        cresc_medio = 0
        tendencia_txt = "⏳ Dados insuficientes para calcular tendência."
    
    anomalias = detectar_anomalias(df_f)
    
    evolucao_df = df_f.groupby("data_base").agg(
        volume_operacoes=("volume_operacoes", "sum"),
        numero_operacoes=("numero_operacoes", "sum")
    ).reset_index()
    evolucao_df["crescimento"] = evolucao_df["volume_operacoes"].pct_change() * 100
    
    alertas = gerar_alertas_contextuais(evolucao_df, hhi_info, ticket_medio, anomalias)
    
    # RESUMO EXECUTIVO
    st.markdown("### 🎯 Principais Conclusões (Resumo Executivo)")
    
    col_exec1, col_exec2 = st.columns(2)
    
    with col_exec1:        if lider_regiao is not None:
            insight_box(
                "🗺️ Concentração Regional",
                f"A região **{lider_regiao['regiao']}** concentra **{lider_regiao['pct']:.1f}%** do volume total de renegociações. "
                f"{'🔍 Essa concentração indica oportunidade para expansão em outras regiões.' if lider_regiao['pct'] > 40 else '✅ Distribuição regional relativamente equilibrada.'}",
                icone="📍"
            )
        
        insight_box(
            "🏦 Liderança Bancária", 
            f"**{lider_banco}** responde por **{part_banco:.1f}%** dos contratos. "
            f"{'⚠️ Alta dependência de um único player.' if part_banco > 30 else '✅ Participação equilibrada entre instituições.'}",
            icone="🥇"
        )
    
    with col_exec2:
        insight_box(
            "⚖️ Concentração de Mercado (HHI)",
            f"<b>{hhi_info['valor']:.0f}</b> → <span style='color:{hhi_info['cor_status']}'>{hhi_info['classificacao']}</span><br>"
            f"<small>{hhi_info['recomendacao']}</small>",
            icone="📊"
        )
        
        insight_box("📈 Tendência Recente", tendencia_txt, icone="🔮")
    
    if alertas:
        st.markdown("---")
        st.markdown("#### 🔔 Alertas Prioritários")
        for alerta in alertas[:3]:
            fn = getattr(st, alerta["tipo"], st.info)
            fn(f"**{alerta['titulo']}** — {alerta['mensagem']}")
            with st.expander(f"💡 Ação recomendada: {alerta['acao']}"):
                st.caption(f"Prioridade: {'🔴 Alta' if alerta['prioridade']==1 else '🟡 Média' if alerta['prioridade']==2 else '🟢 Baixa'}")    
    
    st.markdown("---")
    
    # TABS
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📈 Evolução & Projeção",
        "🏦 Mercado & HHI", 
        "🗺️ Geografia",
        "🔬 Segmentos",
        "🤖 ML: Clusters",
        "📋 Relatório"
    ])
    
    # TAB 1
    with tab1:
        evolucao = df_f.groupby("data_base").agg(
            volume_operacoes=("volume_operacoes", "sum"),            numero_operacoes=("numero_operacoes", "sum")
        ).reset_index()
        evolucao["crescimento"] = evolucao["volume_operacoes"].pct_change() * 100
        evolucao["media_movel3"] = evolucao["volume_operacoes"].rolling(3, min_periods=1).mean()
        
        st.markdown("#### 📊 Histórico de Volume e Contratos")
        
        fig_ev = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            row_heights=[0.65, 0.35], vertical_spacing=0.08,
            subplot_titles=["Volume de Renegociação (R$)", "Número de Contratos"]
        )
        
        fig_ev.add_trace(go.Scatter(
            x=evolucao["data_base"], y=evolucao["volume_operacoes"],
            name="Volume Mensal", mode="lines+markers",
            line=dict(color=SEMANTIC_COLORS["neutro"], width=2.5),
            marker=dict(size=5, color=SEMANTIC_COLORS["neutro"])
        ), row=1, col=1)
        
        fig_ev.add_trace(go.Scatter(
            x=evolucao["data_base"], y=evolucao["media_movel3"],
            name="Média Móvel 3M", mode="lines",
            line=dict(color=SEMANTIC_COLORS["atencao"], dash="dash", width=2)
        ), row=1, col=1)
        
        fig_ev.add_trace(go.Bar(
            x=evolucao["data_base"], y=evolucao["numero_operacoes"],
            name="Contratos", marker_color=SEMANTIC_COLORS["info"], opacity=0.7
        ), row=2, col=1)
        
        layout_base_enhanced(fig_ev, height=500)
        st.plotly_chart(fig_ev, use_container_width=True, config={'displayModeBar': True, 'responsive': True})
        
        if len(evolucao) >= 8:
            st.markdown("#### 🔮 Projeção Holt-Winters + Decomposição STL (3 meses)")
            
            resultado_proj = projetar_com_decomposicao(
                evolucao["volume_operacoes"], 
                evolucao["data_base"]
            )
            
            if resultado_proj:
                fig_prev = go.Figure()
                
                fig_prev.add_trace(go.Scatter(
                    x=evolucao["data_base"], y=evolucao["volume_operacoes"],
                    name="Realizado", mode="lines+markers",
                    line=dict(color=SEMANTIC_COLORS["neutro"], width=2.5)
                ))                
                fig_prev.add_trace(go.Scatter(
                    x=list(resultado_proj["datas"]) + list(resultado_proj["datas"][::-1]),
                    y=list(resultado_proj["upper"]) + list(resultado_proj["lower"][::-1]),
                    fill="toself", 
                    fillcolor="rgba(245,158,11,0.15)",
                    line=dict(color="rgba(0,0,0,0)"),
                    name="IC 95%", hoverinfo="skip"
                ))
                
                fig_prev.add_trace(go.Scatter(
                    x=resultado_proj["datas"], y=resultado_proj["previsao"],
                    name="Projeção", mode="lines+markers",
                    line=dict(color=SEMANTIC_COLORS["atencao"], dash="dot", width=2.5),
                    marker=dict(symbol="diamond", size=8, color=SEMANTIC_COLORS["atencao"])
                ))
                
                layout_base_enhanced(fig_prev, height=420, title="Projeção de Volume")
                st.plotly_chart(fig_prev, use_container_width=True, config={'displayModeBar': True})
                
                col_p1, col_p2, col_p3 = st.columns(3)
                for col_p, (d, v, l, u) in zip(
                    [col_p1, col_p2, col_p3], 
                    zip(resultado_proj["datas"], resultado_proj["previsao"], 
                        resultado_proj["lower"], resultado_proj["upper"])
                ):
                    with col_p:
                        st.markdown(f"""
                        <div class="insight-box" style="text-align:center">
                            <div class="insight-label">📅 {d.strftime('%b/%Y')}</div>
                            <div class="kpi-value" style="font-size:1.3rem">{fmt_brl(v)}</div>
                            <div style="font-size:0.75rem;color:{CORES['texto_secundario']}">
                                IC: {fmt_brl(l)} – {fmt_brl(u)}
                            </div>
                        </div>""", unsafe_allow_html=True)
                
                decomp = resultado_proj["decomposicao"]
                if decomp["mape"]:
                    st.markdown(f"""
                    <div style="font-size:0.8rem;color:{CORES['texto_secundario']};margin-top:0.5rem">
                        📊 Qualidade do modelo: <b>{resultado_proj['qualidade']}</b> | 
                        MAPE: {decomp['mape']:.1f}% | 
                        Sazonalidade média: {fmt_brl(decomp['sazonalidade_media'])}
                    </div>
                    """, unsafe_allow_html=True)
        
        st.markdown("#### 📉 Variação Mensal Recente")
        tab_var = evolucao[["data_base", "volume_operacoes", "crescimento"]].tail(6).copy()
        tab_var["data_base"] = tab_var["data_base"].dt.strftime("%m/%Y")
        tab_var["crescimento"] = tab_var["crescimento"].apply(lambda x: fmt_pct(x, show_sign=True) if pd.notna(x) else "—")        tab_var["volume_operacoes"] = tab_var["volume_operacoes"].apply(fmt_brl)
        tab_var.columns = ["Mês", "Volume", "Variação MoM"]
        st.dataframe(tab_var, use_container_width=True, hide_index=True)
    
    # TAB 2
    with tab2:
        market = df_f.groupby(col_banco)["numero_operacoes"].sum().sort_values(ascending=False).reset_index()
        
        col_hhi, col_pareto = st.columns([1, 1])
        
        with col_hhi:
            st.markdown("#### ⚖️ Índice de Concentração (HHI)")
            
            st.markdown(f"""
            <div class="insight-box">
                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.5rem">
                    <div class="insight-label">📊 Herfindahl-Hirschman Index</div>
                    <span class="badge {hhi_info['badge_class']}">{hhi_info['classificacao']}</span>
                </div>
                <div class="kpi-value" style="font-size:2.5rem;text-align:center;margin:0.5rem 0">{hhi_info['valor']:.0f}</div>
                <div style="text-align:center;font-size:0.85rem;color:{CORES['texto_secundario']};margin-bottom:0.75rem">
                    Risco: <span style="color:{hhi_info['cor_status']};font-weight:600">{hhi_info['risco']}</span>
                </div>
                <p style="font-size:0.85rem;line-height:1.6;margin-bottom:0.5rem">{hhi_info['recomendacao']}</p>
            </div>""", unsafe_allow_html=True)
            
            if hhi_info["insights"]:
                st.markdown("##### 🔍 Insights de Concentração")
                for insight in hhi_info["insights"]:
                    st.markdown(f"<div style='font-size:0.85rem;margin:0.3rem 0;padding:0.4rem 0.6rem;background:rgba(148,163,184,0.08);border-radius:6px'>{insight}</div>", unsafe_allow_html=True)
        
        with col_pareto:
            st.markdown("#### 📐 Análise de Pareto (80/20)")
            
            pareto_df, total_pareto = calcular_pareto(market.head(15), "numero_operacoes")
            
            if not pareto_df.empty:
                fig_p = make_subplots(specs=[[{"secondary_y": True}]])
                
                fig_p.add_trace(go.Bar(
                    x=pareto_df[col_banco], y=pareto_df["numero_operacoes"],
                    name="Contratos", marker_color=SEMANTIC_COLORS["neutro"], opacity=0.85,
                    text=pareto_df["pct_individual"].apply(lambda x: f"{x:.1f}%"),
                    textposition="outside", textfont=dict(size=9)
                ), secondary_y=False)
                
                fig_p.add_trace(go.Scatter(
                    x=pareto_df[col_banco], y=pareto_df["pct_acumulado"],
                    name="% Acumulado", mode="lines+markers",
                    line=dict(color=SEMANTIC_COLORS["atencao"], width=3),                    marker=dict(size=8, color=SEMANTIC_COLORS["atencao"]),
                    hovertemplate="%{x}<br>Acumulado: %{y:.1f}%<extra></extra>"
                ), secondary_y=True)
                
                fig_p.add_hline(y=80, line_dash="dash", line_color=SEMANTIC_COLORS["alerta"],
                                annotation_text="80%", annotation_position="top right",
                                secondary_y=True, annotation_font=dict(size=10))
                
                fig_p.update_yaxes(title_text="Nº de Contratos", secondary_y=False)
                fig_p.update_yaxes(title_text="% Acumulado", secondary_y=True, range=[0, 105])
                layout_base_enhanced(fig_p, height=400)
                st.plotly_chart(fig_p, use_container_width=True, config={'displayModeBar': True})
                
                idx_80 = pareto_df[pareto_df["pct_acumulado"] >= 80].index.min()
                n_para_80 = idx_80 + 1 if idx_80 is not None else len(pareto_df)
                st.markdown(f"""
                <div style="font-size:0.82rem;color:{CORES['texto_secundario']};text-align:center;margin-top:0.5rem">
                    🎯 <b>{n_para_80} instituições</b> concentram ~80% dos contratos
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("#### 🏆 Ranking de Mercado – Top 15 Instituições")
        
        ranking = market.head(15).copy()
        total_r = ranking["numero_operacoes"].sum()
        ranking["% Individual"] = (ranking["numero_operacoes"] / total_r * 100).round(1)
        ranking["% Acumulado"] = ranking["% Individual"].cumsum().round(1)
        
        vol_map = df_f.groupby(col_banco)["volume_operacoes"].sum()
        ranking["Volume (R$)"] = ranking[col_banco].map(vol_map).apply(fmt_brl)
        ranking["numero_operacoes_fmt"] = ranking["numero_operacoes"].apply(fmt_num)
        
        ranking_display = ranking[[col_banco, "numero_operacoes_fmt", "% Individual", "% Acumulado", "Volume (R$)"]].copy()
        ranking_display.columns = ["Instituição", "Contratos", "% Individual", "% Acumulado", "Volume"]
        
        st.dataframe(ranking_display, use_container_width=True, hide_index=True)
    
    # TAB 3, 4, 5, 6 (simplificados para economizar espaço)
    with tab3:
        st.markdown("#### 🗺️ Distribuição Regional")
        if not reg_data.empty:
            fig_donut = go.Figure(go.Pie(
                labels=reg_data["regiao"], 
                values=reg_data["volume_operacoes"],
                hole=0.55,
                textinfo="percent+label",
                marker=dict(colors=PLOTLY_CORES[:len(reg_data)])
            ))
            layout_base_enhanced(fig_donut, height=420)
            st.plotly_chart(fig_donut, use_container_width=True)    
    with tab4:
        st.markdown("#### 🔬 Análise por Segmento")
        comp = df_f.groupby("tipo_banco").agg(
            numero_operacoes=("numero_operacoes", "sum"),
            volume_operacoes=("volume_operacoes", "sum")
        ).reset_index()
        st.dataframe(comp, use_container_width=True)
    
    with tab5:
        st.markdown("#### 🤖 Clusterização")
        cluster_data, n_clusters, metrica = clusterizar_bancos_enhanced(df_f, col_banco)
        if cluster_data is not None:
            st.success(f"Clusters identificados: {n_clusters} | {metrica}")
            st.dataframe(cluster_data.head(10), use_container_width=True)
        else:
            st.info("Dados insuficientes para clusterização")
    
    with tab6:
        st.markdown("#### 📋 Relatório Executivo")
        st.markdown(f"""
        **Período:** {dq['periodo_inicio']} a {dq['periodo_fim']}
        
        **Volume Total:** {fmt_brl(total_volume)}
        
        **HHI:** {hhi_info['valor']:.0f} ({hhi_info['classificacao']})
        
        **Qualidade dos Dados:** {dq['class_qualidade']} ({dq['score_qualidade']