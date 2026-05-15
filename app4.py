import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from datetime import datetime
import re

# ============================================================
# CONFIGURAÇÃO DA PÁGINA (Identidade Corporativa)
# ============================================================
st.set_page_config(
    page_title="Corporate Analytics | Desenrola Brasil",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# PALETA DE CORES INSTITUCIONAL (Executive Dark/Light)
# ============================================================
if "tema" not in st.session_state:
    st.session_state.tema = "claro"

if st.session_state.tema == "claro":
    COR_FUNDO = "#F4F6F9"
    COR_CARD = "#FFFFFF"
    COR_TEXTO = "#1E293B"
    COR_BORDA = "#E2E8F0"
    COR_PRIMARIA = "#0F172A"    # Azul Noturno Real
    COR_SECUNDARIA = "#2563EB"  # Azul Bancário
    COR_SUCESSO = "#16A34A"     # Verde Mercado
    COR_ALERTA = "#DC2626"      # Vermelho Risco
else:
    COR_FUNDO = "#0B0F19"
    COR_CARD = "#111827"
    COR_TEXTO = "#F3F4F6"
    COR_BORDA = "#1F2937"
    COR_PRIMARIA = "#38BDF8"
    COR_SECUNDARIA = "#60A5FA"
    COR_SUCESSO = "#34D399"
    COR_ALERTA = "#F87171"

# ============================================================
# CSS CUSTOMIZADO (Design de Sistema Financeiro)
# ============================================================
st.markdown(f"""
<style>
    html, body, .stApp {{
        background-color: {COR_FUNDO};
        color: {COR_TEXTO};
        font-family: 'Inter', -apple-system, sans-serif;
    }}
    .block-container {{
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }}
    /* KPI Cards Estilo Terminal Bloomberg/Reuters */
    .kpi-card {{
        background: {COR_CARD};
        border-left: 4px solid {COR_SECUNDARIA};
        border-top: 1px solid {COR_BORDA};
        border-right: 1px solid {COR_BORDA};
        border-bottom: 1px solid {COR_BORDA};
        border-radius: 6px;
        padding: 1.2rem;
        margin-bottom: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }}
    .kpi-title {{
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748B;
        font-weight: 600;
    }}
    .kpi-value {{
        font-size: 1.6rem;
        font-weight: 700;
        color: {COR_TEXTO};
        margin-top: 0.3rem;
    }}
    /* Badges de Risco/Concentração */
    .badge {{
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.8rem;
    }}
    .badge-low {{ background-color: rgba(22, 163, 74, 0.1); color: #16A34A; }}
    .badge-mid {{ background-color: rgba(217, 119, 6, 0.1); color: #D97706; }}
    .badge-high {{ background-color: rgba(220, 38, 38, 0.1); color: #DC2626; }}
</style>
""", unsafe_allow_html=True)

# ============================================================
# FUNÇÕES DE FORMATAÇÃO E TRATAMENTO
# ============================================================
def fmt_brl(valor):
    if pd.isna(valor) or valor == 0:
        return "R$ 0,00"
    if valor >= 1_000_000_000:
        return f"R$ {valor/1_000_000_000:.2f} Bi".replace(".", ",")
    if valor >= 1_000_000:
        return f"R$ {valor/1_000_000:.2f} Mi".replace(".", ",")
    return f"R$ {valor:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

def fmt_num(valor):
    if pd.isna(valor):
        return "0"
    return f"{int(valor):,}".replace(",", ".")

def classificar_banco(nome):
    nome = str(nome).upper().strip()
    nome = re.sub(r'\s*-\s*PRUDENCIAL$', '', nome)
    if any(x in nome for x in ["NUBANK", "INTER", "C6", "NEON", "ORIGINAL"]):
        return "Banco Digital"
    if any(x in nome for x in ["ITAU", "BRADESCO", "SANTANDER", "CAIXA", "BANCO DO BRASIL", "BB"]):
        return "Banco Tradicional"
    return "Demais Categorias"

def agrupar_regiao(uf):
    mapa = {
        "Norte": ["AC", "AM", "AP", "PA", "RO", "RR", "TO"],
        "Nordeste": ["AL", "BA", "CE", "MA", "PB", "PE", "PI", "RN", "SE"],
        "Centro-Oeste": ["DF", "GO", "MS", "MT"],
        "Sudeste": ["ES", "MG", "RJ", "SP"],
        "Sul": ["PR", "RS", "SC"]
    }
    for regiao, estados in mapa.items():
        if uf in estados:
            return regiao
    return "Não Identificado"

def calcular_hhi(df, coluna):
    total = df[coluna].sum()
    if total == 0: return 0
    return ((df[coluna] / total) ** 2).sum() * 10000

def interpretar_hhi(hhi):
    if hhi < 1500: return "Baixa Concentração (Mercado Competitivo)", "badge-low"
    elif hhi < 2500: return "Concentração Moderada", "badge-mid"
    return "Altamente Concentrado (Risco Sistêmico)", "badge-high"

def calcular_pareto(df, coluna):
    df_sorted = df.sort_values(coluna, ascending=False).reset_index(drop=True)
    df_sorted['percentual_acumulado'] = (df_sorted[coluna].cumsum() / df_sorted[coluna].sum()) * 100
    return df_sorted

# TEMPLATE DE LAYOUT DOS GRÁFICOS (Padronização Interna)
def aplicar_layout_grafico(fig):
    fig.update_layout(
        margin=dict(l=40, r=40, t=40, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=COR_TEXTO, family="Inter"),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.update_xaxes(showgrid=False, color=COR_TEXTO)
    fig.update_yaxes(showgrid=True, gridcolor=COR_BORDA, color=COR_TEXTO)
    return fig

# ============================================================
# CARREGAMENTO DOS DADOS
# ============================================================
@st.cache_data(ttl=3600)
def carregar_dados():
    encodings = ["utf-8", "latin1", "cp1252"]
    for enc in encodings:
        try:
            df = pd.read_csv("dados_desenrola.csv", sep=";", encoding=enc, low_memory=False)
            df.columns = df.columns.str.lower().str.strip()
            
            for col in ["numero_operacoes", "volume_operacoes"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(
                        df[col].astype(str).str.replace(".", "", regex=False).str.replace(",", ".", regex=False), 
                        errors="coerce"
                    )
            
            df["data_base"] = pd.to_datetime(df["data_base"].astype(str), format="%Y%m", errors="coerce")
            df["tipo_banco"] = df["nome_conglomerated_financeiro" if "nome_conglomerated_financeiro" in df.columns else "nome_conglomerado_financeiro"].apply(classificar_banco)
            df["regiao"] = df["unidade_federacao"].apply(agrupar_regiao)
            return df.dropna(subset=["volume_operacoes", "numero_operacoes"])
        except:
            continue
    return None

df = carregar_dados()
if df is None:
    st.error("Erro crítico: Base de dados 'dados_desenrola.csv' não localizada ou corrompida.")
    st.stop()

# ============================================================
# SIDEBAR CONTROL PANEL
# ============================================================
with st.sidebar:
    st.markdown("### Painel de Controle")
    
    # Seletor de Tema discreto
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        if st.button("Modo Claro", use_container_width=True):
            st.session_state.tema = "claro"
            st.rerun()
    with col_t2:
        if st.button("Modo Escuro", use_container_width=True):
            st.session_state.tema = "escuro"
            st.rerun()
            
    st.markdown("---")
    
    tipos = sorted(df["tipo_desenrola"].unique())
    tipo = st.multiselect("Faixa do Programa", tipos, default=tipos)
    
    regioes = sorted(df["regiao"].unique())
    regiao = st.multiselect("Região Geográfica", regioes, default=regioes)
    
    bancos = sorted(df["tipo_banco"].unique())
    banco = st.multiselect("Segmento Institucional", bancos, default=bancos)

# Filtragem Eficiente
df_filtrado = df[
    (df["tipo_desenrola"].isin(tipo)) &
    (df["regiao"].isin(regiao)) &
    (df["tipo_banco"].isin(banco))
]

if df_filtrado.empty:
    st.warning("Nenhum registro encontrado para o cruzamento de filtros selecionado.")
    st.stop()

# ============================================================
# EXECUTIVE HEADER
# ============================================================
col_header, col_date = st.columns([2, 1])
with col_header:
    st.title("Desenrola Brasil Analytics")
    st.caption("Relatório Gerencial de Monitoramento de Crédito e Renegociações — Fonte: Banco Central do Brasil")
with col_date:
    st.markdown(f"<div style='text-align: right; color: gray; font-size: 0.85rem; padding-top: 25px;'>Atualizado em: {datetime.now().strftime('%d/%m/%Y')}</div>", unsafe_allow_html=True)

st.markdown("---")

# ============================================================
# METRICAS PRINCIPAIS (KPI DASHBOARD)
# ============================================================
total_volume = df_filtrado["volume_operacoes"].sum()
total_operacoes = df_filtrado["numero_operacoes"].sum()
ticket_medio = total_volume / total_operacoes if total_operacoes > 0 else 0
num_bancos = df_filtrado["nome_conglomerado_financeiro"].nunique() if "nome_conglomerado_financeiro" in df_filtrado.columns else df_filtrado["nome_conglomerated_financeiro"].nunique()

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
with kpi1:
    st.markdown(f'<div class="kpi-card"><div class="kpi-title">Volume Carteira Ativa</div><div class="kpi-value">{fmt_brl(total_volume)}</div></div>', unsafe_allow_html=True)
with kpi2:
    st.markdown(f'<div class="kpi-card"><div class="kpi-title">Total de Contratos</div><div class="kpi-value">{fmt_num(total_operacoes)}</div></div>', unsafe_allow_html=True)
with kpi3:
    st.markdown(f'<div class="kpi-card"><div class="kpi-title">Ticket Médio Operacional</div><div class="kpi-value">{fmt_brl(ticket_medio)}</div></div>', unsafe_allow_html=True)
with kpi4:
    st.markdown(f'<div class="kpi-card"><div class="kpi-title">Instituições Credenciadas</div><div class="kpi-value">{fmt_num(num_bancos)}</div></div>', unsafe_allow_html=True)

# ============================================================
# NAVEGAÇÃO EM ABAS (Melhor prática para C-Level)
# ============================================================
aba_temporal, aba_market, aba_regional, aba_insights = st.tabs([
    "📈 Evolução & Projeções", 
    "🏦 Análise de Market Share", 
    "🗺️ Distribuição Regional", 
    "🧠 Diagnóstico Avançado"
])

# ------------------------------------------------------------
# ABA 1: EVOLUÇÃO E PROJEÇÕES
# ------------------------------------------------------------
with aba_temporal:
    col_t_left, col_t_right = st.columns([2, 1])
    
    with col_t_left:
        st.markdown("### Histórico de Volume vs Média Móvel")
        evolucao = df_filtrado.groupby("data_base").agg({"volume_operacoes": "sum", "numero_operacoes": "sum"}).reset_index()
        evolucao["crescimento"] = evolucao["volume_operacoes"].pct_change() * 100
        evolucao["media_movel"] = evolucao["volume_operacoes"].rolling(window=3 if len(evolucao) >=3 else 1).mean()
        
        fig_ev = go.Figure()
        fig_ev.add_trace(go.Scatter(x=evolucao["data_base"], y=evolucao["volume_operacoes"], mode="lines+markers", name="Volume Período", line=dict(color=COR_SECUNDARIA, width=3)))
        fig_ev.add_trace(go.Scatter(x=evolucao["data_base"], y=evolucao["media_movel"], mode="lines", name="Média Móvel (Trimensal)", line=dict(color=COR_TEXTO, dash="dash")))
        st.plotly_chart(aplicar_layout_grafico(fig_ev), use_container_width=True)

    with col_t_right:
        st.markdown("### Desempenho Mensal")
        cresc_tabela = evolucao[["data_base", "crescimento"]].dropna().tail(5).copy()
        cresc_tabela["data_base"] = cresc_tabela["data_base"].dt.strftime("%m/%Y")
        cresc_tabela["crescimento"] = cresc_tabela["crescimento"].apply(lambda x: f"{x:+.2f}%")
        cresc_tabela.columns = ["Competência", "Variação MoM"]
        st.dataframe(cresc_tabela, use_container_width=True, hide_index=True)

    # Projeção Linear Simples
    if len(evolucao) > 3:
        st.markdown("---")
        st.markdown("### Modelo Preditivo Linear (Próximo Trimestre)")
        modelo = evolucao.dropna().copy()
        modelo["indice"] = np.arange(len(modelo))
        
        lr = LinearRegression()
        lr.fit(modelo[["indice"]], modelo["volume_operacoes"])
        
        futuro = pd.DataFrame({"indice": np.arange(len(modelo), len(modelo) + 3)})
        previsao = lr.predict(futuro)
        datas_futuras = pd.date_range(evolucao["data_base"].max(), periods=4, freq="MS")[1:]
        
        fig_prev = go.Figure()
        fig_prev.add_trace(go.Scatter(x=evolucao["data_base"], y=evolucao["volume_operacoes"], name="Histórico Realizado", line=dict(color=COR_SECUNDARIA)))
        fig_prev.add_trace(go.Scatter(x=datas_futuras, y=previsao, name="Tendência Projetada", line=dict(color=COR_ALERTA, dash="dot")))
        st.plotly_chart(aplicar_layout_grafico(fig_prev), use_container_width=True)

# ------------------------------------------------------------
# ABA 2: MARKET SHARE E CONCENTRAÇÃO (HHI & PARETO)
# ------------------------------------------------------------
with aba_market:
    col_m1, col_m2 = st.columns(2)
    col_banco_nome = "nome_conglomerado_financeiro" if "nome_conglomerado_financeiro" in df_filtrado.columns else "nome_conglomerated_financeiro"
    
    market = df_filtrado.groupby(col_banco_nome)["numero_operacoes"].sum().sort_values(ascending=False).reset_index()
    
    with col_m1:
        st.markdown("### Índice de Concentração de Mercado (HHI)")
        hhi = calcular_hhi(market, "numero_operacoes")
        classificacao, classe_css = interpretar_hhi(hhi)
        
        st.markdown(f"""
        <div style="background:{COR_CARD}; border:1px solid {COR_BORDA}; padding: 20px; border-radius:6px;">
            <p style="margin:0; font-size:0.9rem; color:gray;">Índice Herfindahl-Hirschman Cumulado</p>
            <h2 style="margin:5px 0 15px 0; font-size:2.5rem; font-weight:700;">{hhi:.0f}</h2>
            <span class="badge {classe_css}">{classificacao}</span>
        </div>
        """, unsafe_allow_html=True)
        
    with col_m2:
        st.markdown("### Estrutura de Pareto (Top 10 Players)")
        pareto_data = calcular_pareto(market.head(10), "numero_operacoes")
        
        fig_pareto = go.Figure()
        fig_pareto.add_trace(go.Bar(x=pareto_data[col_banco_nome], y=pareto_data["numero_operacoes"], name="Contratos", marker_color=COR_SECUNDARIA))
        fig_pareto.add_trace(go.Scatter(x=pareto_data[col_banco_nome], y=pareto_data["percentual_acumulado"], name="% Acumulado", yaxis="y2", mode="lines+markers", line=dict(color=COR_ALERTA, width=2)))
        
        fig_pareto.update_layout(yaxis2=dict(overlaying="y", side="right", range=[0, 105], showgrid=False))
        st.plotly_chart(aplicar_layout_grafico(fig_pareto), use_container_width=True)

    st.markdown("---")
    st.markdown("### Matriz de Dispersão: Segmentos de Atuação")
    
    dispersao = df_filtrado.groupby(col_banco_nome).agg({"numero_operacoes": "sum", "volume_operacoes": "sum", "tipo_banco": "first"}).reset_index()
    dispersao["ticket_medio"] = dispersao["volume_operacoes"] / dispersao["numero_operacoes"]
    dispersao = dispersao[dispersao["numero_operacoes"] > 100]
    
    fig_disp = px.scatter(dispersao, x="numero_operacoes", y="ticket_medio", color="tipo_banco", size="volume_operacoes",
                          hover_name=col_banco_nome, labels={"numero_operacoes": "Quantidade de Contratos", "ticket_medio": "Ticket Médio (R$)"},
                          color_discrete_map={"Banco Digital": "#10B981", "Banco Tradicional": "#2563EB", "Demais Categorias": "#64748B"})
    st.plotly_chart(aplicar_layout_grafico(fig_disp), use_container_width=True)

# ------------------------------------------------------------
# ABA 3: DISTRIBUIÇÃO REGIONAL
# ------------------------------------------------------------
with aba_regional:
    col_r1, col_r2 = st.columns(2)
    
    with col_r1:
        st.markdown("### Volume Financeiro Alocado por Região")
        regiao_data = df_filtrado.groupby("regiao")["volume_operacoes"].sum().reset_index()
        fig_reg = px.pie(regiao_data, names="regiao", values="volume_operacoes", hole=0.6, color_discrete_sequence=[COR_SECUNDARIA, "#475569", "#94A3B8", "#CBD5E1", "#E2E8F0"])
        st.plotly_chart(aplicar_layout_grafico(fig_reg), use_container_width=True)
        
    with col_r2:
        st.markdown("### Heatmap de Safra Regional (Milhões R$)")
        heat = df_filtrado.groupby(["regiao", df_filtrado["data_base"].dt.strftime("%m/%Y")])["volume_operacoes"].sum().reset_index()
        pivot = heat.pivot(index="regiao", columns="data_base", values="volume_operacoes") / 1_000_000
        fig_heat = px.imshow(pivot, aspect="auto", text_auto=".1f", color_continuous_scale="Blues")
        st.plotly_chart(aplicar_layout_grafico(fig_heat), use_container_width=True)

    st.markdown("---")
    st.markdown("### Eficiência de Escopo: Liderança por Estado (Top 3 Instituições)")
    banco_por_uf = df_filtrado.groupby(["unidade_federacao", col_banco_nome])["numero_operacoes"].sum().reset_index()
    banco_por_uf = banco_por_uf.sort_values(["unidade_federacao", "numero_operacoes"], ascending=[True, False])
    top3_por_uf = banco_por_uf.groupby("unidade_federacao").head(3).reset_index(drop=True)
    top3_por_uf["ranking"] = top3_por_uf.groupby("unidade_federacao").cumcount() + 1
    top3_por_uf["exibicao"] = top3_por_uf.apply(lambda x: f"{x[col_banco_nome]} ({fmt_num(x['numero_operacoes'])})", axis=1)
    
    tabela_lideranca = top3_por_uf.pivot_table(index="unidade_federacao", columns="ranking", values="exibicao", aggfunc="first").reset_index()
    tabela_lideranca.columns = ["Estado (UF)", "Dominante (1º)", "2º Player", "3º Player"]
    st.dataframe(tabela_lideranca, use_container_width=True, hide_index=True)

# ------------------------------------------------------------
# ABA 4: DIAGNÓSTICO AVANÇADO & EXTRAÇÃO
# ------------------------------------------------------------
with aba_insights:
    st.markdown("### Sumário Executivo Automatizado")
    
    crescimento_medio = evolucao["crescimento"].mean() if 'evolucao' in locals() else 0
    maior_regiao = regiao_data.sort_values("volume_operacoes", ascending=False).iloc[0]["regiao"] if not regiao_data.empty else "N/A"
    banco_lider = market.iloc[0][col_banco_nome] if not market.empty else "N/A"
    participacao_lider = (market.iloc[0]["numero_operacoes"] / total_operacoes * 100) if not market.empty else 0

    col_ins1, col_ins2 = st.columns(2)
    with col_ins1:
        if crescimento_medio > 0:
            st.info(f"📈 **Aceleração de Carteira:** O programa demonstra expansão contínua com média MoM de **+{crescimento_medio:.2f}%**.")
        else:
            st.warning(f"📉 **Fase de Desaceleração:** Ritmo de novas renegociações apresenta retração média mensal de **{crescimento_medio:.2f}%**.")
            
        st.markdown(f"📍 **Concentração Geográfica:** A região **{maior_regiao}** absorve o maior market share de volume liquidado.")
        
    with col_ins2:
        st.markdown(f"🏆 **Market Share Liderança:** A instituição **{banco_lider}** retém a liderança em originação com **{participacao_lider:.2f}%** do volume total de contratos.")
        if hhi > 2500:
            st.error("⚠️ **Alerta de Risco:** O mercado atual aponta para índices severos de oligopólio. Recomenda-se acompanhamento de risco de crédito sistêmico.")

    st.markdown("---")
    st.markdown("### Exportação de Relatório Técnico")
    csv = df_filtrado.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Descarregar Dataset Filtrado (Formato CSV para Auditoria)",
        data=csv,
        file_name=f"BACEN_Desenrola_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
        use_container_width=True
    )

# ============================================================
# COMPLIANCE FOOTER
# ============================================================
st.markdown("---")
st.markdown("<p style='text-align: center; color: #64748B; font-size: 0.75rem;'>Aviso Legal: Os dados apresentados decorrem de repositórios públicos informados pelas instituições financeiras ao Banco Central do Brasil. Os modelos estatísticos preditivos constituem estimativas de tendência e não garantem retornos ou comportamentos futuros de carteira.</p>", unsafe_allow_html=True)
