# ===== LIDERANÇA REGIONAL =====
if 'unidade_federacao' in df_filtrado.columns and 'nome_conglomerado_financeiro' in df_filtrado.columns:
    st.markdown(f"""
    <div class="section-header">
        <h2>🗺️ Liderança Regional por Estado</h2>
        <span class="section-badge">Top 3 Bancos</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Função para normalizar texto
    def normalizar_texto(texto):
        if pd.isna(texto):
            return texto
        texto = str(texto)
        import unicodedata
        texto = unicodedata.normalize('NFKD', texto).encode('ASCII', 'ignore').decode('ASCII')
        # Substituições manuais
        substituicoes = {
            'ECONOMICA': 'ECONÔMICA',
            'FEDERAL': 'FEDERAL',
            'CAIXA ECONOMICA FEDERAL': 'CAIXA ECONÔMICA FEDERAL'
        }
        for k, v in substituicoes.items():
            texto = texto.replace(k, v)
        return texto
    
    banco_por_uf = df_filtrado.groupby(['unidade_federacao', 'nome_conglomerado_financeiro'])['numero_operacoes'].sum().reset_index()
    banco_por_uf = banco_por_uf.sort_values(['unidade_federacao', 'numero_operacoes'], ascending=[True, False])
    top3_por_uf = banco_por_uf.groupby('unidade_federacao').head(3).reset_index(drop=True)
    top3_por_uf['ranking'] = top3_por_uf.groupby('unidade_federacao').cumcount() + 1
    top3_por_uf['nome_conglomerado_financeiro'] = top3_por_uf['nome_conglomerado_financeiro'].apply(normalizar_texto)
    top3_por_uf['exibicao'] = top3_por_uf.apply(lambda x: f"{x['ranking']}º - {x['nome_conglomerado_financeiro']} ({fmt_num(x['numero_operacoes'])})", axis=1)
    
    if len(top3_por_uf) > 0:
        lideranca = top3_por_uf[top3_por_uf['ranking'] == 1].groupby('nome_conglomerado_financeiro').size().sort_values(ascending=False)
        if len(lideranca) > 0:
            st.info(f"🏆 **{lideranca.index[0]}** lidera em {lideranca.iloc[0]} estados, seguido por **{lideranca.index[1] if len(lideranca) > 1 else 'N/A'}** com {lideranca.iloc[1] if len(lideranca) > 1 else 0} estados")
        
        # Tabela formatada
        tabela_lideranca = top3_por_uf.pivot_table(index='unidade_federacao', columns='ranking', values='exibicao', aggfunc='first').reset_index()
        tabela_lideranca.columns = ['UF', '🥇 1º Lugar', '🥈 2º Lugar', '🥉 3º Lugar']
        
        st.dataframe(
            tabela_lideranca,
            use_container_width=True,
            hide_index=True,
            column_config={
                "UF": st.column_config.TextColumn("Estado", width="small"),
                "🥇 1º Lugar": st.column_config.TextColumn("Líder", width="medium"),
                "🥈 2º Lugar": st.column_config.TextColumn("Segundo", width="medium"),
                "🥉 3º Lugar": st.column_config.TextColumn("Terceiro", width="medium"),
            }
        )
