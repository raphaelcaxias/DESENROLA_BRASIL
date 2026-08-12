@staticmethod
@st.cache_data
def generate_sample_data(n_records: int = 5000) -> pd.DataFrame:
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
        
        # LINHA CORRIGIDA AQUI:
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
