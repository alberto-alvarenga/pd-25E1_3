import pandas as pd

# Caminho do dataset de produção
df_prod = pd.read_parquet("./data/01_raw/dataset_kobe_prod.parquet")

# Selecionar colunas do modelo
columns = ["lat", "lon", "minutes_remaining", "period", "playoffs", "shot_distance", "shot_made_flag"]
df_prod = df_prod[columns].dropna()

# Verificar a distribuição da variável alvo
distribuicao = df_prod["shot_made_flag"].value_counts(normalize=True)
print("Distribuição da variável 'shot_made_flag' na base de produção:")
for classe, proporcao in distribuicao.items():
    print(f"Classe {int(classe)}: {proporcao*100:.2f}%")
