import pandas as pd
import os

# Carregar o dataset
df = pd.read_parquet("./data/01_raw/dataset_kobe_dev.parquet")

# Colunas selecionadas
cols = [
    "lat", "lon", "minutes_remaining", "period", "playoffs", "shot_distance",
    "action_type", "combined_shot_type", "shot_type", "season", "shot_made_flag"
]

# Selecionar e limpar
df = df[cols].dropna()

# Salvar
os.makedirs("./data/processed", exist_ok=True)
df.to_parquet("./data/processed/data_com_features.parquet", index=False)
print("✅ Dados com novas features salvos.")
