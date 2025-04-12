import pandas as pd
from sklearn.model_selection import train_test_split

# Carregar dataset já filtrado
df = pd.read_parquet("data/processed/data_filtered.parquet")

# Separar features e target
X = df.drop(columns=["shot_made_flag"])
y = df["shot_made_flag"]

# Separação estratificada
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# Reunir as bases
df_train = X_train.copy()
df_train["shot_made_flag"] = y_train

df_test = X_test.copy()
df_test["shot_made_flag"] = y_test

# Salvar arquivos
df_train.to_parquet("data/processed/base_train.parquet", index=False)
df_test.to_parquet("data/processed/base_test.parquet", index=False)

print("Bases salvas:")
print("Train:", df_train.shape)
print("Test:", df_test.shape)
