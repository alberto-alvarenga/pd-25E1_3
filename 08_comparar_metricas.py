import pandas as pd
from sklearn.metrics import log_loss, f1_score
from pycaret.classification import load_model

# Carregar modelo
model = load_model("modelo_kobe_logistic")

# Carregar datasets
df_train = pd.read_parquet("./data/processed/base_train.parquet")
df_test = pd.read_parquet("./data/processed/base_test.parquet")
df_prod = pd.read_parquet("./data/01_raw/dataset_kobe_prod.parquet")

# Padronizar colunas
cols = ["lat", "lon", "minutes_remaining", "period", "playoffs", "shot_distance", "shot_made_flag"]
df_prod = df_prod[cols].dropna()

# Separar X e y
X_train, y_train = df_train.drop("shot_made_flag", axis=1), df_train["shot_made_flag"]
X_test, y_test = df_test.drop("shot_made_flag", axis=1), df_test["shot_made_flag"]
X_prod, y_prod = df_prod.drop("shot_made_flag", axis=1), df_prod["shot_made_flag"]

# Função auxiliar
def avaliar(modelo, X, y):
    probs = modelo.predict_proba(X)[:, 1]
    preds = modelo.predict(X)
    return log_loss(y, probs), f1_score(y, preds)

# Obter métricas
logloss_train, f1_train = avaliar(model, X_train, y_train)
logloss_test, f1_test = avaliar(model, X_test, y_test)
logloss_prod, f1_prod = avaliar(model, X_prod, y_prod)

# Exibir resultados
print(f"TREINAMENTO: Log Loss = {logloss_train:.4f} | F1-Score = {f1_train:.4f}")
print(f"TESTE      : Log Loss = {logloss_test:.4f} | F1-Score = {f1_test:.4f}")
print(f"PRODUÇÃO   : Log Loss = {logloss_prod:.4f} | F1-Score = {f1_prod:.4f}")
