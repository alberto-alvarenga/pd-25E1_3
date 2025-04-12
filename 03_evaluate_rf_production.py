import pandas as pd
from pycaret.classification import load_model
from sklearn.metrics import f1_score, log_loss

model = load_model("modelo_kobe_rf")

# Base de produção
df_prod = pd.read_parquet("./data/01_raw/dataset_kobe_prod.parquet")
cols = [
    "lat", "lon", "minutes_remaining", "period", "playoffs", "shot_distance",
    "action_type", "combined_shot_type", "shot_type", "season", "shot_made_flag"
]
df_prod = df_prod[cols].dropna()
X_prod = df_prod.drop("shot_made_flag", axis=1)
y_prod = df_prod["shot_made_flag"]

# Probabilidades
probs = model.predict_proba(X_prod)[:, 1]

# Testar thresholds
print("Avaliação com múltiplos thresholds:")
for t in [0.3, 0.4, 0.5, 0.6, 0.7]:
    y_pred = (probs >= t).astype(int)
    f1 = f1_score(y_prod, y_pred)
    ll = log_loss(y_prod, probs)
    print(f"Threshold {t:.1f} → F1: {f1:.4f} | Log Loss: {ll:.4f}")
