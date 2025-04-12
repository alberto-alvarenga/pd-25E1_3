import pandas as pd
from sklearn.metrics import log_loss, f1_score
from pycaret.classification import load_model

# Carregar modelos treinados
model_logistic = load_model("modelo_kobe_logistic")
model_tree = load_model("modelo_kobe_tree")

# Carregar base de teste
df_test = pd.read_parquet("./data/processed/base_test.parquet")
X_test = df_test.drop("shot_made_flag", axis=1)
y_test = df_test["shot_made_flag"]

# Regressão logística
preds_log = model_logistic.predict(X_test)
probs_log = model_logistic.predict_proba(X_test)[:, 1]
log_loss_log = log_loss(y_test, probs_log)
f1_log = f1_score(y_test, preds_log)

# Árvore de decisão
preds_tree = model_tree.predict(X_test)
probs_tree = model_tree.predict_proba(X_test)[:, 1]
log_loss_tree = log_loss(y_test, probs_tree)
f1_tree = f1_score(y_test, preds_tree)

# Resultados
print("Comparação de Modelos:")
print(f"Regressão Logística - Log Loss: {log_loss_log:.4f} | F1-Score: {f1_log:.4f}")
print(f"Árvore de Decisão   - Log Loss: {log_loss_tree:.4f} | F1-Score: {f1_tree:.4f}")
