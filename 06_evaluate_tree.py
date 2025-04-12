import pandas as pd
import mlflow
from sklearn.metrics import log_loss, f1_score
from pycaret.classification import load_model

# Configurações
EXPERIMENTO = "PipelineKobe"
RUN_NAME = "AvaliacaoArvore"
MODEL_PATH = "modelo_kobe_tree"
TEST_PATH = "./data/processed/base_test.parquet"

# Carregar modelo
print("Carregando modelo de árvore...")
model = load_model(MODEL_PATH)
print("Modelo carregado com sucesso.")

# Carregar base de teste
df_test = pd.read_parquet(TEST_PATH)
X_test = df_test.drop("shot_made_flag", axis=1)
y_test = df_test["shot_made_flag"]

# Previsões
y_pred = model.predict(X_test)
probs = model.predict_proba(X_test)[:, 1]  # Probabilidade da classe 1

# Calcular métricas
logloss = log_loss(y_test, probs)
f1 = f1_score(y_test, y_pred)

# Log no MLflow
mlflow.set_experiment(EXPERIMENTO)

with mlflow.start_run(run_name=RUN_NAME):
    mlflow.log_metric("log_loss", logloss)
    mlflow.log_metric("f1_score", f1)
    print(f"Log Loss: {logloss:.4f}")
    print(f"F1 Score: {f1:.4f}")
