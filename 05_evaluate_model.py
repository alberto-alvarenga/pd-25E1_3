import pandas as pd
import mlflow
from sklearn.metrics import log_loss
from pycaret.classification import load_model

# --- Configurações ---
EXPERIMENTO = "PipelineKobe"
RUN_NAME = "AvaliacaoFinal"
MODEL_PATH = "modelo_kobe_logistic"
TEST_PATH = "./data/processed/base_test.parquet"

# --- Carregar modelo salvo com PyCaret ---
print("Carregando modelo...")
model = load_model(MODEL_PATH)
print("Modelo carregado com sucesso.")

# --- Carregar base de teste ---
df_test = pd.read_parquet(TEST_PATH)
X_test = df_test.drop("shot_made_flag", axis=1)
y_test = df_test["shot_made_flag"]

# --- Obter probabilidades da classe 1 diretamente do modelo ---
probs = model.predict_proba(X_test)[:, 1]

# --- Calcular Log Loss ---
logloss_value = log_loss(y_test, probs)

# --- Registrar no MLflow ---
mlflow.set_experiment(EXPERIMENTO)

with mlflow.start_run(run_name=RUN_NAME):
    mlflow.log_metric("log_loss", logloss_value)
    print(f"✅ Log Loss registrado no MLflow: {logloss_value:.4f}")
