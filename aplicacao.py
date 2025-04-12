import pandas as pd
import mlflow
from sklearn.metrics import log_loss, f1_score
from pycaret.classification import load_model

# Configurações
EXPERIMENTO = "PipelineKobe"
RUN_NAME = "PipelineAplicacao"
MODEL_PATH = "modelo_kobe_logistic"  # Pode ser o caminho do modelo final salvo
DATA_PROD_PATH = "./data/01_raw/dataset_kobe_prod.parquet"
RESULT_PATH = "./data/processed/resultados_producao.parquet"

# Carregar modelo
model = load_model(MODEL_PATH)

# Carregar dados de produção
df_prod = pd.read_parquet(DATA_PROD_PATH)

# Selecionar mesmas colunas usadas no treino
cols = ["lat", "lon", "minutes_remaining", "period", "playoffs", "shot_distance", "shot_made_flag"]
df_prod = df_prod[cols].dropna()

# Separar features e target
X_prod = df_prod.drop("shot_made_flag", axis=1)
y_prod = df_prod["shot_made_flag"]

# Fazer previsões
y_pred = model.predict(X_prod)
probs = model.predict_proba(X_prod)[:, 1]

# Calcular métricas
logloss = log_loss(y_prod, probs)
f1 = f1_score(y_prod, y_pred)

# Salvar resultados em arquivo
df_result = X_prod.copy()
df_result["real"] = y_prod
df_result["previsto"] = y_pred
df_result["probabilidade"] = probs
df_result.to_parquet(RESULT_PATH, index=False)

# Log no MLflow
mlflow.set_experiment(EXPERIMENTO)

with mlflow.start_run(run_name=RUN_NAME):
    mlflow.log_metric("log_loss_producao", logloss)
    mlflow.log_metric("f1_score_producao", f1)
    mlflow.log_artifact(RESULT_PATH)
    print(f"Log Loss produção: {logloss:.4f}")
    print(f"F1 Score produção: {f1:.4f}")
    print(f"Resultados salvos em {RESULT_PATH}")
