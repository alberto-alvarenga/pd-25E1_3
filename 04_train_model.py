import pandas as pd
import mlflow
from pycaret.classification import setup, compare_models, pull, create_model, evaluate_model, save_model

# Definições iniciais
EXPERIMENTO = "PipelineKobe"
RUN_NAME = "Treinamento"
MODEL_NAME = "logistic_regression"
DATA_PATH = "./data/processed/base_train.parquet"

# Carrega os dados de treino
df_train = pd.read_parquet(DATA_PATH)

# Inicia experimento no MLflow
mlflow.set_experiment(EXPERIMENTO)

with mlflow.start_run(run_name=RUN_NAME):

    # Configuração do PyCaret
    s = setup(
        data=df_train,
        target="shot_made_flag",
        #silent=True,
        session_id=42,
        log_experiment=True,
        experiment_name=EXPERIMENTO,
        log_plots=True,
        log_data=False
    )

    # Cria modelo de regressão logística
    model = create_model("lr")  # Logistic Regression

    # Avaliação (opcional com GUI PyCaret)
    # evaluate_model(model)

    # Puxa as métricas de avaliação
    metrics_df = pull()

    # Log de parâmetros e métricas no MLflow
    mlflow.log_param("modelo", "logistic_regression")
    mlflow.log_metrics(metrics_df.iloc[0].to_dict())

    # Salvar modelo
    save_model(model, "modelo_kobe_logistic")
    mlflow.log_artifact("modelo_kobe_logistic.pkl")

    print("Modelo treinado e registrado com sucesso!")
