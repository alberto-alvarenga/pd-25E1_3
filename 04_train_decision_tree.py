import pandas as pd
import mlflow
from pycaret.classification import setup, create_model, pull, save_model

# Configurações
EXPERIMENTO = "PipelineKobe"
RUN_NAME = "TreinamentoArvore"
MODEL_NAME = "decision_tree"
DATA_PATH = "./data/processed/base_train.parquet"

# Carregar base de treino
df_train = pd.read_parquet(DATA_PATH)

# Iniciar experimento no MLflow
mlflow.set_experiment(EXPERIMENTO)

with mlflow.start_run(run_name=RUN_NAME):
    
    # Configuração do ambiente PyCaret
    setup(
        data=df_train,
        target="shot_made_flag",
        session_id=42,
        log_experiment=True,
        experiment_name=EXPERIMENTO,
        log_plots=True,
        log_data=False,
        verbose=False
    )

    # Criar modelo de árvore de decisão
    model = create_model("dt")  # "dt" = Decision Tree

    # Capturar métricas
    metrics_df = pull()

    # Log de parâmetros e métricas no MLflow
    mlflow.log_param("modelo", MODEL_NAME)
    mlflow.log_metrics(metrics_df.iloc[0].to_dict())

    # Salvar modelo treinado
    save_model(model, "modelo_kobe_tree")
    mlflow.log_artifact("modelo_kobe_tree.pkl")

    print("Modelo de árvore de decisão treinado e registrado no MLflow.")
