import pandas as pd
import mlflow
from pycaret.classification import setup, create_model, pull, save_model

df = pd.read_parquet("./data/processed/data_com_features.parquet")

mlflow.set_experiment("ModeloComFeatures")

with mlflow.start_run(run_name="Treinamento_RF"):
    setup(data=df, target="shot_made_flag", session_id=42,
          log_experiment=True, experiment_name="ModeloComFeatures",
          log_plots=True, log_data=False, verbose=False)

    model = create_model("rf")  # Random Forest

    # Log de métricas
    metrics = pull()
    mlflow.log_param("modelo", "random_forest")
    mlflow.log_metrics(metrics.iloc[0].to_dict())

    save_model(model, "modelo_kobe_rf")
    mlflow.log_artifact("modelo_kobe_rf.pkl")

    print("Modelo Random Forest treinado e salvo.")
