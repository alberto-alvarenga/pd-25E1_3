import pandas as pd
from sklearn.model_selection import train_test_split
import mlflow
import os

# Iniciar experimento
mlflow.set_experiment("PreparacaoDados")

with mlflow.start_run(run_name="SeparacaoTreinoTeste"):

    # Caminho de entrada e saída
    input_path = "./data/processed/data_filtered.parquet"
    path_train = "./data/processed/base_train.parquet"
    path_test = "./data/processed/base_test.parquet"
    os.makedirs("./data/processed", exist_ok=True)

    # Leitura do dataset filtrado
    df = pd.read_parquet(input_path)

    # Separar X (variáveis) e y (alvo)
    X = df.drop("shot_made_flag", axis=1)
    y = df["shot_made_flag"]

    # Divisão estratificada
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Reunir novamente X + y para salvar
    df_train = pd.concat([X_train, y_train], axis=1)
    df_test = pd.concat([X_test, y_test], axis=1)

    # Salvar as bases
    df_train.to_parquet(path_train, index=False)
    df_test.to_parquet(path_test, index=False)

    # Log no MLflow
    mlflow.log_param("test_size", 0.2)
    mlflow.log_metric("train_size", len(df_train))
    mlflow.log_metric("test_size", len(df_test))
