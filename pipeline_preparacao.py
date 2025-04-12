import pandas as pd
from sklearn.model_selection import train_test_split
import mlflow
import os

# Caminhos dos dados
RAW_PATH = "data/01_raw/dataset_kobe_dev.parquet"
PROCESSED_PATH = "data/processed/data_filtered.parquet"
TRAIN_PATH = "data/processed/base_train.parquet"
TEST_PATH = "data/processed/base_test.parquet"

# Criar diretórios, se não existirem
os.makedirs("data/processed", exist_ok=True)

# Iniciar MLflow local
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("Kobe_Preditor")

def pipeline_preparacao():
    with mlflow.start_run(run_name="PreparacaoDados"):
        # 1. Carregar os dados brutos
        df = pd.read_parquet(RAW_PATH)

        # 2. Selecionar colunas relevantes
        colunas = [
            "lat",
            "lon",
            "minutes_remaining",
            "period",
            "playoffs",
            "shot_distance",
            "shot_made_flag"
        ]
        df = df[colunas]

        # 3. Remover nulos
        df = df.dropna()

        # 4. Salvar dataset filtrado
        df.to_parquet(PROCESSED_PATH, index=False)

        # 5. Separar X e y
        X = df.drop(columns=["shot_made_flag"])
        y = df["shot_made_flag"]

        # 6. Split treino/teste estratificado
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            stratify=y,
            random_state=42
        )

        # 7. Salvar datasets
        base_train = X_train.copy()
        base_train["shot_made_flag"] = y_train
        base_train.to_parquet(TRAIN_PATH, index=False)

        base_test = X_test.copy()
        base_test["shot_made_flag"] = y_test
        base_test.to_parquet(TEST_PATH, index=False)

        # 8. Log de parâmetros e métricas no MLflow
        mlflow.log_param("proporcao_teste", 0.2)
        mlflow.log_metric("linhas_filtradas", df.shape[0])
        mlflow.log_metric("linhas_treino", base_train.shape[0])
        mlflow.log_metric("linhas_teste", base_test.shape[0])

        print("Preparação concluída. Tamanho do dataset final:", df.shape)

if __name__ == "__main__":
    pipeline_preparacao()
