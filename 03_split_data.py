import pandas as pd
from sklearn.model_selection import train_test_split
import mlflow
import os

# --- Configurações ---
EXPERIMENTO = "PreparacaoDados"
RUN_NAME = "SeparacaoTreinoTeste"
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Caminhos
input_path = "./data/processed/data_filtered.parquet"
train_path = "./data/processed/base_train.parquet"
test_path = "./data/processed/base_test.parquet"

# Criar pasta se necessário
os.makedirs("./data/processed", exist_ok=True)

# Iniciar experimento
mlflow.set_experiment(EXPERIMENTO)

with mlflow.start_run(run_name=RUN_NAME):
    # 1. Carregar os dados filtrados
    df = pd.read_parquet(input_path)

    # 2. Separar features (X) e target (y)
    X = df.drop("shot_made_flag", axis=1)
    y = df["shot_made_flag"]

    # 3. Divisão estratificada
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    # 4. Combinar e salvar as bases
    df_train = pd.concat([X_train, y_train], axis=1)
    df_test = pd.concat([X_test, y_test], axis=1)
    df_train.to_parquet(train_path, index=False)
    df_test.to_parquet(test_path, index=False)

    # 5. Log no MLflow
    mlflow.log_param("test_size_percentual", TEST_SIZE)
    mlflow.log_metric("train_size", len(df_train))
    mlflow.log_metric("test_size_absolute", len(df_test))

    print(f"Base de treino: {len(df_train)} registros")
    print(f"Base de teste: {len(df_test)} registros")
