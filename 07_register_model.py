import mlflow.pyfunc
from pycaret.classification import load_model

# Carregar modelo final (ex: regressão logística)
model = load_model("modelo_kobe_logistic")

# Registrar no MLflow como modelo
mlflow.set_experiment("PipelineKobe")

with mlflow.start_run(run_name="RegistroModeloFinal"):
    mlflow.sklearn.log_model(model, artifact_path="modelo_final", registered_model_name="modelo_kobe_final")
    print("Modelo registrado como 'modelo_kobe_final'")
