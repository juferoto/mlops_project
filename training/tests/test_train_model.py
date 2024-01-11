import os
import mlflow
import numpy as np
from hydra import compose, initialize
from omegaconf import DictConfig
from sklearn.metrics import accuracy_score, precision_score, recall_score

""" Aqui se agregan todas las funciones necesarias para realizar pruebas sobre un modelo ML """
def get_model_version(modelRegisteredVersions: any, stage: str):
    modelRegistered = next(filter(lambda obj: obj.current_stage == stage, modelRegisteredVersions), None)
    return modelRegistered.version

def get_model(config: DictConfig):
    # Se agrega la direccion en donde esta alojado MLFlow sea remoto o local
    mlflow.set_tracking_uri(config.mlflow.tracking_ui)
    os.environ['MLFLOW_TRACKING_USERNAME'] = config.mlflow.username
    os.environ['MLFLOW_TRACKING_PASSWORD'] = config.mlflow.password

    # Obtiene el modelo que fue asignado al entorno de Produccion
    stage = "Production"
    model_uri=f"models:/{config.model.name}/{stage}"
    model = mlflow.pyfunc.load_model(model_uri)

    client = mlflow.MlflowClient()
    model_registered = client.get_registered_model(config.model.name)
    model_version = get_model_version(model_registered.latest_versions, stage)
    os.environ["MODEL_NAME"] = config.model.name
    os.environ["MODEL_VERSION"] = model_version

    # Exportar las variables de entorno para GitHub Actions
    output_line_model = f"{'MODEL_NAME'}={config.model.name}"
    output_line_version = f"{'MODEL_VERSION'}={model_version}"

    print(f"echo '{output_line_model}' >> $GITHUB_OUTPUT")
    print(f"echo '{output_line_version}' >> $GITHUB_OUTPUT")
    return model

def load_data(path: DictConfig):
    x_train = np.load(path.x_train.path)
    x_test = np.load(path.x_test.path)
    y_test = np.load(path.y_test.path)
    return x_train, x_test, y_test

def test_logistic_regression():

    with initialize(version_base=None, config_path="../../config"):
        config = compose(config_name="main")

    # Se cargan los datos de pruebas
    x_train, x_test, y_test = load_data(config.processed)
    x_train = x_train.reshape(-1, x_train.shape[2])
    x_test = x_test.reshape(-1, x_test.shape[2])

    # Se obtiene el modelo entrenado
    model = get_model(config)

    # Obtener prediccion
    prediction = model.predict(x_test)
    
    # Se calculan algunas metricas
    accuracy = accuracy_score(y_test.ravel(), prediction)
    precision = precision_score(y_test.ravel(), prediction)
    recall = recall_score(y_test.ravel(), prediction)

    # Realizar aserciones sobre los resultados
    assert accuracy >= 0.7 and precision >= 0.7 and recall >= 0.7