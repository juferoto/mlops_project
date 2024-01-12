import os
import mlflow
import numpy as np
from hydra import compose, initialize
from omegaconf import DictConfig
from sklearn.metrics import accuracy_score, precision_score, recall_score

""" Aqui se agregan todas las funciones necesarias para realizar pruebas sobre un modelo ML """
def get_model(config: DictConfig):
    # Se agrega la direccion en donde esta alojado MLFlow sea remoto o local
    mlflow.set_tracking_uri(config.mlflow.tracking_ui)
    os.environ['MLFLOW_TRACKING_USERNAME'] = config.mlflow.username
    os.environ['MLFLOW_TRACKING_PASSWORD'] = config.mlflow.password

    # Obtiene el modelo que fue asignado al entorno de Produccion
    stage = "Production"
    model_uri=f"models:/{config.model.name}/{stage}"
    model = mlflow.pyfunc.load_model(model_uri)
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
    assert accuracy >= config.global_metric and precision >= config.global_metric and recall >= config.global_metric