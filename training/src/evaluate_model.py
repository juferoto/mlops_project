import hydra
import joblib

import os
import mlflow
import mlflow.sklearn
import numpy as np
from hydra.utils import to_absolute_path as abspath

from omegaconf import DictConfig
from sklearn.metrics import accuracy_score, precision_score, recall_score
from helper import BaseLogger

""" Aqui se agregan todas las funciones necesarias para evaluar un modelo """
logger = BaseLogger()

def load_data(path: DictConfig):
    x_train = np.load(abspath(path.x_train.path))
    x_test = np.load(abspath(path.x_test.path))
    y_test = np.load(abspath(path.y_test.path))
    return x_train, x_test, y_test

def log_params(params: object):
    for arg, value in params.items():
        logger.log_params({arg: value})

@hydra.main(version_base=None, config_path="../../config", config_name="main")
def evaluate(config: DictConfig):
    test_size_percentage = config.model.test_size_percentage

    # Indica la url en donde esta el entorno de MLFlow local o remoto
    mlflow.set_tracking_uri(config.mlflow.tracking_ui)
    os.environ['MLFLOW_TRACKING_USERNAME'] = config.mlflow.username
    os.environ['MLFLOW_TRACKING_PASSWORD'] = config.mlflow.password
    with mlflow.start_run():

        # Se cargan los datos de pruebas
        x_train, x_test, y_test = load_data(config.processed)
        x_train = x_train.reshape(-1, x_train.shape[2])
        x_test = x_test.reshape(-1, x_test.shape[2])

        # Se obtiene el modelo entrenado
        model = joblib.load(abspath(config.model.path))

        # Obtener prediccion
        prediction = model.predict(x_test)

        # Se calculan algunas metricas
        accuracy = accuracy_score(y_test.ravel(), prediction)
        precision = precision_score(y_test.ravel(), prediction)
        recall = recall_score(y_test.ravel(), prediction)

        # Se obtienen y se guardan los parametros del modelo
        data_total_size = len(x_train) + len(x_test)
        params = {
            'dataset_size': data_total_size,
            'test_size_percentage': test_size_percentage
        }

        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall
        }

        log_params(params)
        logger.log_model(model, config.model.artifact_path, config.model.name)
        logger.log_metrics(metrics)

    # Muestra los resultados despues de haber entrenado y validado el modelo
    print(f'Accuracy: {metrics["accuracy"] * 100}')
    print(f'Precision: {metrics["precision"] * 100}')
    print(f'Recall: {metrics["recall"] * 100}')

if __name__ == "__main__":
    evaluate()