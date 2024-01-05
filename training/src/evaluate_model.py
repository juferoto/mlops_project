import hydra
import os
import mlflow
import mlflow.sklearn
import numpy as np

from omegaconf import DictConfig

from train_model import train
from helper import BaseLogger

logger = BaseLogger()

""" Aqui se agregan todas las funciones necesarias para evaluar un modelo """
def log_params(model: object, features: dict):
    logger.log_params({"model_class": type(model).__name__})
    model_params = model.get_params()

    for arg, value in model_params.items():
        logger.log_params({arg: value})

    logger.log_params({"features": features})

@hydra.main(version_base=None, config_path="../../config", config_name="main")
def evaluate(config: DictConfig):

    # Indica la url en donde esta el entorno de MLFlow local o remoto
    mlflow.set_tracking_uri(config.mlflow.tracking_ui)
    os.environ['MLFLOW_TRACKING_USERNAME'] = config.mlflow.username
    os.environ['MLFLOW_TRACKING_PASSWORD'] = config.mlflow.password
    with mlflow.start_run() as run:

        # Se entrena y se obtiene el modelo despues de evaluarlo
        confusion_matrixs, model, params = train(config)

        # Se obtienen las metricas
        metrics = get_metrics(confusion_matrixs)

        log_params(model, params)
        logger.log_model(model, config.model.artifact_path, config.model.name)
        logger.log_metrics(metrics)
        logger.log_artifact(config.model.path)

    # Muestra los resultados despues de haber entrenado y validado el modelo
    print(f'Accuracy Final: {metrics["accuracy"] * 100}')
    print(f'Precision Final: {metrics["precision"] * 100}')
    print(f'Recall Final: {metrics["recall"] * 100}')

def get_metrics(matrices_confusion):
    matrices_confusion_promedio = np.mean(matrices_confusion, axis=0)

    recall = matrices_confusion_promedio[1, 1] / (matrices_confusion_promedio[1, 1] + matrices_confusion_promedio[1, 0])
    accuracy = (matrices_confusion_promedio[0, 0] + matrices_confusion_promedio[1, 1]) / np.sum(matrices_confusion_promedio)
    precision = matrices_confusion_promedio[1, 1] / (matrices_confusion_promedio[1, 1] + matrices_confusion_promedio[0, 1])

    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall
    }

    logger.log_metrics(metrics)
    return metrics

if __name__ == "__main__":
    evaluate()