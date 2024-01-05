import hydra
import os
import joblib
import mlflow
import mlflow.sklearn

from omegaconf import DictConfig
from sklearn.linear_model import LogisticRegression

from train_model import get_metrics, run_training
from helper import BaseLogger
from preprocessors import ImageTransformation

logger = BaseLogger()

""" Aqui se agregan todas las funciones necesarias para evaluar un modelo """
def execute_model(config: DictConfig):
    input_dir = config.raw.path
    categories = config.raw.types
    evaluations_number = config.model.evaluations_number
    sampling_size = config.model.sampling_size
    test_size = config.model.test_size

    # Indica la url en donde esta el entorno de MLFlow local o remoto
    mlflow.set_tracking_uri(config.mlflow.tracking_ui)
    os.environ['MLFLOW_TRACKING_USERNAME'] = config.mlflow.username
    os.environ['MLFLOW_TRACKING_PASSWORD'] = config.mlflow.password
    with mlflow.start_run() as run:

        # Se obtiene el modelo a usar
        model = LogisticRegression()

        # Se realiza el pre-procesamiento de los datos
        image_normalizer = ImageTransformation(input_dir, categories)

        # Se entrena el modelo
        confusion_matrix = run_training(
            model, image_normalizer, evaluations_number, sampling_size, test_size
        )
        model_path = config.model.path

        # Se guarda el modelo
        joblib.dump(model, model_path)
        logger.log_model(model, config.model.artifact_path, config.model.name)

        # Se obtienen las metricas
        results = get_metrics(confusion_matrix)

    # Muestra los resultados despues de haber entrenado y validado el modelo
    print(f'Accuracy Final: {results["accuracy"] * 100}')
    print(f'Precision Final: {results["precision"] * 100}')
    print(f'Recall Final: {results["recall"] * 100}')