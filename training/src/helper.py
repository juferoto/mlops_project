import mlflow
from dagshub import DAGsHubLogger

""" Clase base para hacer el logueo de la informacion usando MLFlow y DagsHub """
class BaseLogger:
    def __init__(self):
        self.logger = DAGsHubLogger()

    def log_metrics(self, metrics: dict):
        mlflow.log_metrics(metrics)
        self.logger.log_metrics(metrics)

    def log_params(self, params: dict):
        mlflow.log_params(params)
        self.logger.log_hyperparams(params)

    def log_model(self, model: object, artifact_path: str, model_name: str):
        mlflow.sklearn.log_model(model, 
                                artifact_path,
                                registered_model_name=model_name)
        
    def log_artifact(self, model_name: str):
        mlflow.log_artifact(model_name)