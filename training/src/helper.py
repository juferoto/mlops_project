import mlflow

class BaseLogger:

    def log_metrics(self, metrics: dict):
        mlflow.log_metrics(metrics)

    def log_params(self, params: dict):
        mlflow.log_params(params)

    def log_model(self, model: object, model_name: str):
        mlflow.sklearn.log_model(model, model_name)