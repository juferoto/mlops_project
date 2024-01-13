import os
import mlflow
from hydra import compose, initialize

def get_model_version(modelRegisteredVersions: any, stage: str):
    modelRegistered = next(filter(lambda obj: obj.current_stage == stage, modelRegisteredVersions), None)
    return modelRegistered.version

def set_output(name, value):
    with open(os.environ['GITHUB_OUTPUT'], 'a') as fh:
        print(f'{name}={value}', file=fh)

def get_variables():

    with initialize(version_base=None, config_path="../../config"):
        config = compose(config_name="main")

    mlflow.set_tracking_uri(config.mlflow.tracking_ui)
    os.environ['MLFLOW_TRACKING_USERNAME'] = config.mlflow.username
    os.environ['MLFLOW_TRACKING_PASSWORD'] = config.mlflow.password

    stage = "Production"
    client = mlflow.MlflowClient()
    model_registered = client.get_registered_model(config.model.name)
    model_version = get_model_version(model_registered.latest_versions, stage)
    
    env_file = os.getenv('GITHUB_OUTPUT')
    if env_file is not None:
        # Escribe las variables a exportar
        set_output("MODEL_NAME", config.model.name)
        set_output("MODEL_VERSION", model_version)

if __name__ == "__main__":
    get_variables()
