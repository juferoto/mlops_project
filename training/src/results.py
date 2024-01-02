import hydra
import joblib
from omegaconf import DictConfig

from pipeline import create_pipeline
from train_pipeline import get_metrics, run_training
import mlflow
import mlflow.sklearn

@hydra.main(version_base=None, config_path="../../config", config_name="main")
def show_results(config: DictConfig):
    input_dir = config.raw.path
    categories = config.raw.types
    evaluations_number = config.model.evaluations_number
    sampling_size = config.model.sampling_size
    test_size = config.model.test_size

    # Indica la url en donde esta el entorno de MLFlow local o remoto
    mlflow.set_tracking_uri(config.mlflow_tracking_ui)
    mlflow.set_experiment(config.mlflow_experiment_name)
    with mlflow.start_run() as run:

        # Crea el pipeline
        pipeline = create_pipeline(input_dir, categories)
        matrices_confusion = run_training(
            pipeline, evaluations_number, sampling_size, test_size
        )
        model_name = "model_regression.joblib"

        joblib.dump(pipeline, model_name)
        # mlflow.log_artifact(model_name)

        # Calcula las metricas apartir de todas las matrices de confusión
        results = get_metrics(matrices_confusion)

    # Calcular y mostrar las métricas finales
    print(f'Accuracy Final: {results["accuracy"] * 100}')
    print(f'Precision Final: {results["precision"] * 100}')
    print(f'Recall Final: {results["recall"] * 100}')


if __name__ == "__main__":
    show_results()
