from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# import mlflow
from helper import BaseLogger

logger = BaseLogger()

def log_params(model: object, features: dict):
    logger.log_params({"model_class": type(model).__name__})
    model_params = model.get_params()

    for arg, value in model_params.items():
        logger.log_params({arg: value})

    logger.log_params({"features": features})


def log_metrics(**metrics: dict):
    logger.log_metrics(metrics)

def run_training(pipeline, num_evaluaciones, tamano_sampling, tamano_pruebas):
    matrices_confusion = []

    data_sin_plagas, labels_sin_plagas, data_plagas, labels_plagas = pipeline.named_steps['normalize'].image_normalize()
    data_total_size = len(data_sin_plagas) + len(data_plagas)

    # Repetir el proceso de entrenamiento y evaluación n veces (10 por defecto)
    for _ in range(num_evaluaciones):

        # Realizar subsampling en la clase mayoritaria
        indices_resample = np.random.choice(len(data_plagas), size=tamano_sampling, replace=True)
        
        data_plagas_resample = np.asarray(data_plagas)[indices_resample]
        labels_plagas_resample = np.asarray(labels_plagas)[indices_resample]

        # Combinar datos de 'sin_plaga' y datos resampleados de 'plaga'
        data_resample = np.concatenate([data_sin_plagas, data_plagas_resample])
        labels_resample = np.concatenate([labels_sin_plagas, labels_plagas_resample])

        # Dividir los datos en conjuntos de entrenamiento y prueba
        x_train, x_test, y_train, y_test = train_test_split(
            data_resample,
            labels_resample,
            test_size=tamano_pruebas,
            shuffle=True,
            random_state=42
        )

        x_train_size = len(x_train)
        x_test_size = len(x_train)

        # Entrenar el modelo y obtener las métricas
        pipeline.fit(x_train, y_train)
        y_pred = pipeline.predict(x_test)

        # Calcular métricas adicionales
        matrices_confusion.append(confusion_matrix(y_test, y_pred))
    params = {
        'dataset_size': data_total_size,
        'training_set_size': x_train_size,
        'test_set_size': x_test_size
    }
    # mlflow.log_params({
    #     'dataset_size': data_total_size,
    #     'training_set_size': x_train_size,
    #     'test_set_size': x_test_size
    # })

    # mlflow.sklearn.log_model(
    #     sk_model=pipeline,
    #     artifact_path="regressions-model",
    #     registered_model_name="RegressionModel"
    # )
    log_params(pipeline, params)
    return matrices_confusion

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

    # mlflow.log_metrics(metrics)
    log_metrics(metrics)
    return metrics