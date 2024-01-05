import os
import cv2
import joblib
import numpy as np
from hydra import compose, initialize
from hydra.utils import to_absolute_path as abspath
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

""" Aqui se agregan todas las funciones necesarias para realizar pruebas sobre un modelo ML """
def test_logistic_regression():

    with initialize(version_base=None, config_path="../../config"):
        config = compose(config_name="main")

    model_path = abspath(config.model.path)
    model = joblib.load(model_path)
    input_dir = config.raw.path
    categories = config.raw.types
    evaluations_number = config.model.evaluations_number
    sampling_size = config.model.sampling_size
    test_size = config.model.test_size
    
    accuracy_list, precision_list, recall_list = verify_model(input_dir, categories, model, evaluations_number, sampling_size, test_size)

    # Calcular el promedio de las métricas
    average_accuracy = np.mean(accuracy_list)
    average_precision = np.mean(precision_list)
    average_recall = np.mean(recall_list)

    # Realizar aserciones sobre los resultados promedio
    assert average_accuracy >= 0.7 and average_precision >= 0.7 and average_recall >= 0.7

def image_normalize(input_dir, categories):
        data_sin_plagas = []
        data_plagas = []
        labels_sin_plagas = []
        labels_plagas = []

        for category_idx, category in enumerate(categories):
            for file in os.listdir(os.path.join(input_dir, category)):
                img_path = os.path.join(input_dir, category, file)

                img = cv2.imread(img_path)
                image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                scaled_image = cv2.resize(src=image_rgb, dsize=(640, 640))
                if category == "sin_plaga":
                    data_sin_plagas.append(scaled_image.flatten())
                    labels_sin_plagas.append(category_idx)
                else:
                    data_plagas.append(scaled_image.flatten())
                    labels_plagas.append(category_idx)

        data_sin_plagas = np.asarray(data_sin_plagas)
        labels_sin_plagas = np.asarray(labels_sin_plagas)
        data_plagas = np.asarray(data_plagas)
        labels_plagas = np.asarray(labels_plagas)

        return data_sin_plagas, labels_sin_plagas, data_plagas, labels_plagas

def verify_model(input_dir, categories, model, evaluations_number, sampling_size, test_size):

    data_sin_plagas, labels_sin_plagas, data_plagas, labels_plagas = image_normalize(input_dir, categories)

    # Listas para almacenar métricas de cada iteración
    accuracy_list, precision_list, recall_list = [], [], []

    # Repetir el proceso de entrenamiento y evaluación n veces
    for _ in range(evaluations_number):
        # Realizar subsampling en la clase mayoritaria
        indices_resample = np.random.choice(len(data_plagas), size=sampling_size, replace=True)

        data_plagas_resample = np.asarray(data_plagas)[indices_resample]
        labels_plagas_resample = np.asarray(labels_plagas)[indices_resample]

        # Combinar datos de 'sin_plaga' y datos resampleados de 'plaga'
        data_resample = np.concatenate([data_sin_plagas, data_plagas_resample])
        labels_resample = np.concatenate([labels_sin_plagas, labels_plagas_resample])

        # Dividir los datos en conjuntos de entrenamiento y prueba
        x_train, x_test, y_train, y_test = train_test_split(
            data_resample,
            labels_resample,
            test_size=test_size,
            shuffle=True,
            random_state=42
        )

        # Realizar predicciones con el modelo cargado
        y_test_pred = model.predict(x_test)

        # Calcular métricas de rendimiento para esta iteración
        accuracy = accuracy_score(y_test, y_test_pred)
        precision = precision_score(y_test, y_test_pred)
        recall = recall_score(y_test, y_test_pred)

        # Almacenar métricas en listas
        accuracy_list.append(accuracy)
        precision_list.append(precision)
        recall_list.append(recall)

    return accuracy_list, precision_list, recall_list