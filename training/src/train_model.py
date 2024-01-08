import csv
import os
import hydra
import joblib
import numpy as np
from omegaconf import DictConfig
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from preprocessors import ImageTransformation

""" Aqui se agregan todas las funciones necesarias para entrenar un modelo """
@hydra.main(version_base=None, config_path="../../config", config_name="main")
def train(config: DictConfig):

    input_dir = config.raw.path
    categories = config.raw.types
    evaluation_number = config.model.evaluations_number
    sampling_size = config.model.sampling_size
    test_size_percentage = config.model.test_size_percentage

    confusion_matrixs = []

    # Se obtiene el modelo a usar
    model = LogisticRegression()

    # Se obtiene la clase para el pre-procesamiento de los datos (pueden ser funciones)
    image_normalizer = ImageTransformation(input_dir, categories)

    # Se realiza la normalizacion de las imagenes
    data_sin_plagas, labels_sin_plagas, data_plagas, labels_plagas = image_normalizer.image_normalize()
    data_total_size = len(data_sin_plagas) + len(data_plagas)


    # Repetir el proceso de entrenamiento y evaluación n veces (10 por defecto)
    for _ in range(evaluation_number):

        # Realizar subsampling en la clase mayoritaria
        indices_resample = np.random.choice(len(data_plagas), size=sampling_size, replace=True)
        
        data_plagas_resample = np.asarray(data_plagas)[indices_resample]
        labels_plagas_resample = np.asarray(labels_plagas)[indices_resample]

        # Combinar datos de 'sin_plaga' y datos subsampling de 'plaga'
        data_resample = np.concatenate([data_sin_plagas, data_plagas_resample])
        labels_resample = np.concatenate([labels_sin_plagas, labels_plagas_resample])

        # Dividir los datos en conjuntos de entrenamiento y prueba
        x_train, x_test, y_train, y_test = train_test_split(
            data_resample,
            labels_resample,
            test_size=test_size_percentage,
            shuffle=True,
            random_state=42
        )

        # Entrena el modelo
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)

        # Calcular métricas adicionales
        confusion_matrixs.append(confusion_matrix(y_test, y_pred))
    params = {
        'tamano_conjunto': data_total_size,
        'porcentaje_pruebas': test_size_percentage
    }
    
    # Se guarda el modelo
    joblib.dump(model, config.model.path)
    
    return confusion_matrixs, model, params

if __name__ == "__main__":
    train()