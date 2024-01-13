import hydra
import numpy as np
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split

from preprocessors import ImageTransformation


@hydra.main(version_base=None, config_path="../../config", config_name="main")
def process_data(config: DictConfig):
    input_dir = config.raw.path
    categories = config.raw.types
    sampling_size = config.model.sampling_size
    evaluation_number = config.model.evaluations_number
    test_size_percentage = config.model.test_size_percentage
    random_state = config.model.random_state

    # Se obtiene la clase para el pre-procesamiento de los datos
    image_normalizer = ImageTransformation(input_dir, categories)

    # Se remueve el fondo de las imagenes
    # image_normalizer.remove_background(config.processed.dir)

    # Se realiza la transformacion de las imagenes a arreglos de NumPy y se guardan
    data_sin_plagas, labels_sin_plagas, data_plagas, labels_plagas = image_normalizer.image_normalize()

    x_train_final = []
    x_test_final = []
    y_train_final = []
    y_test_final = []
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
            random_state=random_state
        )

        x_train_final.append(x_train)
        x_test_final.append(x_test)
        y_train_final.append(y_train)
        y_test_final.append(y_test)
    np.save(config.processed.x_train.path, x_train_final)
    np.save(config.processed.x_test.path, x_test_final)
    np.save(config.processed.y_train.path, y_train_final)
    np.save(config.processed.y_test.path, y_test_final)


if __name__ == "__main__":
    process_data()