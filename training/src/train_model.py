import hydra
import joblib
import numpy as np
from hydra.utils import to_absolute_path as abspath
from omegaconf import DictConfig
from sklearn.linear_model import LogisticRegression

""" Aqui se agregan todas las funciones necesarias para entrenar un modelo """
def load_data(path: DictConfig):
    x_train = np.load(abspath(path.x_train.path))
    x_test = np.load(abspath(path.x_test.path))
    y_train = np.load(abspath(path.y_train.path))
    y_test = np.load(abspath(path.y_test.path))
    return x_train, x_test, y_train, y_test


@hydra.main(version_base=None, config_path="../../config", config_name="main")
def train(config: DictConfig):

    # Se obtiene el modelo a usar
    model = LogisticRegression()

    # Se realiza la normalizacion de las imagenes
    x_train, x_test, y_train, y_test = load_data(config.processed)

    x_train = x_train.reshape(-1, x_train.shape[2])
    x_test = x_test.reshape(-1, x_test.shape[2]) 
    y_train = y_train.reshape(y_train.shape[0], -1) 
    y_test = y_test.reshape(y_test.shape[0], -1) 

    model.fit(x_train, y_train.ravel())
    
    # Se guarda el modelo
    joblib.dump(model, config.model.path)

if __name__ == "__main__":
    train()