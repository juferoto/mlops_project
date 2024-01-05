import hydra
from results import execute_model


""" Metodo que ejecuta todo un modelo de ML """
@hydra.main(version_base=None, config_path="../../config", config_name="main")
def main(config):
    # Se agregan todos los metodos necesarios para trabajar un modelo ML
    execute_model(config)

if __name__ == "__main__":
    main()
