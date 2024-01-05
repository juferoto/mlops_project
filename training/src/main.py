import hydra
from evaluate_model import evaluate
from train_model import train


""" Metodo que ejecuta todo un modelo de ML """
@hydra.main(version_base=None, config_path="../../config", config_name="main")
def main(config):
    # Se agregan todos los metodos necesarios para trabajar un modelo de ML
    evaluate(config)

if __name__ == "__main__":
    main()
