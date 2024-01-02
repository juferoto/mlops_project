import hydra
from results import show_results

@hydra.main(version_base=None, config_path="../../config", config_name="main")
def main(config):
    show_results(config)

if __name__ == "__main__":
    main()
