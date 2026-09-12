import hydra
from omegaconf import DictConfig

from simn.train.trainer.trainer import run_training


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    run_training(cfg)


if __name__ == "__main__":
    main()
