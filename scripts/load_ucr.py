import os
import sys

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig

from simn.train.dataset import make_ucr_dataloaders, resolve_ucr_dataset_names


@hydra.main(version_base=None, config_path="../configs/data", config_name="ucr128")
def main(cfg: DictConfig) -> None:
    root = cfg.path
    if not os.path.isabs(root):
        root = os.path.join(get_original_cwd(), root)
    root = os.path.abspath(root)

    if not os.path.isdir(root):
        print(
            f"[load_ucr] {root} not found; "
            "place the UCRArchive_2018 dataset directory under data/"
        )
        sys.exit(1)

    datasets = cfg.datasets
    names = resolve_ucr_dataset_names(root, datasets)
    print(f"[load_dataset] root={root}")
    print(f"[load_dataset] requested={list(datasets)} -> resolved={len(names)} datasets")

    if not names:
        print("[load_ucr] no matching datasets found")
        sys.exit(1)

    tasks = make_ucr_dataloaders(
        root=root,
        datasets=datasets,
        batch_size=cfg.get("batch_size", 32),
        seed=cfg.get("seed", 0),
        normalize=cfg.get("normalize", True),
    )

    for task in tasks:
        x0, y0 = next(iter(task.train_loader))
        m = task.meta
        print(
            f"{m['name']:24s} T={m['T']:<5d} C={m['num_classes']:<4d} "
            f"train={m['train_n']:<6d} test={m['test_n']:<6d} "
            f"batch x={tuple(x0.shape)}"
        )


if __name__ == "__main__":
    main()