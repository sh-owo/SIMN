import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset


def load_ucr_tsv(filepath: str) -> tuple[np.ndarray, np.ndarray]:
    arr = pd.read_csv(filepath, sep="\t", header=None).to_numpy(dtype=np.float64)
    return arr[:, 1:], arr[:, 0]


def resolve_ucr_dataset_names(root: str, datasets: list[str]) -> list[str]:
    root = str(root)
    if list(datasets) == ["all"]:
        names = []
        for folder in sorted(os.listdir(root)):
            if os.path.isdir(os.path.join(root, folder)) and os.path.isfile(
                os.path.join(root, folder, f"{folder}_TRAIN.tsv")
            ):
                names.append(folder)
        return names

    real = {}
    for folder in os.listdir(root):
        if os.path.isfile(os.path.join(root, folder, f"{folder}_TRAIN.tsv")):
            real[folder.lower()] = folder

    resolved = []
    for d in datasets:
        key = str(d).lower()
        if key in real:
            resolved.append(real[key])
        else:
            print(f"[ucr] warning: dataset '{d}' not found under {root}, skipping")
    return sorted(set(resolved))


def _znormalize(X: np.ndarray) -> np.ndarray:
    std = X.std(axis=1, keepdims=True)
    mean = X.mean(axis=1, keepdims=True)
    out = (X - mean) / std
    out[np.isnan(out)] = 0.0
    return out


def _remap_labels(y_train: np.ndarray, y_test: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    classes = sorted(set(np.unique(y_train)) | set(np.unique(y_test)))
    table = {v: i for i, v in enumerate(classes)}
    return (
        np.array([table[v] for v in y_train], dtype=np.int64),
        np.array([table[v] for v in y_test], dtype=np.int64),
        len(classes),
    )


@dataclass
class UCRTask:
    name: str
    train_loader: DataLoader
    test_loader: DataLoader
    meta: dict


def make_ucr_dataloaders(
    root: str,
    datasets: list[str],
    batch_size: int,
    seed: int = 0,
    normalize: bool = True,
) -> list[UCRTask]:
    names = resolve_ucr_dataset_names(root, datasets)
    tasks = []
    for name in names:
        train_path = f"{root}/{name}/{name}_TRAIN.tsv"
        test_path = f"{root}/{name}/{name}_TEST.tsv"

        x_train, y_train_raw = load_ucr_tsv(train_path)
        x_test, y_test_raw = load_ucr_tsv(test_path)
        if normalize:
            x_train = _znormalize(x_train)
            x_test = _znormalize(x_test)

        y_train, y_test, num_classes = _remap_labels(y_train_raw, y_test_raw)

        x_train_t = torch.from_numpy(x_train).float().unsqueeze(-1)
        x_test_t = torch.from_numpy(x_test).float().unsqueeze(-1)
        y_train_t = torch.from_numpy(y_train).long()
        y_test_t = torch.from_numpy(y_test).long()

        generator = torch.Generator().manual_seed(seed)
        train_loader = DataLoader(
            TensorDataset(x_train_t, y_train_t), batch_size=batch_size, shuffle=True, generator=generator
        )
        test_loader = DataLoader(
            TensorDataset(x_test_t, y_test_t), batch_size=batch_size, shuffle=False
        )

        meta = {
            "name": name,
            "num_classes": num_classes,
            "T": x_train.shape[1],
            "train_n": x_train.shape[0],
            "test_n": x_test.shape[0],
        }
        tasks.append(UCRTask(name=name, train_loader=train_loader, test_loader=test_loader, meta=meta))
    return tasks