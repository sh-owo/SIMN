from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

WEATHER_FEATURES = [
    "p",
    "T",
    "Tpot",
    "Tdew",
    "rh",
    "VPmax",
    "VPact",
    "VPdef",
    "sh",
    "H2OC",
    "rho",
    "wv",
    "max.wv",
    "wd",
]

MISSING_VALUES = (-9999.0,)


def load_weather(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; download the Jena climate dataset first")
    df_orig = pd.read_csv(path)
    df = pd.DataFrame()
    for i, f in enumerate(WEATHER_FEATURES):
        col = df_orig.columns[i + 1]
        vals = df_orig[col].to_numpy(dtype=np.float64)
        for mv in MISSING_VALUES:
            vals = np.where(vals == mv, np.nan, vals)
        df[f] = pd.Series(vals).ffill().bfill().to_numpy(dtype=np.float64)
    return df


def make_windows(values: np.ndarray, lookback: int, stride: int) -> tuple[np.ndarray, np.ndarray]:
    """values: [N, D]. Returns (X: [M, L, D], y: [M, D]) with y[t] = x after window."""
    n = len(values)
    ends = list(range(lookback, n, stride))
    X = np.stack([values[e - lookback : e] for e in ends]).astype(np.float64)
    y = values[ends].astype(np.float64)
    return X, y


@dataclass
class WeatherTask:
    name: str
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    meta: dict


def make_weather_dataloaders(
    path: str | Path,
    lookback: int = 24,
    stride: int = 6,
    batch_size: int = 128,
    seed: int = 42,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
) -> WeatherTask:
    path = Path(path)
    df = load_weather(path)
    vals = df[WEATHER_FEATURES].to_numpy(np.float64)  # [N, D]

    n = len(vals)
    n_test = int(n * test_frac)
    n_val = int(n * val_frac)
    n_train = n - n_test - n_val

    off_val = n_train
    off_test = n_train + n_val

    x_tr, y_tr = make_windows(vals[:off_val], lookback, stride)
    x_va, y_va = make_windows(vals[off_val - lookback : off_test], lookback, stride)
    x_te, y_te = make_windows(vals[off_test - lookback :], lookback, stride)

    mu = x_tr.reshape(-1, x_tr.shape[-1]).mean(0, keepdims=True)
    sd = x_tr.reshape(-1, x_tr.shape[-1]).std(0, keepdims=True)
    sd = np.where(sd == 0, 1.0, sd)

    def norm(x):
        return (x - mu) / sd

    x_tr = norm(x_tr)
    y_tr = norm(y_tr)
    x_va = norm(x_va)
    y_va = norm(y_va)
    x_te = norm(x_te)
    y_te = norm(y_te)

    t_tr = torch.from_numpy(x_tr).float()
    y_tr_t = torch.from_numpy(y_tr).float()
    t_va = torch.from_numpy(x_va).float()
    y_va_t = torch.from_numpy(y_va).float()
    t_te = torch.from_numpy(x_te).float()
    y_te_t = torch.from_numpy(y_te).float()

    gen = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        TensorDataset(t_tr, y_tr_t), batch_size=batch_size, shuffle=True, generator=gen
    )
    val_loader = DataLoader(TensorDataset(t_va, y_va_t), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(t_te, y_te_t), batch_size=batch_size, shuffle=False)

    meta = {
        "name": "jena_climate",
        "features": WEATHER_FEATURES,
        "dim": x_tr.shape[-1],
        "lookback": lookback,
        "train_n": len(x_tr),
        "val_n": len(x_va),
        "test_n": len(x_te),
        "off_val": off_val,
        "off_test": off_test,
        "mu": mu.squeeze(0),
        "sd": sd.squeeze(0),
    }
    return WeatherTask("jena", train_loader, val_loader, test_loader, meta)