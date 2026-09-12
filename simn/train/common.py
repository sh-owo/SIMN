import math
import random

import numpy as np
import torch
from torch import nn


def cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    epochs: int,
    lr_peak: float,
    lr_end: float,
    warmup_epochs: int = 0,
) -> torch.optim.lr_scheduler.LambdaLR:
    w = int(warmup_epochs)
    total = max(epochs - w, 1)

    def lr_factor(epoch: int) -> float:
        if w and epoch < w:
            return (epoch + 1) / w
        t = (epoch - w) / total
        c = lr_end / lr_peak
        return c + 0.5 * (1 - c) * (1 + math.cos(math.pi * t))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_factor)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    grad_clip: float | None = None,
    collect_preds: bool = False,
) -> dict:
    model.train()

    total_loss = 0.0
    total_samples = 0
    all_pred = []
    all_y = []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        model.reset_shunting_state()

        optimizer.zero_grad(set_to_none=True)

        pred = model(x)

        loss = criterion(pred, y)

        loss.backward()

        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        batch_size = x.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        if collect_preds:
            all_pred.append(pred.detach().cpu())
            all_y.append(y.detach().cpu())

    out = {"loss": total_loss / total_samples}
    if collect_preds:
        out["preds"] = torch.cat(all_pred)
        out["ys"] = torch.cat(all_y)
    return out


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
) -> dict:
    model.eval()

    total_loss = 0.0
    total_samples = 0
    all_pred = []
    all_y = []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        model.reset_shunting_state()

        pred = model(x)

        loss = criterion(pred, y)

        batch_size = x.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        all_pred.append(pred.cpu())
        all_y.append(y.cpu())

    return {
        "loss": total_loss / total_samples,
        "preds": torch.cat(all_pred),
        "ys": torch.cat(all_y),
    }
