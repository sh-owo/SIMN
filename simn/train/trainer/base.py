import math

import torch
from torch import nn

from simn.train.common import evaluate, train_one_epoch


def accuracy_metrics(preds: torch.Tensor, ys: torch.Tensor) -> dict:
    return {"acc": (preds.argmax(dim=1) == ys).float().mean().item()}


def regression_metrics(preds: torch.Tensor, ys: torch.Tensor) -> dict:
    r_per_feature = torch.stack(
        [
            torch.corrcoef(torch.stack([preds[:, i], ys[:, i]]))[0, 1]
            for i in range(ys.shape[1])
        ]
    )
    return {
        "rmse": math.sqrt(((preds - ys) ** 2).mean().item()),
        "mae": (preds - ys).abs().mean().item(),
        "r": r_per_feature.mean().item(),
        "r_per_feature": r_per_feature,
    }


def train_loop(
    model: nn.Module,
    train_loader,
    val_loader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epochs: int,
    criterion: nn.Module,
    metric,
    grad_clip: float | None = None,
    device: str = "cuda",
    track_best: str = "row",
    lower_is_better: bool = False,
    test_loader=None,
) -> dict:
    """Generic epoch-level training loop shared by task trainers.

    track_best:
      "row"   -> returns the best history record in `best`
      "state" -> snapshots best-epoch weights, restores them, and evaluates
                 on val/test (`val`/`test` in the result)
    """
    device = torch.device(device)
    model = model.to(device)

    history = []
    best_epoch = None
    best_metric = None
    best_state = None

    for epoch in range(1, epochs + 1):
        tr = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            grad_clip=grad_clip,
            collect_preds=True,
        )
        va = evaluate(model=model, loader=val_loader, criterion=criterion, device=device)

        train_metrics = metric(tr["preds"], tr["ys"])
        val_metrics = metric(va["preds"], va["ys"])

        if scheduler is not None:
            scheduler.step()

        record = {
            "epoch": epoch,
            "train_loss": tr["loss"],
            **{f"train_{k}": v for k, v in train_metrics.items() if not isinstance(v, torch.Tensor)},
            "val_loss": va["loss"],
            **{f"val_{k}": v for k, v in val_metrics.items() if not isinstance(v, torch.Tensor)},
        }
        history.append(record)

        parts = "  ".join(f"{k}={v:.4f}" for k, v in record.items())
        print(f"[{epoch:03d}/{epochs:03d}] {parts}", flush=True)

        key = "rmse" if "rmse" in val_metrics else "acc"
        cur = val_metrics[key]
        if best_metric is None:
            improved = True
        elif lower_is_better:
            improved = cur < best_metric
        else:
            improved = cur > best_metric

        if improved:
            best_metric = cur
            best_epoch = epoch
            if track_best == "state":
                best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in model.state_dict().items()
                }

    out = {
        "history": history,
        "best": None,
        "best_epoch": best_epoch,
        "val": None,
        "test": None,
    }

    if track_best == "state":
        if best_state is not None:
            model.load_state_dict(best_state)
        model = model.to(device)
        out["val"] = _eval_dict(evaluate(model, val_loader, criterion, device), metric)
        if test_loader is not None:
            out["test"] = _eval_dict(evaluate(model, test_loader, criterion, device), metric)
    elif best_epoch is not None:
        out["best"] = history[best_epoch - 1]

    return out


def _eval_dict(m: dict, metric) -> dict:
    mt = {k: v for k, v in metric(m["preds"], m["ys"]).items() if not isinstance(v, torch.Tensor)}
    return {"loss": m["loss"], **mt}