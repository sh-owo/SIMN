import torch
from torch import nn

from simn.train.common import cosine_scheduler
from simn.train.trainer.base import regression_metrics, train_loop


def train(
    model: nn.Module,
    train_loader,
    val_loader,
    test_loader,
    epochs: int,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    grad_clip: float | None = 1.0,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler=None,
    device: str = "cuda",
) -> dict:
    if optimizer is None:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
    if scheduler is None:
        scheduler = cosine_scheduler(
            optimizer, epochs, lr_peak=lr, lr_end=0.1 * lr, warmup_epochs=2
        )
    return train_loop(
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        epochs,
        criterion=nn.MSELoss(),
        metric=regression_metrics,
        grad_clip=grad_clip,
        device=device,
        track_best="state",
        lower_is_better=True,
        test_loader=test_loader,
    )