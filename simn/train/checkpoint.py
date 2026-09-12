import os

import torch


def save_checkpoint(path, model, optimizer, scheduler, epoch, best_acc):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_acc": best_acc,
        },
        path,
    )


def load_checkpoint(path, model, device):
    state = torch.load(path, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    return state
