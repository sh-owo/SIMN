import os

import torch
from hydra.utils import get_original_cwd

from simn.models.simn.rnn import SIMNRNN
from simn.train.checkpoint import load_checkpoint, save_checkpoint
from simn.train.common import cosine_scheduler, set_seed
from simn.train.dataset import make_ucr_dataloaders
from simn.train.dataset.weather import make_weather_dataloaders
from simn.train.trainer.ucr_acc import train as train_ucr
from simn.train.trainer.weather import train as train_weather


def _resolve_path(cfg) -> str:
    root = cfg.data.path
    if not os.path.isabs(root):
        root = os.path.join(get_original_cwd(), root)
    return root


def _device(cfg) -> str:
    if cfg.get("device"):
        return cfg.device
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _checkpoint_path(cfg, name: str) -> str:
    ckpt_dir = cfg.train.get("checkpoint")
    if not ckpt_dir:
        return os.path.join(os.getcwd(), f"{name}_best.pt")
    if not os.path.isabs(ckpt_dir):
        ckpt_dir = os.path.join(get_original_cwd(), ckpt_dir)
    return os.path.join(ckpt_dir, f"{name}_best.pt")


def _resolve_compile(model_cfg) -> bool:
    setting = model_cfg.get("torch_compile", "auto")
    if setting is True or str(setting).lower() == "true":
        return True
    if str(setting).lower() == "auto":
        return torch.cuda.is_available()
    return False


def _build_ucr_model(task_meta: dict, model_cfg) -> SIMNRNN:
    return SIMNRNN(
        input_size=1,
        hidden_size=model_cfg.hidden_size,
        num_layers=model_cfg.num_layers,
        output_size=task_meta["num_classes"],
        rank=model_cfg.rank,
        alpha=model_cfg.alpha,
        bias=model_cfg.bias,
        chunk=model_cfg.get("chunk"),
        torch_compile=_resolve_compile(model_cfg),
    )


def _build_weather_model(task_meta: dict, model_cfg) -> SIMNRNN:
    return SIMNRNN(
        input_size=task_meta["dim"],
        hidden_size=model_cfg.hidden_size,
        num_layers=model_cfg.num_layers,
        output_size=task_meta["dim"],
        rank=model_cfg.rank,
        alpha=model_cfg.alpha,
        bias=model_cfg.bias,
        chunk=model_cfg.get("chunk"),
        torch_compile=_resolve_compile(model_cfg),
    )


def _run_ucr(cfg) -> dict:
    root = _resolve_path(cfg)
    tasks = make_ucr_dataloaders(
        root=root,
        datasets=cfg.data.datasets,
        batch_size=cfg.train.batch_size,
        seed=cfg.seed,
        normalize=cfg.data.get("normalize", True),
    )
    device = _device(cfg)
    print(f"[trainer] device={device} datasets={len(tasks)}")

    results: dict = {}
    for task in tasks:
        name = task.meta["name"]
        print(
            f"\n[trainer] === {name} "
            f"(T={task.meta['T']}, C={task.meta['num_classes']}, "
            f"train={task.meta['train_n']}, test={task.meta['test_n']}) ==="
        )

        model = _build_ucr_model(task.meta, cfg.model)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.train.lr_peak,
            weight_decay=cfg.train.weight_decay,
        )
        scheduler = cosine_scheduler(
            optimizer,
            epochs=cfg.train.epochs,
            lr_peak=cfg.train.lr_peak,
            lr_end=cfg.train.lr_end,
            warmup_epochs=cfg.train.get("warmup_epochs", 0),
        )

        resume_path = cfg.train.get("resume")
        if resume_path and os.path.exists(resume_path):
            state = load_checkpoint(resume_path, model, torch.device(device))
            print(
                f"[trainer] resumed from {resume_path} "
                f"(epoch={state.get('epoch')}, best_acc={state.get('best_acc')})"
            )

        out = train_ucr(
            model,
            task.train_loader,
            task.test_loader,
            epochs=cfg.train.epochs,
            lr=cfg.train.lr_peak,
            weight_decay=cfg.train.weight_decay,
            grad_clip=cfg.train.get("grad_clip"),
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
        )

        best = out["best"]
        results[name] = {
            "best_epoch": best["epoch"],
            "train_loss": best["train_loss"],
            "train_acc": best["train_acc"],
            "val_loss": best["val_loss"],
            "val_acc": best["val_acc"],
        }

        ckpt_path = _checkpoint_path(cfg, name)
        save_checkpoint(
            ckpt_path,
            model,
            optimizer,
            scheduler,
            best["epoch"],
            best["val_acc"],
        )
        print(f"[trainer] checkpoint saved: {ckpt_path}")

    print("\n[trainer] === summary ===")
    print(f"{'dataset':24s} {'best_epoch':>10s} {'val_acc':>8s}")
    for name, r in results.items():
        print(f"{name:24s} {r['best_epoch']:>10d} {r['val_acc']:>8.4f}")

    return results


def _run_weather(cfg) -> dict:
    csv_path = os.path.join(_resolve_path(cfg), cfg.data.name)
    task = make_weather_dataloaders(
        csv_path,
        lookback=cfg.data.get("lookback", 24),
        stride=cfg.data.get("stride", 6),
        batch_size=cfg.train.batch_size,
        seed=cfg.seed,
    )
    device = _device(cfg)
    name = task.meta["name"]
    print(
        f"[trainer] === {name} "
        f"(dim={task.meta['dim']}, lookback={task.meta['lookback']}, "
        f"train={task.meta['train_n']}, val={task.meta['val_n']}, "
        f"test={task.meta['test_n']}) ==="
    )

    model = _build_weather_model(task.meta, cfg.model)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.train.lr_peak,
        weight_decay=cfg.train.weight_decay,
    )
    scheduler = cosine_scheduler(
        optimizer,
        epochs=cfg.train.epochs,
        lr_peak=cfg.train.lr_peak,
        lr_end=cfg.train.lr_end,
        warmup_epochs=cfg.train.get("warmup_epochs", 0),
    )

    resume_path = cfg.train.get("resume")
    if resume_path and os.path.exists(resume_path):
        state = load_checkpoint(resume_path, model, torch.device(device))
        print(
            f"[trainer] resumed from {resume_path} "
            f"(epoch={state.get('epoch')}, best_rmse={state.get('best_acc')})"
        )

    out = train_weather(
        model,
        task.train_loader,
        task.val_loader,
        task.test_loader,
        epochs=cfg.train.epochs,
        lr=cfg.train.lr_peak,
        weight_decay=cfg.train.weight_decay,
        grad_clip=cfg.train.get("grad_clip"),
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
    )

    results = {
        "best_epoch": out["best_epoch"],
        "val_rmse": out["val"]["rmse"],
        "val_mae": out["val"]["mae"],
        "val_r": out["val"]["r"],
        "test_rmse": out["test"]["rmse"],
        "test_mae": out["test"]["mae"],
        "test_r": out["test"]["r"],
    }

    ckpt_path = _checkpoint_path(cfg, name)
    save_checkpoint(
        ckpt_path,
        model,
        optimizer,
        scheduler,
        out["best_epoch"],
        out["val"]["rmse"],
    )
    print(f"[trainer] checkpoint saved: {ckpt_path}")

    print("\n[trainer] === summary ===")
    print(
        f"{'dataset':24s} {'best_epoch':>10s} {'val_rmse':>9s} "
        f"{'test_rmse':>9s} {'test_mae':>9s} {'test_r':>7s}"
    )
    print(
        f"{name:24s} {results['best_epoch']:>10d} {results['val_rmse']:>9.4f} "
        f"{results['test_rmse']:>9.4f} {results['test_mae']:>9.4f} "
        f"{results['test_r']:>7.4f}"
    )

    return results


def run_training(cfg) -> dict:
    set_seed(cfg.seed)
    data_type = cfg.data.get("type")
    if data_type == "ucr":
        return _run_ucr(cfg)
    elif data_type == "weather":
        return _run_weather(cfg)
    else:
        raise ValueError(f"unsupported data type: {data_type!r}")