"""Profile SIMN vs LSTM forward+backward speed across devices.

Usage:
  python scripts/profile_speed.py [--device cuda|mps|cpu] [--T 1000] [--batch 64]
      [--hidden 16] [--layers 2] [--rank 8]

Prints per-iteration wall time and, if available, a kernel-count breakdown via
torch.profiler. This is the measurement harness behind SIMN speed tuning.
"""

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from simn.models.simn.rnn import SIMNRNN
from simn.train.common import set_seed
from test.baselines import LSTMNet


def default_device():
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_models(feat, hidden, layers, out, rank, device, torch_compile=False):
    simn = SIMNRNN(feat, hidden, layers, out, rank=rank, alpha=0.05,
                   bias=True, torch_compile=torch_compile).to(device)
    lstm = LSTMNet(feat, hidden, layers, out).to(device)
    return simn, lstm


def time_fwd_bwd(model, x, y, warmup, iters, device):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    crit = torch.nn.CrossEntropyLoss()
    for _ in range(warmup):
        opt.zero_grad(set_to_none=True)
        loss = crit(model(x), y)
        loss.backward()
        opt.step()
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        opt.zero_grad(set_to_none=True)
        loss = crit(model(x), y)
        loss.backward()
        opt.step()
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def profile(model, x, y, device):
    try:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
            loss = torch.nn.CrossEntropyLoss()(model(x), y)
            loss.backward()
        if device == "cuda":
            torch.cuda.synchronize()
        total = 0
        kinds = {}
        for evt in prof.key_averages():
            if not evt.self_device_time_total and not evt.self_cpu_time_total:
                continue
            name = evt.key.split("[")[0].split("  ")[-1].strip()
            kinds[name] = kinds.get(name, 0) + 1
            total += 1
        top = sorted(kinds.items(), key=lambda kv: -kv[1])[:15]
        return total, top
    except Exception as e:  # profiler unsupported on some backends
        return None, [("profiler-error", str(e))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=str, default=default_device())
    ap.add_argument("--feat", type=int, default=1)
    ap.add_argument("--hidden", type=int, default=16)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--out", type=int, default=10)
    ap.add_argument("--T", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--compile", action="store_true",
                    help="enable torch.compile on SIMN (CUDA recommended)")
    args = ap.parse_args()

    set_seed(0)
    device = args.device
    if device not in ("cuda", "mps", "cpu"):
        raise SystemExit(f"invalid device: {device}")

    x = torch.randn(args.batch, args.T, args.feat, device=device)
    y = torch.randint(0, args.out, (args.batch,), device=device)

    simn, lstm = build_models(args.feat, args.hidden, args.layers, args.out, args.rank,
                              device, torch_compile=args.compile)
    n_param_simn = sum(p.numel() for p in simn.parameters())
    n_param_lstm = sum(p.numel() for p in lstm.parameters())

    t_simn = time_fwd_bwd(simn, x, y, args.warmup, args.iters, device)
    t_lstm = time_fwd_bwd(lstm, x, y, args.warmup, args.iters, device)

    tot_s, top_s = profile(simn, x, y, device)
    tot_l, top_l = profile(lstm, x, y, device)

    print(f"device={device} T={args.T} batch={args.batch} hidden={args.hidden}"
          + (f" torch_compile on" if args.compile else ""))
    print(f"params: simn={n_param_simn} lstm={n_param_lstm}")
    print(f"simn {t_simn:8.2f} ms/iter  kernels={tot_s}")
    print(f"lstm {t_lstm:8.2f} ms/iter  kernels={tot_l}")
    print(f"ratio simn/lstm = {t_simn / t_lstm:6.2f}x")
    print("\n[simn top kernels]")
    for name, c in top_s or []:
        print(f"  {c:6d}  {name}")
    print("\n[lstm top kernels]")
    for name, c in top_l or []:
        print(f"  {c:6d}  {name}")


if __name__ == "__main__":
    main()