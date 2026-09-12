import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEST_DIR = ROOT / "test"

WEATHER_CSV = ROOT / "data" / "weather" / "jena_climate_2009_2016.csv"
UCR_ROOT = ROOT / "data" / "UCRArchive_2018"

TESTS = [
    ("test_nextstep.py", False),
    ("test_simn_vs_lstm.py", True),
]


def default_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def check_data() -> None:
    missing = []
    if not WEATHER_CSV.exists():
        missing.append(f"{WEATHER_CSV}  (run: python scripts/download_weather.py)")
    if not UCR_ROOT.is_dir():
        missing.append(f"{UCR_ROOT}  (run: python scripts/load_ucr.py)")
    if missing:
        print("[run_tests] missing data:")
        for m in missing:
            print(f"  - {m}")
        sys.exit(1)
    n_ucr = len([d for d in os.listdir(UCR_ROOT) if os.path.isdir(UCR_ROOT / d)])
    print(f"[run_tests] data ok: UCR({n_ucr} datasets), weather csv present")


def build_cmd(test, args, takes_datasets) -> list[str]:
    cmd = [sys.executable, str(TEST_DIR / test)]
    cmd += ["--out", str(ROOT / "runs" / test[:-3])]
    if args.epochs is not None:
        cmd += ["--epochs", str(args.epochs)]
    if args.batch is not None:
        cmd += ["--batch", str(args.batch)]
    if args.lr is not None:
        cmd += ["--lr", str(args.lr)]
    if not takes_datasets and args.lookback is not None:
        cmd += ["--lookback", str(args.lookback)]
    cmd += ["--device", args.device]
    if args.seeds:
        cmd += ["--seeds"] + [str(s) for s in args.seeds]
    if takes_datasets and args.datasets:
        cmd += ["--datasets"] + args.datasets
    return cmd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["all"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--lookback", type=int, default=None)
    parser.add_argument("--device", type=str, default=default_device())
    parser.add_argument("--skip", nargs="+", default=[])
    args = parser.parse_args()

    check_data()

    results = {}
    for test, takes_datasets in TESTS:
        base = test[:-3]
        if test in args.skip or base in args.skip:
            print(f"\n[skip] {test}", flush=True)
            results[test] = "skipped"
            continue
        cmd = build_cmd(test, args, takes_datasets)
        print(f"\n{'=' * 72}\n  RUN {test}\n  {' '.join(cmd)}\n{'=' * 72}", flush=True)
        rc = subprocess.run(cmd).returncode
        results[test] = "OK" if rc == 0 else f"FAIL(rc={rc})"
        print(f"\n[runner] {test} -> {results[test]}", flush=True)

    print(f"\n{'=' * 72}\n  SUMMARY\n{'=' * 72}")
    for test, status in results.items():
        print(f"  {test:26s} {status}")
    failed = [t for t, s in results.items() if s.startswith("FAIL")]
    print("\nALL DONE" if not failed else f"\nFAILED: {failed}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()