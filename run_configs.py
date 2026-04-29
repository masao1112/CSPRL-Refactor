"""
Run train.py sequentially with multiple configurations.

Each config is a dict of CLI arguments for train.py.
Runs are isolated via subprocess so GPU memory is properly freed between runs.

Usage:
    python run_configs.py                  # run all configs
    python run_configs.py --only 0 2       # run only config #0 and #2
    python run_configs.py --start_from 3   # skip configs 0-2, start from #3
"""

import subprocess
import sys
import time
import os

# ──────────────────────────────────────────────────────────────────────
# Define your experiment configs here.
# Each dict maps CLI argument names (without "--") to their values.
# Omit any key to use the default from train.py.
# ──────────────────────────────────────────────────────────────────────
CONFIGS = [
    # Config 0: DongDa batch_size=64, lr=1e-4, net_arch=[256, 256]
    {
        "location": "DongDa",
        "learning_rate": 1e-4,
        "features_dim": 256,
        "batch_size": 64,
        "net_arch": [256, 256],
    },
    # Config 1: DongDa batch_size=64, lr=1e-4, net_arch=[512, 512]
    {
        "location": "DongDa",
        "learning_rate": 1e-4,
        "features_dim": 256,
        "batch_size": 64,
        "net_arch": [512, 512],
    },
    # Config 2: DongDa batch_size=64, lr=4e-5, net_arch=[256, 256]
    {
        "location": "DongDa",
        "learning_rate": 4e-5,
        "features_dim": 256,
        "batch_size": 64,
        "net_arch": [256, 256],
    },
    # Config 3: DongDa batch_size=64, lr=4e-5, net_arch=[512, 512]
    {
        "location": "DongDa",
        "learning_rate": 4e-5,
        "features_dim": 256,
        "batch_size": 64,
        "net_arch": [512, 512],
    },
    # Config 4: DongDa batch_size=128, lr=1e-4, net_arch=[256, 256]
    {
        "location": "DongDa",
        "learning_rate": 1e-4,
        "features_dim": 256,
        "batch_size": 128,
        "net_arch": [256, 256],
    },
    # Config 5: DongDa batch_size=128, lr=1e-4, net_arch=[512, 512]
    {
        "location": "DongDa",
        "learning_rate": 1e-4,
        "features_dim": 256,
        "batch_size": 128,
        "net_arch": [512, 512],
    },
    # Config 6: DongDa batch_size=128, lr=4e-5, net_arch=[256, 256]
    {
        "location": "DongDa",
        "learning_rate": 4e-5,
        "features_dim": 256,
        "batch_size": 128,
        "net_arch": [256, 256],
    },
    # Config 7: DongDa batch_size=128, lr=4e-5, net_arch=[512, 512]
    {
        "location": "DongDa",
        "learning_rate": 4e-5,
        "features_dim": 256,
        "batch_size": 128,
        "net_arch": [512, 512],
    },
    # Config 8: DongDa batch_size=256, lr=1e-4, net_arch=[256, 256]
    {
        "location": "DongDa",
        "learning_rate": 1e-4,
        "features_dim": 256,
        "batch_size": 256,
        "net_arch": [256, 256],
    },
    # Config 9: DongDa batch_size=256, lr=1e-4, net_arch=[512, 512]
    {
        "location": "DongDa",
        "learning_rate": 1e-4,
        "features_dim": 256,
        "batch_size": 256,
        "net_arch": [512, 512],
    },
    # Config 10: DongDa batch_size=256, lr=4e-5, net_arch=[256, 256]
    {
        "location": "DongDa",
        "learning_rate": 4e-5,
        "features_dim": 256,
        "batch_size": 256,
        "net_arch": [256, 256],
    },
    # Config 11: DongDa batch_size=256, lr=4e-5, net_arch=[512, 512]
    {
        "location": "DongDa",
        "learning_rate": 4e-5,
        "features_dim": 256,
        "batch_size": 256,
        "net_arch": [512, 512],
    }
]


def config_to_args(config: dict) -> list[str]:
    """Convert a config dict to a list of CLI arguments for train.py."""
    args = []
    for key, value in config.items():
        flag = f"--{key}"
        if isinstance(value, bool):
            if value:
                args.append(flag)
        elif isinstance(value, list):
            args.append(flag)
            args.extend(str(v) for v in value)
        else:
            args.append(flag)
            args.append(str(value))
    return args


def run_config(index: int, config: dict) -> bool:
    """Run a single training config. Returns True if successful."""
    label = config.get("location", f"config_{index}")
    cli_args = config_to_args(config)
    cmd = [sys.executable, "train.py"] + cli_args

    print("\n" + "=" * 70)
    print(f"  CONFIG {index}/{len(CONFIGS) - 1}:  {label}")
    print(f"  Command: {' '.join(cmd)}")
    print("=" * 70 + "\n")

    start = time.time()
    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
    elapsed = time.time() - start

    minutes, seconds = divmod(int(elapsed), 60)
    hours, minutes = divmod(minutes, 60)

    if result.returncode == 0:
        print(f"\n✓ Config {index} ({label}) completed in {hours}h {minutes}m {seconds}s")
        return True
    else:
        print(f"\n✗ Config {index} ({label}) FAILED (exit code {result.returncode}) after {hours}h {minutes}m {seconds}s")
        return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run train.py with multiple configs sequentially.")
    parser.add_argument("--only", type=int, nargs="+", help="Run only these config indices (0-based)")
    parser.add_argument("--start_from", type=int, default=0, help="Skip configs before this index")
    cli = parser.parse_args()

    # Determine which configs to run
    if cli.only is not None:
        indices = [i for i in cli.only if 0 <= i < len(CONFIGS)]
    else:
        indices = list(range(cli.start_from, len(CONFIGS)))

    print(f"Will run {len(indices)} config(s): {indices}")
    total_start = time.time()
    results = {}

    for idx in indices:
        config = CONFIGS[idx].copy()
        # Default namespace to config index to prevent overwriting
        if "ns" not in config:
            config["ns"] = f"config_{idx}"
            
        success = run_config(idx, config)
        results[idx] = success

    # Summary
    total_elapsed = time.time() - total_start
    m, s = divmod(int(total_elapsed), 60)
    h, m = divmod(m, 60)

    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    for idx, success in results.items():
        label = CONFIGS[idx].get("location", f"config_{idx}")
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  Config {idx:>2} ({label:>12}):  {status}")
    print(f"\nTotal time: {h}h {m}m {s}s")
    print("=" * 70)
