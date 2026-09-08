import os
import sys
import glob
import json
import re
import csv
import argparse
import io
import contextlib
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from stable_baselines3 import DQN

# Add project root to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

import custom_environment.helpers as H
from custom_environment.StationPlacementEnv import StationPlacement
from run_metrics import travel_metric, waiting_metric


def detect_settings(path_dir: str, args) -> tuple:
    """
    Detect location from config.json, directory path, or filenames.
    """
    location = args.location

    # 1. Check config.json in the directory
    config_path = os.path.join(path_dir, "config.json")
    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
            if not location:
                location = config.get("location")
            print(f"[DETECT] Loaded config.json: location={location}")
        except Exception as e:
            print(f"[DETECT] Warning reading config.json: {e}")

    # 2. Check directory path parts
    if not location:
        path_parts = os.path.normpath(path_dir).split(os.sep)
        known_locations = ["DongDa", "BaDinh", "CauGiay", "TayHo", "NamTuLiem",
                          "HaiBaTrung", "HoanKiem", "ThanhXuan"]
        for part in path_parts:
            if part in known_locations:
                location = part
                break

    # 3. Check filenames
    if not location:
        zip_files = glob.glob(os.path.join(path_dir, "*.zip"))
        for z in zip_files:
            base = os.path.basename(z)
            if base.startswith("best_model_"):
                parts = base.split("_")
                if len(parts) >= 4:
                    # best_model_DongDa_pcrl_12345.zip -> location = DongDa
                    location = parts[2]
                    break

    if not location:
        location = "DongDa"
        print(f"[DETECT] Location not detected. Defaulting to: {location}")

    return location


def parse_step(filename: str) -> int:
    """
    Extract the training step from filename (e.g. best_model_DongDa_pcrl_103354.zip -> 103354).
    """
    base = os.path.basename(filename)
    match = re.search(r'_(\d+)\.zip$', base)
    if match:
        return int(match.group(1))
    return 0


def evaluate_single_model(model_path: str, env: StationPlacement, episodes: int = 1,
                          seed: int = 1) -> Dict[str, float]:
    """
    Load a model and evaluate it over the specified number of episodes.
    """
    model = DQN.load(model_path, env=env)

    per_episode: Dict[str, List[float]] = {k: [] for k in (
        "score", "total_reward", "num_stations", "used_budget_ratio",
        "benefit", "cost", "charg_time", "wait_time", "cost_travel",
        "travel_max", "wait_max", "dist_penalty", "cap_penalty", "overloaded_buses",
    )}

    # Suppress internal prints during evaluation
    with contextlib.redirect_stdout(io.StringIO()):
        for ep in range(episodes):
            obs, _ = env.reset(seed=seed + ep)
            done = False
            total_reward = 0

            while not done:
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = env.step(action)
                total_reward += reward
                if done or truncated:
                    break

            # Retrieve performance metrics
            best_node_list, best_plan = env.render()

            # Recompute the score from best_plan
            dist_penalty = cap_penalty = 0.0
            n_overloaded = 0
            grid_penalty = None
            if env.grid_adapter:
                station_nodes = [(s[0], s[2]["capability"]) for s in best_plan]
                dist_penalty, cap_penalty, _, _ = env.grid_adapter.calculate_grid_penalty(station_nodes)
                grid_penalty = {"dist_penalty": dist_penalty, "cap_penalty": cap_penalty}
                n_overloaded = len(env.grid_adapter.get_grid_violations(station_nodes))
            score, benefit, cost, charg_time, wait_time, cost_travel = H.norm_score(
                best_plan, best_node_list,
                env.plan_instance.norm_benefit, env.plan_instance.norm_charg,
                env.plan_instance.norm_wait, env.plan_instance.norm_travel,
                grid_penalty,
            )

            # Budget calculation
            basic_cost = getattr(env.plan_instance, "basic_cost", sum(station[2]["fee"] for station in best_plan))
            total_inst_cost = (sum(station[2]["fee"] for station in best_plan) - basic_cost) / H.BUDGET

            for key, value in (
                ("score", score), ("total_reward", total_reward),
                ("num_stations", len(best_plan)), ("used_budget_ratio", total_inst_cost),
                ("benefit", benefit), ("cost", cost),
                ("charg_time", charg_time), ("wait_time", wait_time),
                ("cost_travel", cost_travel),
                ("travel_max", travel_metric(best_node_list)),
                ("wait_max", waiting_metric(best_plan)),
                ("dist_penalty", dist_penalty), ("cap_penalty", cap_penalty),
                ("overloaded_buses", n_overloaded),
            ):
                per_episode[key].append(float(value))

    return {k: float(np.mean(v)) for k, v in per_episode.items()}


def print_table(results: List[Dict[str, Any]]):
    """
    Print results as a formatted ASCII table.
    """
    headers = ["Model Filename", "Step", "Score (norm)", "Total Reward", "Stations", "Budget Used (%)"]
    col_widths = [45, 10, 15, 15, 10, 15]

    header_str = " | ".join(f"{h:<{w}}" for h, w in zip(headers, col_widths))
    print("-" * len(header_str))
    print(header_str)
    print("-" * len(header_str))

    for r in results:
        name = os.path.basename(r["file"])
        step = r["step"]
        score = f"{r['score']:.6f}"
        reward = f"{r['total_reward']:.3f}"
        stations = f"{int(r['num_stations'])}"
        budget = f"{r['used_budget_ratio']*100:.2f}%"

        row_str = (
            f"{name:<45} | "
            f"{step:<10} | "
            f"{score:<15} | "
            f"{reward:<15} | "
            f"{stations:<10} | "
            f"{budget:<15}"
        )
        print(row_str)
    print("-" * len(header_str))


def plot_results(results: List[Dict[str, Any]], save_path: str):
    """
    Create a plot showing Score and Reward vs Step.
    """
    df = pd.DataFrame(results)
    df = df.sort_values(by="step")

    sns.set_theme(style="darkgrid")
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot Score
    color = "tab:blue"
    ax1.set_xlabel("Training Steps", fontweight="bold", labelpad=10)
    ax1.set_ylabel("Normalized Score", color=color, fontweight="bold")
    line1 = ax1.plot(df["step"], df["score"], color=color, marker="o", label="Eval Score")
    ax1.tick_params(axis="y", labelcolor=color)

    # Plot Reward
    ax2 = ax1.twinx()
    color = "tab:orange"
    ax2.set_ylabel("Total Reward", color=color, fontweight="bold")
    line2 = ax2.plot(df["step"], df["total_reward"], color=color, marker="x", linestyle="--", label="Total Reward")
    ax2.tick_params(axis="y", labelcolor=color)

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper left")

    plt.title("Model Performance vs. Training Steps", fontsize=14, fontweight="bold", pad=15)
    fig.tight_layout()

    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[PLOT] Performance plot saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate all models in a directory to find the best performing model.")
    parser.add_argument("--path_dir", type=str, required=True, help="Directory containing the saved .zip models")
    parser.add_argument("--metric", type=str, choices=["score", "reward"], default="score",
                        help="Metric to rank models (default: score)")
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes per model evaluation (default: 1)")
    parser.add_argument("--seed", type=int, default=1, help="Random seed for environment reset (default: 1)")
    parser.add_argument("--location", type=str, default=None, help="Override detected location")
    parser.add_argument("--min_step", type=int, default=60000,
                         help="Skip model checkpoints saved before this training step (default: 60000)")

    args = parser.parse_args()

    path_dir = args.path_dir
    if not os.path.isdir(path_dir):
        print(f"Error: {path_dir} is not a valid directory.")
        sys.exit(1)

    print(f"\nEvaluating models in directory: {path_dir}")
    location = detect_settings(path_dir, args)

    # Adopt the reward parameters this run was trained with
    run_cfg = {}
    run_cfg_path = os.path.join(path_dir, "config.json")
    if os.path.exists(run_cfg_path):
        try:
            with open(run_cfg_path, "r") as f:
                run_cfg = json.load(f)
        except Exception as e:
            print(f"[CONFIG] Warning reading {run_cfg_path}: {e}")
    if "grid_penalty_weight" in run_cfg:
        H.GRID_PENALTY_WEIGHT = float(run_cfg["grid_penalty_weight"])
    print(f"[CONFIG] grid_penalty_weight={H.GRID_PENALTY_WEIGHT}")

    # Setup files
    base_data_dir = os.path.join(current_dir, "custom_environment", "data")
    graph_file = os.path.join(base_data_dir, "Graph", location, f"{location}.graphml")

    node_file = os.path.join(base_data_dir, "Graph", location, f"nodes_extended_{location}.txt")
    if not os.path.exists(node_file):
        alt = os.path.join(base_data_dir, "Graph", location, f"og_nodes_extended_{location}.txt")
        if os.path.exists(alt):
            node_file = alt

    plan_file = os.path.join(base_data_dir, "Graph", location, f"new_existingplan_{location}.pkl")
    if not os.path.exists(plan_file):
        alt = os.path.join(base_data_dir, "Graph", location, f"existingplan_{location}.pkl")
        if os.path.exists(alt):
            plan_file = alt

    print(f"[ENV] Initializing StationPlacement environment:")
    print(f"  Location: {location}")
    print(f"  Graph file: {graph_file}")
    print(f"  Node file: {node_file}")
    print(f"  Plan file: {plan_file}")

    env = StationPlacement(graph_file, node_file, plan_file, location=location)

    # Find all zip files
    zip_files = glob.glob(os.path.join(path_dir, "*.zip"))
    if not zip_files:
        print(f"No .zip model files found in {path_dir}.")
        sys.exit(0)

    print(f"Found {len(zip_files)} model files to evaluate.")

    results = []
    for idx, z in enumerate(zip_files):
        filename = os.path.basename(z)
        step = parse_step(filename)
        if step >= args.min_step:
            print(f"[{idx+1}/{len(zip_files)}] Evaluating {filename} (Step: {step})...")

            try:
                metrics = evaluate_single_model(z, env, episodes=args.episodes, seed=args.seed)
                metrics["file"] = z
                metrics["step"] = step
                results.append(metrics)
            except Exception as e:
                print(f"  [ERROR] Failed to evaluate {filename}: {e}")
        else:
            print(f"[{idx+1}/{len(zip_files)}] Skipping {filename} (Step: {step} < min_step {args.min_step})")

    if not results:
        print("No models were successfully evaluated.")
        sys.exit(1)

    # Sort results
    if args.metric == "score":
        results = sorted(results, key=lambda x: x["score"], reverse=True)
    else:
        results = sorted(results, key=lambda x: x["total_reward"], reverse=True)

    # Print results summary
    print("\nEvaluation Results (Sorted by {}):".format(args.metric))
    print_table(results)

    # Identify the best model
    best_model = results[0]
    print(f"\n=== BEST PERFORMING MODEL ({args.metric.upper()}) ===")
    print(f"  File: {best_model['file']}")
    print(f"  Step: {best_model['step']}")
    print(f"  Score: {best_model['score']:.6f}")
    print(f"  Total Reward: {best_model['total_reward']:.3f}")
    print(f"  Number of Stations: {int(best_model['num_stations'])}")
    print(f"  Budget Used: {best_model['used_budget_ratio']*100:.2f}%")
    print(f"  Benefit: {best_model['benefit']:.4f}")
    print(f"  Social Cost: {best_model['cost']:.4f}")
    print(f"  Grid Distance Penalty: {best_model['dist_penalty']:.4f}")
    print(f"  Grid Capacity Penalty: {best_model['cap_penalty']:.4f}")
    print(f"  Overloaded Buses: {int(best_model['overloaded_buses'])}")

    # Save CSV report
    csv_path = os.path.join(path_dir, "evaluation_results.csv")
    metric_cols = ["score", "total_reward", "benefit", "cost",
                   "charg_time", "wait_time", "cost_travel", "travel_max", "wait_max",
                   "dist_penalty", "cap_penalty", "overloaded_buses",
                   "num_stations", "used_budget_ratio"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model_file", "step"] + metric_cols)
        for r in results:
            writer.writerow([os.path.basename(r["file"]), r["step"]]
                            + [r.get(c, "") for c in metric_cols])
    print(f"\n[REPORT] Saved evaluation results to {csv_path}")

    # Plot results
    plot_path = os.path.join(path_dir, "evaluation_plot.png")
    try:
        plot_results(results, plot_path)
    except Exception as e:
        print(f"[PLOT] Warning: Could not generate performance plot: {e}")


if __name__ == "__main__":
    main()
