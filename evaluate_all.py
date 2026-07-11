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
import matplotlib.pyplot as plt
import seaborn as sns
from stable_baselines3 import DQN

# Add project root to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

import custom_environment.helpers as H
from custom_environment.StationPlacementEnv import StationPlacement

def detect_settings(path_dir: str, args) -> tuple:
    """
    Detect location, use_gnn, and obs_type from config.json, directory path, or filenames.
    """
    location = args.location
    use_gnn = args.use_gnn
    obs_type = None
    # 1. Check config.json in the directory
    config_path = os.path.join(path_dir, "config.json")
    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
            if not location:
                location = config.get("location")
            if use_gnn is None:
                use_gnn = config.get("use_gnn")
            # Prefer explicit obs_type from config if available
            obs_type = config.get("obs_type")
            print(f"[DETECT] Loaded config.json: location={location}, use_gnn={use_gnn}, obs_type={obs_type}")
        except Exception as e:
            print(f"[DETECT] Warning reading config.json: {e}")
            
    # 2. Check directory path parts
    if not location or (use_gnn is None and obs_type is None):
        path_parts = os.path.normpath(path_dir).split(os.sep)
        known_locations = ["DongDa", "BaDinh", "CauGiay", "TayHo", "NamTuLiem", "HaiBaTrung", "HoanKiem", "ThanhXuan"]
        for part in path_parts:
            if part in known_locations:
                if not location:
                    location = part
            if obs_type is None:
                if part.lower() == "gnn":
                    use_gnn = True if use_gnn is None else use_gnn
                elif part.lower() == "mlp":
                    use_gnn = False if use_gnn is None else use_gnn
                elif part.lower() == "mlp_graph":
                    obs_type = "mlp_graph"
                    use_gnn = False
                
    # 3. Check filenames
    if not location or (use_gnn is None and obs_type is None):
        zip_files = glob.glob(os.path.join(path_dir, "*.zip"))
        for z in zip_files:
            base = os.path.basename(z)
            if base.startswith("best_model_"):
                parts = base.split("_")
                if len(parts) >= 4:
                    model_type = parts[2].lower()
                    if obs_type is None:
                        if model_type == "gnn":
                            use_gnn = True
                        elif model_type == "mlp":
                            use_gnn = False
                        # Handle mlp_graph: filename would be best_model_mlp_graph_...
                        # but split on _ gives ["best", "model", "mlp", "graph", ...]
                        if len(parts) >= 5 and parts[2] == "mlp" and parts[3] == "graph":
                            obs_type = "mlp_graph"
                            use_gnn = False
                    if not location:
                        # For mlp_graph, location is parts[4]; for others it's parts[3]
                        loc_idx = 4 if (obs_type == "mlp_graph") else 3
                        if len(parts) > loc_idx:
                            location = parts[loc_idx]
                    break
                    
    # Defaults
    if not location:
        location = "DongDa"
        print(f"[DETECT] Location not detected. Defaulting to: {location}")
    if use_gnn is None:
        use_gnn = False
        print(f"[DETECT] use_gnn not detected. Defaulting to: False (MLP)")
    
    # Resolve obs_type from use_gnn if not explicitly set
    if obs_type is None:
        obs_type = "gnn" if use_gnn else "mlp"
    return location, use_gnn, obs_type

def parse_step(filename: str) -> int:
    """
    Extract the training step from filename if present (e.g. best_model_mlp_DongDa_config_3_103354.zip -> 103354).
    """
    base = os.path.basename(filename)
    match = re.search(r'_(\d+)\.zip$', base)
    if match:
        return int(match.group(1))
    return 0

def evaluate_single_model(model_path: str, env: StationPlacement, use_gnn: bool, episodes: int = 1, seed: int = 1) -> Dict[str, float]:
    """
    Load a model and evaluate it over the specified number of episodes.
    """
    # Determine algo class
    if "ppo" in model_path.lower():
        from stable_baselines3 import PPO
        algo_class = PPO
    else:
        algo_class = DQN

    # Load model
    if use_gnn:
        from custom_environment.gnn_extractor import GNNFeaturesExtractor
        custom_objects = {"GNNFeaturesExtractor": GNNFeaturesExtractor}
        model = algo_class.load(model_path, env=env, custom_objects=custom_objects)
    else:
        model = algo_class.load(model_path, env=env)
        
    all_scores = []
    all_rewards = []
    all_stations = []
    all_budgets = []
    
    # Suppress internal prints of env steps/render to keep output clean
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
            score = env.best_score
            
            # Budget calculation
            basic_cost = sum(s[2]["fee"] for s in env.plan_instance.extend_existing_plan)
            total_inst_cost = (sum(station[2]["fee"] for station in best_plan) - basic_cost) / H.BUDGET
            
            all_scores.append(score)
            all_rewards.append(total_reward)
            all_stations.append(len(best_plan))
            all_budgets.append(total_inst_cost)
            
    return {
        "score": float(np.mean(all_scores)),
        "total_reward": float(np.mean(all_rewards)),
        "num_stations": float(np.mean(all_stations)),
        "used_budget_ratio": float(np.mean(all_budgets))
    }

def print_table(results: List[Dict[str, Any]]):
    """
    Print results as a formatted ASCII table.
    """
    headers = ["Model Filename", "Step", "Score (norm)", "Total Reward", "Stations", "Budget Used (%)"]
    col_widths = [45, 10, 15, 15, 10, 15]
    
    # Header format
    header_str = " | ".join(f"{h:<{w}}" for h, w in zip(headers, col_widths))
    print("-" * len(header_str))
    print(header_str)
    print("-" * len(header_str))
    
    # Rows
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
    Create a beautiful plot showing Score and Reward vs Step.
    """
    # Filter out models with step = 0 (or keep them at step 0)
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
    parser.add_argument("--path_dir", type=str, help="Directory containing the saved .zip models")
    parser.add_argument("--metric", type=str, choices=["score", "reward"], default="score", help="Metric to rank models (default: score)")
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes per model evaluation (default: 1)")
    parser.add_argument("--seed", type=int, default=1, help="Random seed for environment reset (default: 1)")
    parser.add_argument("--location", type=str, default=None, help="Override detected location")
    parser.add_argument("--use_gnn", type=bool, default=None, help="Override detected GNN settings")
    
    args = parser.parse_args()
    
    path_dir = args.path_dir
    if not os.path.isdir(path_dir):
        print(f"Error: {path_dir} is not a valid directory.")
        sys.exit(1)
        
    print(f"\nEvaluating models in directory: {path_dir}")
    location, use_gnn, obs_type = detect_settings(path_dir, args)
    
    # Setup files
    base_data_dir = os.path.join(current_dir, "custom_environment", "data")
    graph_file = os.path.join(base_data_dir, "Graph", location, f"{location}.graphml")
    node_file = os.path.join(base_data_dir, "Graph", location, f"nodes_extended_{location}.txt")
    plan_file = os.path.join(base_data_dir, "Graph", location, f"existingplan_{location}.pkl")
    
    print(f"[ENV] Initializing StationPlacement environment:")
    print(f"  Location: {location}")
    print(f"  Observation type: {obs_type}")
    print(f"  Graph file: {graph_file}")
    print(f"  Node file: {node_file}")
    print(f"  Plan file: {plan_file}")
    
    env = StationPlacement(graph_file, node_file, plan_file, location=location, obs_type=obs_type)
    
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
        if step >= 60000:
            print(f"[{idx+1}/{len(zip_files)}] Evaluating {filename} (Step: {step})...")

            try:
                metrics = evaluate_single_model(z, env, use_gnn, episodes=args.episodes, seed=args.seed)
                metrics["file"] = z
                metrics["step"] = step
                results.append(metrics)
            except Exception as e:
                print(f"  [ERROR] Failed to evaluate {filename}: {e}")
            
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
    print(f"\n🏆 BEST PERFORMING MODEL ({args.metric.upper()}):")
    print(f"  File: {best_model['file']}")
    print(f"  Step: {best_model['step']}")
    print(f"  Score: {best_model['score']:.6f}")
    print(f"  Total Reward: {best_model['total_reward']:.3f}")
    print(f"  Number of Stations: {int(best_model['num_stations'])}")
    print(f"  Budget Used: {best_model['used_budget_ratio']*100:.2f}%")
    
    # Save CSV report
    csv_path = os.path.join(path_dir, "evaluation_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model_file", "step", "score", "total_reward", "num_stations", "used_budget_ratio"])
        for r in results:
            writer.writerow([
                os.path.basename(r["file"]),
                r["step"],
                r["score"],
                r["total_reward"],
                r["num_stations"],
                r["used_budget_ratio"]
            ])
    print(f"\n[REPORT] Saved evaluation results to {csv_path}")
    
    # Plot results
    plot_path = os.path.join(path_dir, "evaluation_plot.png")
    try:
        plot_results(results, plot_path)
    except Exception as e:
        print(f"[PLOT] Warning: Could not generate performance plot: {e}")

if __name__ == "__main__":
    main()
