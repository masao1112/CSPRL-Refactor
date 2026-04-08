"""
Analyze Hyperparameter Tuning Results

This script automates the evaluation, analysis, and visualization 
of different hyperparameter combinations tested during the reinforcement learning runs.
"""

import os
import sys
import glob
import json
import re
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from stable_baselines3 import DQN

# Ensure custom classes are available to SB3 during load
try:
    import custom_environment.gnn_extractor
    _GNN_AVAILABLE = True
except ImportError:
    _GNN_AVAILABLE = False

from custom_environment.StationPlacementEnv import StationPlacement

def get_latest_experiment_dir(base_dir="Results/hyperparameter_tuning"):
    """Find the most recently modified experiment directory."""
    if not os.path.exists(base_dir):
        raise FileNotFoundError(f"Base directory {base_dir} does not exist.")
    
    subdirs = [os.path.join(base_dir, d) for d in os.listdir(base_dir) 
               if os.path.isdir(os.path.join(base_dir, d)) and 'experiment_' in d]
    
    if not subdirs:
        raise FileNotFoundError(f"No experiment directories found in {base_dir}")
        
    latest_dir = max(subdirs, key=os.path.getmtime)
    return latest_dir

def find_best_model_zip(run_dir):
    """
    Finds the model zip with the highest timestep.
    Assumes naming convection like best_model_gnn_DongDa_partial_1600.zip
    """
    search_pattern = os.path.join(run_dir, "*.zip")
    zip_files = glob.glob(search_pattern)
    
    if not zip_files:
        return None
        
    highest_timestep = -1
    best_file = None
    
    for f in zip_files:
        basename = os.path.basename(f)
        # Extract digits right before .zip
        match = re.search(r'_(\d+)\.zip$', basename)
        if match:
            timestep = int(match.group(1))
            if timestep > highest_timestep:
                highest_timestep = timestep
                best_file = f
        else:
            # If no timestep is strictly defined, we can check if it just says 'best_model.zip'
            if 'best_model' in basename and best_file is None:
                best_file = f
                
    return best_file

def evaluate_model(model_path, config):
    """
    Reinstantiates the StationPlacement environment according to training_config 
    and evaluates the model deterministically.
    """
    location = config.get("location", "DongDa_partial")
    use_gnn = config.get("use_gnn", True)

    # Resolve paths for the environment dataset
    base_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "custom_environment", "data")
    graph_file = os.path.join(base_data_dir, "Graph", location, location + ".graphml")
    node_file = os.path.join(base_data_dir, "Graph", location, "nodes_extended_" + location + ".txt")
    plan_file = os.path.join(base_data_dir, "Graph", location, "existingplan_" + location + ".pkl")
    
    # Check if plan_file physically exists (as handled in train.py)
    if not os.path.exists(plan_file):
        plan_file = None

    obs_type = "gnn" if use_gnn else "mlp"
    
    # Instantiate the environment using identical methodology from train.py
    env = StationPlacement(
        my_graph_file=graph_file, 
        my_node_file=node_file, 
        my_plan_file=plan_file, 
        location=location, 
        obs_type=obs_type
    )

    try:
        # Load the saved model to RAM. 
        # Using device="cpu" usually avoids VRAM overallocation for purely inference tasks.
        model = DQN.load(model_path, custom_objects={}, device="cpu")
        
        # Evaluate for a single deterministic episode
        obs, info = env.reset(seed=42)
        done = False
        total_reward = 0.0
        
        while not done:
            # Deterministic=True enforces policy argmax (best reliable output)
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

        # Extract environment's deterministic best score attained this episode.
        # Since unwrapped handles Gym nuances:
        final_score = env.unwrapped.best_score
        
        return final_score, total_reward

    except Exception as e:
        print(f"Error evaluating {model_path}: {e}")
        return None, None

def create_visualizations(df, output_dir):
    """Generate informative visualizations from an engineering perspective."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Filter only numerical columns for correlation matrix
    numeric_df = df.select_dtypes(include=[np.number])
    
    # Exclude run_id from analysis plots
    if 'run_id' in numeric_df.columns:
        numeric_df = numeric_df.drop(columns=['run_id'])
    
    # Visual 1: Correlation Heatmap
    if len(numeric_df.columns) > 1 and len(numeric_df) > 1:
        plt.figure(figsize=(10, 8))
        corr = numeric_df.corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", square=True,
                    cbar_kws={'label': 'Pearson Correlation'})
        plt.title('Feature Correlation with Network Performance (Eval Score)', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'), dpi=300)
        plt.close()

    # Visual 2: Pairplot focusing on key hyperparameters vs. eval_score
    hyper_cols = [c for c in df.columns if c in ['learning_rate', 'gnn_hidden_dims', 'scaling_factor', 'r_search', 'distance_decay_factor']]
    if hyper_cols and 'eval_score' in df.columns and len(df) > 1:
        # We ensure standard categorical interpretation by converting some columns to category for visual hue if discrete
        # Instead of generic pairplot, create specific scatter plots comparing hyperparams with eval_score
        fig, axes = plt.subplots(1, len(hyper_cols), figsize=(5 * len(hyper_cols), 5))
        if len(hyper_cols) == 1:
            axes = [axes]
        
        for ax, col in zip(axes, hyper_cols):
            sns.scatterplot(data=df, x=col, y='eval_score', hue='eval_score', size='eval_reward', 
                            sizes=(50, 200), palette='viridis', alpha=0.8, ax=ax, legend=False)
            ax.set_title(f'Score vs {col}')
            ax.grid(True, linestyle='--', alpha=0.6)
            
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'hyperparameters_scatter.png'), dpi=300)
        plt.close()

def main():
    parser = argparse.ArgumentParser(description="Analyze Hyperparameter Tuning Results")
    parser.add_argument('--experiment_dir', type=str, default=None, 
                        help="Path to the specific experiment directory to analyze. Defaults to newest.")
    args = parser.parse_args()

    print(f"\n{'='*80}")
    print("Initiating Evaluation Analysis Pipeline")
    print(f"{'='*80}\n")

    # 1. Look for the directory
    try:
        if args.experiment_dir:
            experiment_dir = args.experiment_dir
            if not os.path.exists(experiment_dir):
                raise FileNotFoundError(f"Provided directory not found: {experiment_dir}")
        else:
            experiment_dir = get_latest_experiment_dir()
            
        print(f">> Analyzing Experiment Directory: {experiment_dir}\n")
    except Exception as e:
        print(f"Error locating experiment directories: {e}")
        return

    run_dirs = sorted(glob.glob(os.path.join(experiment_dir, "run_*")))
    
    results = []

    # 2. Iterate through runs
    for count, run_dir in enumerate(run_dirs, 1):
        config_path = os.path.join(run_dir, "training_config.json")
        if not os.path.exists(config_path):
            continue
            
        with open(config_path, 'r') as f:
            config = json.load(f)
            
        # Find Model
        best_model_path = find_best_model_zip(run_dir)
        if not best_model_path:
            continue
            
        run_name = os.path.basename(run_dir)
        print(f"[{count}/{len(run_dirs)}] Evaluating {run_name} ...", end="", flush=True)
        
        eval_score, eval_reward = evaluate_model(best_model_path, config)
        
        if eval_score is not None:
            # Flatten dict for generic dataframe consumption
            row = {
                "run_id": run_name,
                "learning_rate": config.get("learning_rate"),
                "gnn_hidden_dims": config.get("gnn_hidden_dims"),
                "scaling_factor": config.get("scaling_factor"),
                "distance_decay_factor": config.get("distance_decay_factor"),
                "r_search": config.get("r_search"),
                "eval_score": eval_score,
                "eval_reward": eval_reward,
                "model_path": best_model_path
            }
            results.append(row)
            print(f" Score: {eval_score:.3f} | Reward: {eval_reward:.3f}")
        else:
            print(" [FAILED]")

    if not results:
        print("\nNo viable models found or evaluations all failed. Exiting.")
        return

    # 3. Aggregate into Pandas Dataframe
    df = pd.DataFrame(results)
    csv_output = os.path.join(experiment_dir, "evaluated_hyperparameter_results.csv")
    df.to_csv(csv_output, index=False)
    print(f"\n>> Saved comprehensive analytical results to: {csv_output}")

    # 4. Generate Visualizations
    vis_dir = os.path.join(experiment_dir, "visualizations")
    print(f">> Generating engineering visualizations in: {vis_dir}")
    create_visualizations(df, vis_dir)

    # 5. Extract Best Performer and Print Final Console Summary
    best_config = df.loc[df['eval_score'].idxmax()]
    
    print(f"\n{'='*80}")
    print("HYPERPARAMETER ANALYSIS REPORT")
    print(f"{'='*80}")
    print("Optimization Criteria Driven By Maximum Deterministic Evaluation Score.\n")
    print(f"Best Performing Run: {best_config['run_id']}")
    print("-" * 35)
    print(f"  Highest Score (Best) : {best_config['eval_score']:.4f}")
    print(f"  Total Episode Reward : {best_config['eval_reward']:.4f}")
    print("-" * 35)
    print("Configured Hyperparameters:")
    print(f"  Learning Rate          : {best_config['learning_rate']}")
    print(f"  GNN Hidden Dimensions  : {best_config['gnn_hidden_dims']}")
    print(f"  Scaling Factor         : {best_config['scaling_factor']}")
    print(f"  Distance Decay Factor  : {best_config['distance_decay_factor']}")
    print(f"  Search Radius (r)      : {best_config['r_search']}")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()
