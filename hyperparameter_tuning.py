"""
Hyperparameter Tuning Script for CSLP using Reinforcement Learning

This script automates the process of testing different hyperparameter combinations
and logs the results for later analysis and visualization.

Usage:
    python hyperparameter_tuning.py [--num_episodes NUM] [--location LOCATION]
    
Example:
    python hyperparameter_tuning.py --num_episodes 100 --location DongDa_partial
"""

import os
import sys
import json
import subprocess
import argparse
from datetime import datetime
from pathlib import Path
from itertools import product
import numpy as np
import pandas as pd

# Define hyperparameter search space
HYPERPARAMETER_SPACE = {
    'lr': [0.00001, 0.0001, 0.001],
    'gnn_hidden_dims': [64, 128, 256],
    'scaling_factor': [0.3, 0.47, 0.7],
    'distance_decay_factor': [0.7, 0.89, 1.0],
    'r_search': [0.1, 0.2, 0.5],
}

# Training configuration
DEFAULT_LOCATION = "DongDa_partial"
DEFAULT_NUM_EPISODES = 100
BASE_RESULTS_DIR = "Results/hyperparameter_tuning"


def create_results_directory():
    """Create the base results directory structure."""
    os.makedirs(BASE_RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = os.path.join(BASE_RESULTS_DIR, f"experiment_{timestamp}")
    os.makedirs(experiment_dir, exist_ok=True)
    return experiment_dir


def generate_hyperparameter_combinations():
    """Generate all combinations of hyperparameters using grid search."""
    keys = list(HYPERPARAMETER_SPACE.keys())
    values = list(HYPERPARAMETER_SPACE.values())
    combinations = []
    
    for combo in product(*values):
        param_dict = dict(zip(keys, combo))
        combinations.append(param_dict)
    
    return combinations


def run_training(params, location, num_episodes, experiment_dir, run_id):
    """Run training with specified hyperparameters."""
    results_subdir = os.path.join(experiment_dir, f"run_{run_id:03d}")
    os.makedirs(results_subdir, exist_ok=True)
    
    # Build command
    cmd = [
        sys.executable,
        "train.py",
        f"--location={location}",
        f"--num_episodes={num_episodes}",
        f"--results_dir={results_subdir}",
        f"--lr={params['lr']}",
        f"--gnn_hidden_dims={params['gnn_hidden_dims']}",
        f"--scaling_factor={params['scaling_factor']}",
        f"--distance_decay_factor={params['distance_decay_factor']}",
        f"--r_search={params['r_search']}",
    ]
    
    print(f"\n{'='*80}")
    print(f"Run {run_id}: Starting training with hyperparameters:")
    for key, value in params.items():
        print(f"  {key}: {value}")
    print(f"{'='*80}\n")
    
    try:
        # Run the training script
        result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
        
        # Prepare result data
        result_data = {
            'run_id': run_id,
            'timestamp': datetime.now().isoformat(),
            'hyperparameters': params,
            'location': location,
            'num_episodes': num_episodes,
            'results_dir': results_subdir,
            'exit_code': result.returncode,
        }
        
        # Check if training logs exist
        log_file = os.path.join(results_subdir, "training_results.json")
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                result_data['training_results'] = json.load(f)
        
        return result_data
    
    except Exception as e:
        print(f"Error during training: {e}")
        return {
            'run_id': run_id,
            'timestamp': datetime.now().isoformat(),
            'hyperparameters': params,
            'error': str(e),
            'exit_code': -1,
        }


def save_results(all_results, experiment_dir):
    """Save all results to a summary file."""
    # Convert to DataFrame for easier analysis
    results_data = []
    for result in all_results:
        row = {
            'run_id': result['run_id'],
            'lr': result['hyperparameters']['lr'],
            'gnn_hidden_dims': result['hyperparameters']['gnn_hidden_dims'],
            'scaling_factor': result['hyperparameters']['scaling_factor'],
            'distance_decay_factor': result['hyperparameters']['distance_decay_factor'],
            'r_search': result['hyperparameters']['r_search'],
            'exit_code': result.get('exit_code', -1),
        }
        
        # Add training results if available
        if 'training_results' in result:
            training = result['training_results']
            row['final_reward'] = training.get('final_reward', None)
            row['best_reward'] = training.get('best_reward', None)
            row['avg_reward'] = training.get('avg_reward', None)
        
        results_data.append(row)
    
    # Save as CSV
    df = pd.DataFrame(results_data)
    csv_path = os.path.join(experiment_dir, "results_summary.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")
    
    # Save as JSON for detailed information
    json_path = os.path.join(experiment_dir, "results_detailed.json")
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"Detailed results saved to: {json_path}")
    
    # Save hyperparameter space
    space_path = os.path.join(experiment_dir, "hyperparameter_space.json")
    with open(space_path, 'w') as f:
        json.dump(HYPERPARAMETER_SPACE, f, indent=2)
    print(f"Hyperparameter space saved to: {space_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Hyperparameter Tuning for CSLP using RL"
    )
    parser.add_argument(
        '--num_episodes',
        type=int,
        default=DEFAULT_NUM_EPISODES,
        help=f'Number of episodes for training (default: {DEFAULT_NUM_EPISODES})'
    )
    parser.add_argument(
        '--location',
        type=str,
        default=DEFAULT_LOCATION,
        help=f'Location for training (default: {DEFAULT_LOCATION})'
    )
    parser.add_argument(
        '--dry_run',
        action='store_true',
        help='Print hyperparameter combinations without running training'
    )
    
    args = parser.parse_args()
    
    # Generate combinations
    combinations = generate_hyperparameter_combinations()
    print(f"\nTotal combinations to test: {len(combinations)}")
    
    if args.dry_run:
        print("\nDry run: Hyperparameter combinations to be tested:")
        for i, combo in enumerate(combinations, 1):
            print(f"\n{i}. {combo}")
        return
    
    # Create results directory
    experiment_dir = create_results_directory()
    print(f"\nExperiment results directory: {experiment_dir}")
    
    # Run training for each combination
    all_results = []
    for run_id, params in enumerate(combinations, 1):
        result = run_training(
            params,
            args.location,
            args.num_episodes,
            experiment_dir,
            run_id
        )
        all_results.append(result)
    
    # Save results
    save_results(all_results, experiment_dir)
    
    print(f"\n{'='*80}")
    print("Hyperparameter tuning complete!")
    print(f"Results saved to: {experiment_dir}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
