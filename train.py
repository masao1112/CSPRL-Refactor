from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.callbacks import BaseCallback
import os
import numpy as np
import torch
import random
import argparse
import json
import pandas as pd
from custom_environment.StationPlacementEnv import StationPlacement

"""
Trai the model by reinforcement learning.
"""

class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Code from Stable Baselines3,
    https://colab.research.google.com/github/Stable-Baselines-Team/rl-colab-notebooks/blob/sb3/monitor_training.ipynb
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param my_log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """

    def __init__(self, check_freq: int, my_log_dir: str, my_modelname: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = my_log_dir
        self.modelname = my_modelname
        self.save_path = os.path.join(self.log_dir, self.modelname)
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

            # Retrieve training reward
            x, y = ts2xy(load_results(self.log_dir), 'timesteps')
            if len(x) > 0 and len(y) > 0:
                # Mean training reward over the last 100 episodes
                my_mean_reward = np.mean(y[-10:])
                if self.verbose > 0:
                    print("Num timesteps: {}".format(self.num_timesteps))
                    print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(
                        self.best_mean_reward, my_mean_reward))

                if my_mean_reward > self.best_mean_reward:
                    self.best_mean_reward = my_mean_reward
                    # Example for saving best model
                    if self.verbose > 0:
                        print("New best mean reward: {:.2f}".format(self.best_mean_reward))
                        # we want to make sure that the best models are not overwritten
                        new_name = self.modelname + str(self.num_timesteps)
                        if self.save_path is not None:
                            os.makedirs(self.save_path, exist_ok=True)
                        self.save_path = os.path.join(self.log_dir, new_name)
                        print("Saving new best model to {}".format(self.save_path))
                    self.model.save(self.save_path)
                # else:
                #     if self.verbose > 0:
                #         new_name = self.modelname + str(self.num_timesteps)
                #         if self.save_path is not None:
                #             os.makedirs(self.save_path, exist_ok=True)
                #         self.save_path = os.path.join(self.log_dir, new_name)
                #         print("Saving model on frequency to {}".format(self.save_path))
                #     self.model.save(self.save_path)

        return True


if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train RL model for CSLP")
    parser.add_argument('--location', type=str, default="DongDa_partial", help='Location to train on')
    parser.add_argument('--use_gnn', type=bool, default=True, help='Use GNN policy')
    parser.add_argument('--num_episodes', type=int, default=40000, help='Number of timesteps to train')
    parser.add_argument('--results_dir', type=str, default=None, help='Directory to save results')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate for the model')
    parser.add_argument('--gnn_hidden_dims', type=int, default=128, help='Hidden dimensions for GNN')
    parser.add_argument('--scaling_factor', type=float, default=0.47, help='Scaling factor for dynamic demand')
    parser.add_argument('--distance_decay_factor', type=float, default=0.89, help='Distance decay factor')
    parser.add_argument('--r_search', type=float, default=0.2, help='Search radius for benefit')
    
    args = parser.parse_args()
    
    # Set reproducibility seed
    os.environ['PYTHONASHSEED'] = '0'
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    seed = 1
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    np.random.seed(seed)
    random.seed(seed)
    
    # Prepare environment paths
    location = args.location
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "custom_environment", "data")
    graph_file = os.path.join(base_dir, "Graph", location, location + ".graphml")
    node_file = os.path.join(base_dir, "Graph", location, "nodes_extended_" + location + ".txt")
    plan_file = os.path.join(base_dir, "Graph", location, "existingplan_" + location + ".pkl")
    
    # Check if plan file actually exists
    if not os.path.exists(plan_file):
        plan_file = None

    use_gnn = args.use_gnn
    obs_type = "gnn" if use_gnn else "mlp"
    
    # Create environment with hyperparameters
    env = StationPlacement(graph_file, node_file, plan_file, location=location, obs_type=obs_type)
    
    # Set results directory
    if args.results_dir:
        log_dir = args.results_dir
    else:
        log_dir = f"Results/tmp/gnn/{location}/"
    
    modelname = "best_model_" + location + "_"

    # Define and train the agent
    os.makedirs(log_dir, exist_ok=True)
    env = Monitor(env, os.path.join(log_dir, "monitor.csv"))
    
    if use_gnn:
        from custom_environment.gnn_extractor import GNNFeaturesExtractor
        policy_kwargs = dict(
            features_extractor_class=GNNFeaturesExtractor,
            features_extractor_kwargs=dict(features_dim=args.gnn_hidden_dims),
            net_arch=[args.gnn_hidden_dims, args.gnn_hidden_dims]
        )
        policy_type = "MultiInputPolicy"
        modelname = "best_model_gnn_" + location + "_"
    else:
        policy_kwargs = dict(net_arch=[256, 256])  # hidden layers
        policy_type = "MlpPolicy"
    
    # Create and train model with specified hyperparameters
    model = DQN(
        policy_type, env, verbose=1, batch_size=128, buffer_size=10000, 
        learning_rate=args.lr,
        exploration_initial_eps=1, exploration_final_eps=0.05, exploration_fraction=0.2, 
        policy_kwargs=policy_kwargs,
        device='cuda' if torch.cuda.is_available() else 'cpu', seed=seed
    )
    callback = SaveOnBestTrainingRewardCallback(check_freq=400, my_log_dir=log_dir, my_modelname=modelname)
    model.learn(total_timesteps=args.num_episodes, log_interval=10 ** 4, callback=callback)
    
    # Save training configuration and results summary
    config = {
        'location': location,
        'use_gnn': use_gnn,
        'num_episodes': args.num_episodes,
        'learning_rate': args.lr,
        'gnn_hidden_dims': args.gnn_hidden_dims,
        'scaling_factor': args.scaling_factor,
        'distance_decay_factor': args.distance_decay_factor,
        'r_search': args.r_search,
    }
    
    config_path = os.path.join(log_dir, "training_config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\nTraining configuration saved to: {config_path}")
    
    # Extract training results from monitor.csv and save to training_results.json
    monitor_path = os.path.join(log_dir, "monitor.csv")
    if os.path.exists(monitor_path):
        try:
            # Read the monitor CSV (skip the metadata line)
            df = pd.read_csv(monitor_path, skiprows=1)
            
            # Extract reward statistics
            rewards = df['r'].values
            training_results = {
                'final_reward': float(rewards[-1]) if len(rewards) > 0 else None,
                'best_reward': float(np.max(rewards)) if len(rewards) > 0 else None,
                'avg_reward': float(np.mean(rewards)) if len(rewards) > 0 else None,
                'min_reward': float(np.min(rewards)) if len(rewards) > 0 else None,
                'total_episodes': len(rewards),
            }
            
            # Save training results
            results_path = os.path.join(log_dir, "training_results.json")
            with open(results_path, 'w') as f:
                json.dump(training_results, f, indent=2)
            
            print(f"Training results saved to: {results_path}")
            print(f"\nTraining Summary:")
            print(f"  Final Reward: {training_results['final_reward']:.4f}")
            print(f"  Best Reward: {training_results['best_reward']:.4f}")
            print(f"  Average Reward: {training_results['avg_reward']:.4f}")
            print(f"  Min Reward: {training_results['min_reward']:.4f}")
            print(f"  Total Episodes: {training_results['total_episodes']}")
            
        except Exception as e:
            print(f"Error extracting training results: {e}")