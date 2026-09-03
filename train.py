import argparse
import os
import json
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.callbacks import BaseCallback

import custom_environment.helpers as H
from custom_environment.StationPlacementEnv import StationPlacement

"""
Train the model by reinforcement learning (DQN) with configurable hyperparameters.
"""

class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving the best model based on training reward and tracking history.
    """

    def __init__(self, check_freq: int, my_log_dir: str, my_modelname: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = my_log_dir
        self.modelname = my_modelname
        self.save_path = os.path.join(self.log_dir, self.modelname)
        self.best_mean_reward = -np.inf
        self.history = []

    def _init_callback(self) -> None:
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            try:
                x, y = ts2xy(load_results(self.log_dir), 'timesteps')
                if len(x) > 0:
                    my_mean_reward = np.mean(y[-10:])
                    self.history.append((self.num_timesteps, float(my_mean_reward)))
                    if self.verbose > 0 and self.n_calls % (self.check_freq * 100) == 0:
                        print("Num timesteps: {}".format(self.num_timesteps))
                        print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(
                            self.best_mean_reward, my_mean_reward))

                    if my_mean_reward > self.best_mean_reward:
                        self.best_mean_reward = my_mean_reward
                        if self.verbose > 0:
                            print(f"New best mean reward: {self.best_mean_reward:.2f}")
                            new_name = f"{self.modelname}{self.num_timesteps}.zip"
                            save_file = os.path.join(self.log_dir, new_name)
                            print(f"Saving new best model to {save_file}")
                            self.model.save(save_file)
            except Exception:
                pass
        return True

    def save_history(self, log_dir):
        history_file = os.path.join(log_dir, "reward_history.json")
        with open(history_file, "w") as f:
            json.dump(self.history, f, indent=2)
        print(f"Saved reward history to {history_file}")

    def plot_history(self, log_dir):
        if not self.history:
            return
        try:
            timesteps, rewards = zip(*self.history)
            plt.figure(figsize=(10, 5))
            plt.plot(timesteps, rewards, label="Mean Reward (last 10 eps)")
            plt.xlabel("Timesteps")
            plt.ylabel("Reward")
            plt.title("Training Reward Curve")
            plt.grid(True)
            plt.legend()
            plot_path = os.path.join(log_dir, "training_curve.png")
            plt.savefig(plot_path)
            print(f"Saved training plot to {plot_path}")
        except Exception as e:
            print(f"Could not generate training plot: {e}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train StationPlacement DQN Agent")
    parser.add_argument("--location", type=str, default="DongDa", help="Location/District name (default: DongDa)")
    parser.add_argument("--total_timesteps", type=int, default=200000, help="Total timesteps to train (default: 200000)")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for DQN (default: 128)")
    parser.add_argument("--buffer_size", type=int, default=20000, help="Replay buffer size (default: 20000)")
    parser.add_argument("--learning_rate", type=float, default=8e-5, help="Learning rate (default: 8e-5)")
    parser.add_argument("--exploration_fraction", type=float, default=0.3, help="Exploration fraction (default: 0.3)")
    parser.add_argument("--exploration_initial_eps", type=float, default=0.95, help="Initial exploration epsilon (default: 0.95)")
    parser.add_argument("--exploration_final_eps", type=float, default=0.05, help="Final exploration epsilon (default: 0.05)")
    parser.add_argument("--target_update_interval", type=int, default=500, help="Target update interval (default: 500)")
    parser.add_argument("--max_grad_norm", type=float, default=10.0, help="Max gradient norm (default: 10.0)")
    parser.add_argument("--net_arch", type=int, nargs="+", default=[256, 256], help="Hidden layers architecture (default: [256, 256])")
    parser.add_argument("--seed", type=int, default=1, help="Random seed (default: 1)")
    parser.add_argument("--ns", type=str, default="pcrl", help="Namespace / run identifier subfolder (default: pcrl)")
    parser.add_argument("--check_freq", type=int, default=1, help="Callback check frequency (default: 1)")
    parser.add_argument("--grid_penalty_weight", type=float, default=H.GRID_PENALTY_WEIGHT, help="Weight on grid penalty in score")
    return parser.parse_args()


def main():
    args = parse_args()

    # Set seed for reproducibility
    os.environ['PYTHONHASHSEED'] = '0'
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    np.random.seed(args.seed)
    random.seed(args.seed)

    location = args.location
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "custom_environment", "data")
    graph_file = os.path.join(base_dir, "Graph", location, f"{location}.graphml")
    
    # Check node file
    node_file = os.path.join(base_dir, "Graph", location, f"nodes_extended_{location}.txt")
    if not os.path.exists(node_file):
        alt_node_file = os.path.join(base_dir, "Graph", location, f"og_nodes_extended_{location}.txt")
        if os.path.exists(alt_node_file):
            node_file = alt_node_file

    # Check plan file
    plan_file = os.path.join(base_dir, "Graph", location, f"new_existingplan_{location}.pkl")
    if not os.path.exists(plan_file):
        alt_plan_file = os.path.join(base_dir, "Graph", location, f"existingplan_{location}.pkl")
        if os.path.exists(alt_plan_file):
            plan_file = alt_plan_file

    # Set globals
    H.GRID_PENALTY_WEIGHT = args.grid_penalty_weight

    print("\n--- Training Configuration ---")
    print(f"Location:           {location}")
    print(f"Graph file:         {graph_file}")
    print(f"Node file:          {node_file}")
    print(f"Plan file:          {plan_file}")
    print(f"Total timesteps:    {args.total_timesteps}")
    print(f"Learning rate:      {args.learning_rate}")
    print(f"Batch size:         {args.batch_size}")
    print(f"Buffer size:        {args.buffer_size}")
    print(f"Network arch:       {args.net_arch}")
    print(f"Grid penalty weight:{H.GRID_PENALTY_WEIGHT}")

    env = StationPlacement(graph_file, node_file, plan_file, location=location)

    if args.ns:
        log_dir = os.path.join("Results", "tmp", location, args.ns)
        modelname = f"best_model_{location}_{args.ns}_"
    else:
        log_dir = os.path.join("Results", "tmp", location)
        modelname = f"best_model_{location}_"

    os.makedirs(log_dir, exist_ok=True)

    # Save training configuration
    config_path = os.path.join(log_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"Saved config to {config_path}")

    env = Monitor(env, os.path.join(log_dir, "monitor.csv"))
    policy_kwargs = dict(net_arch=args.net_arch)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    model = DQN(
        "MlpPolicy",
        env,
        verbose=1,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        learning_rate=args.learning_rate,
        exploration_initial_eps=args.exploration_initial_eps,
        exploration_final_eps=args.exploration_final_eps,
        exploration_fraction=args.exploration_fraction,
        target_update_interval=args.target_update_interval,
        max_grad_norm=args.max_grad_norm,
        policy_kwargs=policy_kwargs,
        device=device,
        seed=args.seed
    )

    callback = SaveOnBestTrainingRewardCallback(check_freq=args.check_freq, my_log_dir=log_dir, my_modelname=modelname)
    model.learn(total_timesteps=args.total_timesteps, log_interval=10 ** 4, callback=callback)

    # Save training history & curve
    callback.save_history(log_dir)
    callback.plot_history(log_dir)
    print(f"\nTraining completed! Results saved to {log_dir}")


if __name__ == '__main__':
    main()