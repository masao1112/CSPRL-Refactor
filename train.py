import argparse
import os
import json
import random
import csv
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import deque
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
    Callback for saving a model (the check is done every ``check_freq`` episodes)
    based on training reward and environment score.
    """

    def __init__(self, check_freq: int, my_log_dir: str, my_modelname: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = my_log_dir
        self.modelname = my_modelname
        self.save_path = os.path.join(self.log_dir, self.modelname)
        self.rewards = deque(maxlen=5)
        self.best_mean_reward = -np.inf
        self.best_env_score = -np.inf
        self.n_episodes = 0
        # Episode history for plotting
        self.episode_rewards = []
        self.episode_best_scores = []

    def _init_callback(self) -> None:
        if self.log_dir is not None:
            os.makedirs(self.log_dir, exist_ok=True)

    def _on_step(self) -> bool:
        # Check if an episode finished
        if self.locals["dones"][0]:
            self.n_episodes += 1

            # Access current learning rate from optimizer
            lr = self.model.policy.optimizer.param_groups[0]["lr"]

            # Query the environment for the best_score
            try:
                env_best_score = self.training_env.get_attr('last_episode_best_score')[0]
            except Exception:
                env_best_score = -np.inf

            # Get total episode reward from Monitor wrapper
            info = self.locals["infos"][0]
            episode_reward = info.get("episode", {}).get("r", 0.0) if info else 0.0

            # Record history
            self.episode_rewards.append(episode_reward)
            self.episode_best_scores.append(env_best_score)

            # Store reward for mean calculation
            self.rewards.append(episode_reward)

            if self.n_episodes % self.check_freq == 0:
                my_mean_reward = np.mean(self.rewards)

                if self.verbose > 0:
                    print("-" * 20)
                    print("Num timesteps: {}, Episode: {}".format(self.num_timesteps, self.n_episodes))
                    print("Current LR: {:.2e}".format(lr))
                    print("Reward  -> Current: {:.3f} | Mean: {:.3f} | Best Mean: {:.3f}".format(
                        episode_reward, my_mean_reward, self.best_mean_reward))
                    print("Score   -> Current: {:.6f} | Best: {:.6f}".format(
                        env_best_score, self.best_env_score))

                # Save by best mean reward
                new_best_mean = my_mean_reward > self.best_mean_reward
                if new_best_mean:
                    new_name = self.modelname + str(self.num_timesteps)
                    if self.log_dir is not None:
                        os.makedirs(self.log_dir, exist_ok=True)
                    self.save_path = os.path.join(self.log_dir, new_name)
                    print(">>> New best mean REWARD: {:.3f}. Saving to {}".format(my_mean_reward, self.save_path))
                    self.best_mean_reward = my_mean_reward
                    self.model.save(self.save_path)

                # Save by best score (preserves peak performance)
                if env_best_score > self.best_env_score and not new_best_mean:
                    self.best_env_score = env_best_score
                    score_save_path = os.path.join(self.log_dir, self.modelname + str(self.num_timesteps))
                    print(">>> New best SCORE: {:.6f}. Saving to {}".format(env_best_score, score_save_path))
                    self.model.save(score_save_path)

        return True

    def save_history(self, path: str):
        """Save episode history to a CSV file."""
        os.makedirs(path, exist_ok=True)
        csv_path = os.path.join(path, "episode_history.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["episode", "total_reward", "best_score"])
            for i, (reward, score) in enumerate(zip(self.episode_rewards, self.episode_best_scores)):
                writer.writerow([i + 1, reward, score])
        print(f"Saved episode history to {csv_path}")

    def plot_history(self, path: str):
        """Plot episode reward and score curves."""
        if not self.episode_rewards:
            return
        try:
            fig, ax1 = plt.subplots(figsize=(12, 5))
            episodes = list(range(1, len(self.episode_rewards) + 1))

            # Plot reward
            color_r = "tab:blue"
            ax1.set_xlabel("Episode", fontweight="bold")
            ax1.set_ylabel("Episode Reward", color=color_r, fontweight="bold")
            ax1.plot(episodes, self.episode_rewards, color=color_r, alpha=0.3, linewidth=0.5, label="Reward")
            # Smoothed reward (rolling mean of 10)
            if len(self.episode_rewards) > 10:
                smoothed = np.convolve(self.episode_rewards, np.ones(10)/10, mode='valid')
                ax1.plot(range(10, 10 + len(smoothed)), smoothed, color=color_r, linewidth=2, label="Reward (avg 10)")
            ax1.tick_params(axis="y", labelcolor=color_r)

            # Plot score
            ax2 = ax1.twinx()
            color_s = "tab:orange"
            ax2.set_ylabel("Best Score", color=color_s, fontweight="bold")
            ax2.plot(episodes, self.episode_best_scores, color=color_s, alpha=0.3, linewidth=0.5, label="Score")
            if len(self.episode_best_scores) > 10:
                smoothed_s = np.convolve(self.episode_best_scores, np.ones(10)/10, mode='valid')
                ax2.plot(range(10, 10 + len(smoothed_s)), smoothed_s, color=color_s, linewidth=2, label="Score (avg 10)")
            ax2.tick_params(axis="y", labelcolor=color_s)

            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

            plt.title("Training Progress", fontsize=14, fontweight="bold")
            fig.tight_layout()
            plot_path = os.path.join(path, "training_curve.png")
            plt.savefig(plot_path, dpi=200)
            plt.close()
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
    parser.add_argument("--target_update_interval", type=int, default=1000, help="Target update interval (default: 500)")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm (default: 10.0)")
    parser.add_argument("--net_arch", type=int, nargs="+", default=[256, 256], help="Hidden layers architecture (default: [256, 256])")
    parser.add_argument("--seed", type=int, default=1, help="Random seed (default: 1)")
    parser.add_argument("--ns", type=str, default="pcrl", help="Namespace / run identifier subfolder (default: pcrl)")
    parser.add_argument("--check_freq", type=int, default=1, help="Callback check frequency in episodes (default: 1)")
    parser.add_argument("--grid_penalty_weight", type=float, default=H.GRID_PENALTY_WEIGHT,
                        help="Weight on grid penalty in norm_score (default: 1.0)")
    return parser.parse_args()


if __name__ == '__main__':
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

    # Set globals before env construction
    H.GRID_PENALTY_WEIGHT = args.grid_penalty_weight
    print(f"[CONFIG] grid_penalty_weight={H.GRID_PENALTY_WEIGHT}")

    env = StationPlacement(graph_file, node_file, plan_file, location=location)

    if args.ns:
        log_dir = os.path.join("Results", "tmp", location, args.ns)
        modelname = f"best_model_{location}_{args.ns}_"
    else:
        log_dir = os.path.join("Results", "tmp", location)
        modelname = f"best_model_{location}_"

    """
    Define and train the agent
    """
    os.makedirs(log_dir, exist_ok=True)

    # Write config BEFORE training so interrupted runs can still be evaluated
    config_path = os.path.join(log_dir, "config.json")
    config_data = vars(args).copy()
    with open(config_path, "w") as f:
        json.dump(config_data, f, indent=2)
    print(f"Config saved to {config_path}")

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

    # Save episode history and plot
    callback.save_history(log_dir)
    callback.plot_history(log_dir)
    print(f"\nTraining completed! Results saved to {log_dir}")