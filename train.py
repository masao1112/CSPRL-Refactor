from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import LinearSchedule
import os
import numpy as np
import torch
import random
import csv
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import deque
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
        self.scores = deque(maxlen=5)
        self.best_mean_score = -np.inf
        self.n_episodes = 0
        self.best_score = -np.inf
        # Episode history for plotting
        self.episode_rewards = []    # total reward per episode
        self.episode_best_scores = []  # best_score per episode

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.log_dir is not None:
            os.makedirs(self.log_dir, exist_ok=True)

    def _on_step(self) -> bool:
        # Check if an episode finished
        if self.locals["dones"][0]:
            self.n_episodes += 1
            # Query the environment for the best_score
            try:
                # training_env is usually a VecEnv in SB3
                env_best_score = self.training_env.get_attr('last_episode_best_score')[0]
            except Exception:
                env_best_score = -np.inf

            # Get total episode reward from Monitor wrapper
            info = self.locals["infos"][0]
            episode_reward = info.get("episode", {}).get("r", 0.0) if info else 0.0

            # Record history
            self.episode_rewards.append(episode_reward)
            self.episode_best_scores.append(env_best_score)

            # Store score for mean calculation
            self.scores.append(env_best_score)

            if self.n_episodes % self.check_freq == 0:
                # Mean training score over the last 10 checks
                my_mean_score = np.mean(self.scores)

                if self.verbose > 0:
                    print("Num timesteps: {}, Episode: {}".format(self.num_timesteps, self.n_episodes))
                    print("Current best_score: {:.3f} - Mean score: {:.3f} (Best Mean: {:.3f})".format(
                        env_best_score, my_mean_score, self.best_mean_score))

                new_best_mean = my_mean_score > self.best_mean_score
                new_best_score = env_best_score > self.best_score

                if new_best_mean or new_best_score:
                    new_name = self.modelname + str(self.num_timesteps)
                    if self.log_dir is not None:
                        os.makedirs(self.log_dir, exist_ok=True)
                    self.save_path = os.path.join(self.log_dir, new_name)

                    if new_best_mean and new_best_score:
                        print("New best mean score: {:.3f} and new best score: {:.3f}. Saving to {}".format(
                            my_mean_score, env_best_score, self.save_path))
                    elif new_best_mean:
                        print("New best mean score: {:.3f}. Saving to {}".format(my_mean_score, self.save_path))
                    else:
                        print("New best score: {:.3f}. Saving to {}".format(env_best_score, self.save_path))

                    if new_best_mean:
                        self.best_mean_score = my_mean_score
                    if new_best_score:
                        self.best_score = env_best_score

                    self.model.save(self.save_path)

        return True

    def save_history(self, path: str):
        """Save episode history to a CSV file."""
        os.makedirs(path, exist_ok=True)
        csv_path = os.path.join(path, "episode_history.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["episode", "total_reward", "best_score"])
            for i, (r, s) in enumerate(zip(self.episode_rewards, self.episode_best_scores), 1):
                writer.writerow([i, r, s])
        if self.verbose > 0:
            print(f"Episode history saved to {csv_path}")

    def plot_history(self, path: str):
        """Plot total reward and best score per episode and save the figure."""
        if not self.episode_rewards:
            return

        episodes = np.arange(1, len(self.episode_rewards) + 1)
        rewards = np.array(self.episode_rewards)
        best_scores = np.array(self.episode_best_scores)

        fig, ax1 = plt.subplots(figsize=(12, 5))

        color_reward = "#3b82f6"   # blue
        color_score = "#ef4444"    # red

        # Plot total reward
        ax1.set_xlabel("Episode")
        ax1.set_ylabel("Total Reward", color=color_reward)
        ax1.plot(episodes, rewards, color=color_reward, alpha=0.3, linewidth=0.8, label="Total Reward")
        # Smoothed (rolling mean, window=10)
        if len(rewards) >= 10:
            smooth_r = np.convolve(rewards, np.ones(10) / 10, mode="valid")
            ax1.plot(episodes[9:], smooth_r, color=color_reward, linewidth=2, label="Reward (MA-10)")
        ax1.tick_params(axis="y", labelcolor=color_reward)
        ax1.legend(loc="upper left")

        # Plot best score on secondary y-axis
        ax2 = ax1.twinx()
        ax2.set_ylabel("Best Score", color=color_score)
        ax2.plot(episodes, best_scores, color=color_score, alpha=0.3, linewidth=0.8, label="Best Score")
        if len(best_scores) >= 10:
            smooth_s = np.convolve(best_scores, np.ones(10) / 10, mode="valid")
            ax2.plot(episodes[9:], smooth_s, color=color_score, linewidth=2, label="Best Score (MA-10)")
        ax2.tick_params(axis="y", labelcolor=color_score)
        ax2.legend(loc="upper right")

        plt.title("Training Progress")
        fig.tight_layout()
        os.makedirs(path, exist_ok=True)
        plot_path = os.path.join(path, "training_plot.png")
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        if self.verbose > 0:
            print(f"Training plot saved to {plot_path}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Train a DQN agent for station placement.")
    parser.add_argument("--location", type=str, default="DongDa", help="District name (default: DongDa)")
    parser.add_argument("--use_gnn", action="store_true", default=True, help="Use GNN policy (default: True)")
    parser.add_argument("--no_gnn", action="store_true", help="Disable GNN, use MLP policy")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate (default: 1e-4)")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size (default: 128)")
    parser.add_argument("--buffer_size", type=int, default=10000, help="Replay buffer size (default: 10000)")
    parser.add_argument("--features_dim", type=int, default=256, help="GNN features dimension (default: 256)")
    parser.add_argument("--net_arch", type=int, nargs="+", default=[256, 256], help="Network architecture layers (default: 256 256)")
    parser.add_argument("--exploration_initial_eps", type=float, default=1.0, help="Initial exploration epsilon (default: 1.0)")
    parser.add_argument("--exploration_final_eps", type=float, default=0.05, help="Final exploration epsilon (default: 0.05)")
    parser.add_argument("--exploration_fraction", type=float, default=0.3, help="Exploration fraction (default: 0.3)")
    parser.add_argument("--total_timesteps", type=int, default=100000, help="Total training timesteps (default: 200000)")
    parser.add_argument("--seed", type=int, default=1, help="Random seed (default: 1)")
    parser.add_argument("--ns", type=str, default="", help="Namespace of your training run")
    args = parser.parse_args()

    if args.no_gnn:
        args.use_gnn = False

    # Set seed for reproducibility
    os.environ['PYTHONASHSEED'] = '0'
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Instantiate the env
    location = args.location
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "custom_environment", "data")
    graph_file = os.path.join(base_dir, "Graph", location, location + ".graphml")
    node_file = os.path.join(base_dir, "Graph", location, "nodes_extended_" + location + ".txt")
    plan_file = os.path.join(base_dir, "Graph", location, "existingplan_" + location + ".pkl")

    obs_type = "gnn" if args.use_gnn else "mlp"
    env = StationPlacement(graph_file, node_file, plan_file, location=location, obs_type=obs_type)
    if args.ns:
        log_dir = f"Results/tmp/{location}/{obs_type}/{args.ns}"
        modelname = f"best_model_{obs_type}_{location}_{args.ns}_"
    else:
        log_dir = f"Results/tmp/{location}/{obs_type}"
        modelname = f"best_model_{obs_type}_{location}_"

    """
    Define and train the agent 
    """
    os.makedirs(log_dir, exist_ok=True)
    env = Monitor(env, os.path.join(log_dir, "monitor.csv"))

    if args.use_gnn:
        from custom_environment.gnn_extractor import GNNFeaturesExtractor
        policy_kwargs = dict(
            features_extractor_class=GNNFeaturesExtractor,
            features_extractor_kwargs=dict(features_dim=args.features_dim),
            net_arch=args.net_arch
        )
        policy_type = "MultiInputPolicy"
    else:
        policy_kwargs = dict(net_arch=args.net_arch)
        policy_type = "MlpPolicy"

    model = DQN(policy_type, env, verbose=1,
                batch_size=args.batch_size,
                buffer_size=args.buffer_size,
                learning_rate=args.learning_rate,
                exploration_initial_eps=args.exploration_initial_eps,
                exploration_final_eps=args.exploration_final_eps,
                exploration_fraction=args.exploration_fraction,
                policy_kwargs=policy_kwargs,
                device='cuda' if torch.cuda.is_available() else 'cpu',
                seed=args.seed)
    callback = SaveOnBestTrainingRewardCallback(check_freq=1, my_log_dir=log_dir, my_modelname=modelname)
    model.learn(total_timesteps=args.total_timesteps, log_interval=10 ** 4, callback=callback)

    # Save config, episode history, and plot
    config_path = os.path.join(log_dir, "config.json")
    config_data = vars(args).copy()
    config_data.pop("no_gnn", None)  # redundant with use_gnn
    with open(config_path, "w") as f:
        json.dump(config_data, f, indent=2)
    print(f"Config saved to {config_path}")

    callback.save_history(log_dir)
    callback.plot_history(log_dir)