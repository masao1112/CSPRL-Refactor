from collections import deque

from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.callbacks import BaseCallback
import os
import numpy as np
import torch
import random
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
        self.scores = deque(maxlen=10)
        self.best_mean_score = -np.inf
        self.n_episodes = 0

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        # Check if an episode finished
        if self.locals["dones"][0]:
            self.n_episodes += 1

            if self.n_episodes % self.check_freq == 0:
                # Query the environment for the best_score
                try:
                    # training_env is usually a VecEnv in SB3
                    env_best_score = self.training_env.get_attr('last_episode_best_score')[0]
                except Exception:
                    env_best_score = -np.inf

                # Store score for mean calculation
                self.scores.append(env_best_score)

                # Mean training score over the last 10 checks
                my_mean_score = np.mean(self.scores)

                if self.verbose > 0:
                    print("Num timesteps: {}, Episode: {}".format(self.num_timesteps, self.n_episodes))
                    print("Current best_score: {:.2f} - Mean score: {:.2f} (Best Mean: {:.2f})".format(
                        env_best_score, my_mean_score, self.best_mean_score))

                if my_mean_score > self.best_mean_score:
                    self.best_mean_score = my_mean_score
                    if self.verbose > 0:
                        print("New best mean score: {:.2f}. Saving model...".format(self.best_mean_score))
                        new_name = self.modelname + str(self.num_timesteps)
                        if self.log_dir is not None:
                            os.makedirs(self.log_dir, exist_ok=True)
                        self.save_path = os.path.join(self.log_dir, new_name)
                    self.model.save(self.save_path)

        return True


if __name__ == '__main__':
    #pip uninstall -r requirements.txt -ypip uninstall -r requirements.txt -ypip uninstall -r requirements.txt -y set a seed for reproducibility
    os.environ['PYTHONASHSEED'] = '0'
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    seed = 1
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    np.random.seed(seed)
    random.seed(seed)
    # Instantiate the env
    location = "DongDa"  # take a location of your choice
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "custom_environment", "data")
    graph_file = os.path.join(base_dir, "Graph", location, location + ".graphml")
    node_file = os.path.join(base_dir, "Graph", location, "nodes_extended_" + location + ".txt")
    plan_file = os.path.join(base_dir, "Graph", location, "existingplan_" + location + ".pkl")

    env = StationPlacement(graph_file, node_file, plan_file, location=location)
    log_dir = f"Results/tmp6/{location}/"
    modelname = "best_model_" + location + "_"

    """
    Define and train the agent 
    """
    os.makedirs(log_dir, exist_ok=True)
    env = Monitor(env, os.path.join(log_dir, "monitor.csv"))
    policy_kwargs = dict(net_arch=[256, 256]) # hidden layers
    model = DQN("MlpPolicy", env, verbose=1, batch_size=128, buffer_size=10000, learning_rate=1e-5,
                exploration_initial_eps=1, exploration_final_eps=0.05, exploration_fraction=0.2,
                policy_kwargs=policy_kwargs,
                device='cuda' if torch.cuda.is_available() else 'cpu', seed=seed)
    callback = SaveOnBestTrainingRewardCallback(check_freq=5, my_log_dir=log_dir, my_modelname=modelname)
    model.learn(total_timesteps=200000, log_interval=10 ** 4, callback=callback)