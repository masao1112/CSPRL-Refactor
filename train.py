from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.utils import get_linear_fn, LinearSchedule
import os
import numpy as np
import torch
import random
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
            # Query the environment for the best_score
            try:
                # training_env is usually a VecEnv in SB3
                env_best_score = self.training_env.get_attr('last_episode_best_score')[0]
            except Exception:
                env_best_score = -np.inf

            # Store score for mean calculation
            self.scores.append(env_best_score)

            if self.n_episodes % self.check_freq == 0:

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

    use_gnn = True  # Set to True to train with GNN policy

    obs_type = "gnn" if use_gnn else "mlp"
    env = StationPlacement(graph_file, node_file, plan_file, location=location, obs_type=obs_type)
    log_dir = f"Results/tmp/{location}/{obs_type}"
    modelname = "best_model_" + location + "_"

    """
    Define and train the agent 
    """
    os.makedirs(log_dir, exist_ok=True)
    env = Monitor(env, os.path.join(log_dir, "monitor.csv"))
    
    if use_gnn:
        from custom_environment.gnn_extractor import GNNFeaturesExtractor
        # 7.1: pass edge_index so norm_adj is cached once at init time
        _base_env = env.env if hasattr(env, 'env') else env
        policy_kwargs = dict(
            features_extractor_class=GNNFeaturesExtractor,
            features_extractor_kwargs=dict(
                features_dim=256,
                edge_index_np=_base_env.edge_index_array
            ),
            net_arch=[256, 256]
        )
        policy_type = "MultiInputPolicy"
        modelname = "best_model_gnn_" + location + "_"
    else:
        policy_kwargs = dict(net_arch=[256, 256]) # hidden layers
        policy_type = "MlpPolicy"

    lr_schedule = LinearSchedule(
        start=3e-4,  # bắt đầu cao
        end=1e-5,  # giảm dần khi gần kết thúc
        end_fraction=1.0
    )
    
    model = DQN(policy_type, env,
                verbose=1,
                batch_size=256,  # tăng để gradient ổn định hơn
                buffer_size=50000,  # ~333 episode — đủ đa dạng
                learning_rate=lr_schedule,  # chuẩn Adam cho RL
                learning_starts=2000,  # bắt đầu học sau 2000 bước random
                train_freq=4,  # học mỗi 4 bước env
                target_update_interval=500,
                exploration_initial_eps=1.0,
                exploration_final_eps=0.05,
                exploration_fraction=0.5,  # khám phá đến 100K bước (50%)
                gamma=0.99,
                policy_kwargs=policy_kwargs,
                device='cuda' if torch.cuda.is_available() else 'cpu',
                seed=seed
                )
    # callback = SaveOnBestTrainingRewardCallback(check_freq=3, my_log_dir=log_dir, my_modelname=modelname)
    # Using EvalCallback instead of creating one
    eval_env = StationPlacement(graph_file, node_file, plan_file,
                                location=location, obs_type=obs_type)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=log_dir,
        log_path=log_dir,
        eval_freq=5000,  # đánh giá mỗi 5000 bước
        n_eval_episodes=5,  # trung bình 5 episode
        deterministic=True,
        render=False
    )
    model.learn(total_timesteps=200000, log_interval=10 ** 4, callback=eval_callback)