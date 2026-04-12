from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.callbacks import BaseCallback
try:
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.wrappers import ActionMasker
    _HAS_SB3_CONTRIB = True
except ImportError:
    from stable_baselines3 import PPO as MaskablePPO
    _HAS_SB3_CONTRIB = False
    print("Warning: sb3-contrib not found. Install it with `pip install sb3-contrib` for action masking support.")
import os
import numpy as np
import torch
import random
import sys

# Add project root to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.append(project_root)

from custom_environment.StationPlacementEnv import StationPlacement
from custom_environment.gnn_extractor import GNNFeaturesExtractor

"""
Train the model by reinforcement learning (PPO).
"""

class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Code from Stable Baselines3
    Callback for saving a model based on the training reward.
    """
    def __init__(self, check_freq: int, my_log_dir: str, my_modelname: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = my_log_dir
        self.modelname = my_modelname
        self.save_path = os.path.join(self.log_dir, self.modelname)
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            x, y = ts2xy(load_results(self.log_dir), 'timesteps')
            if len(x) > 0:
                my_mean_reward = np.mean(y[-10:])  # Mean reward over last 10 episodes
                if self.verbose > 0:
                    print(f"Num timesteps: {self.num_timesteps}")
                    print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward: {my_mean_reward:.2f}")

                if my_mean_reward > self.best_mean_reward:
                    self.best_mean_reward = my_mean_reward
                    new_name = self.modelname + str(self.num_timesteps)
                    final_path = os.path.join(self.log_dir, new_name)
                    if self.verbose > 0:
                        print(f"New best mean reward: {self.best_mean_reward:.2f}")
                        print(f"Saving new best model to {final_path}")
                    self.model.save(final_path)
                else:
                    new_name = self.modelname + str(self.num_timesteps)
                    final_path = os.path.join(self.log_dir, new_name)
                    if self.verbose > 0:
                        print(f"Saving model on frequency to {final_path}")
                    self.model.save(final_path)
        return True

def train_ppo(location="DongDa", total_timesteps=100000):
    os.environ['PYTHONHASHSEED'] = '0'
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Instantiate the env
    base_dir = os.path.join(project_root, "custom_environment", "data")
    graph_file = os.path.join(base_dir, "Graph", location, location + ".graphml")
    node_file = os.path.join(base_dir, "Graph", location, "nodes_extended_" + location + ".txt")
    plan_file = os.path.join(base_dir, "Graph", location, "existingplan_" + location + ".pkl")

    base_env = StationPlacement(graph_file, node_file, plan_file, location=location, obs_type="gnn")
    log_dir = f"Results/ppo/{location}/"
    modelname = f"best_model_ppo_{location}_"
    os.makedirs(log_dir, exist_ok=True)

    # 3: Wrap with ActionMasker if sb3-contrib is available
    if _HAS_SB3_CONTRIB:
        env = ActionMasker(base_env, lambda e: e.action_masks())
    else:
        env = base_env

    # 7.1: Pass edge_index to extractor so norm_adj is cached at init time
    policy_kwargs = dict(
        features_extractor_class=GNNFeaturesExtractor,
        features_extractor_kwargs=dict(features_dim=256, edge_index_np=base_env.edge_index_array),
        net_arch=dict(pi=[256, 256], vf=[256, 256])
    )

    model = MaskablePPO("MultiInputPolicy", env, verbose=1,
                        learning_rate=3e-4,
                        n_steps=4096,
                        batch_size=128,
                        n_epochs=10,
                        gamma=0.99,
                        gae_lambda=0.95,
                        clip_range=0.2,
                        ent_coef=0.01,
                        vf_coef=0.5,
                        max_grad_norm=0.5,
                        policy_kwargs=policy_kwargs,
                        device='cuda' if torch.cuda.is_available() else 'cpu',
                        seed=seed)

    callback = SaveOnBestTrainingRewardCallback(check_freq=4096, my_log_dir=log_dir, my_modelname=modelname)
    model.learn(total_timesteps=total_timesteps, callback=callback)
    
    final_model_path = os.path.join(log_dir, f"ppo_final_{location}")
    model.save(final_model_path)
    print(f"Final PPO model saved to {final_model_path}")

if __name__ == '__main__':
    # Default training for DongDa
    train_ppo(location="DongDa", total_timesteps=500000)
