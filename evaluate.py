from stable_baselines3 import DQN, PPO, A2C
from stable_baselines3.common.monitor import Monitor
import pickle
import osmnx as ox
from custom_environment.StationPlacementEnv import StationPlacement
from visualise import visualise_stations
import seaborn as sns
import matplotlib.pyplot as plt
import os

"""
Generate a charging plan based on the model.
"""
# Prepare the environment
base_dir = "custom_environment/data"
location = "DongDa"

graph_file = os.path.join(base_dir, "Graph", location, location + ".graphml")
node_file = os.path.join(base_dir, "Graph", location, "nodes_extended_" + location + ".txt")
plan_file = os.path.join(base_dir, "Graph", location, "existingplan_" + location + ".pkl")

ns = "ppo_test"
use_gnn = True  # Default
obs_type = "gnn" if use_gnn else "mlp"

# Updated to match the log directory used in train.py (Results/tmp/gnn/)
if ns is not None:
    log_dir = f"Results/tmp/{location}/{obs_type}/{ns}/"
else:
    log_dir = f"Results/tmp/{location}/{obs_type}/"

# Load config to correctly reconstruct environment parameters
config_path = os.path.join(log_dir, "config.json")
action_type = "multidiscrete"
algo = "ppo"
if os.path.exists(config_path):
    import json
    with open(config_path, "r") as f:
        config_data = json.load(f)
        action_type = config_data.get("action_type", "discrete")
        use_gnn = config_data.get("use_gnn", use_gnn)
        obs_type = "gnn" if use_gnn else "mlp"
        algo = config_data.get("algo", "dqn")

env = StationPlacement(graph_file, node_file, plan_file, location=location, obs_type=obs_type, action_type=action_type)

"""
Start evaluating
"""
print("Evaluation for best model")
os.makedirs(log_dir, exist_ok=True)
env = Monitor(env, log_dir)  # new environment for evaluation
G = ox.load_graphml(graph_file)

step = 106422

ModelClass = {"dqn": DQN, "ppo": PPO, "a2c": A2C}[algo.lower()]

if use_gnn:
    from custom_environment.gnn_extractor import GNNFeaturesExtractor

    custom_objects = {"GNNFeaturesExtractor": GNNFeaturesExtractor}
    if ns is not None:
        model = ModelClass.load(log_dir + "best_model_gnn_" + location + f"_{ns}_{step}.zip", env=env,
                                custom_objects=custom_objects)
    else:
        model = ModelClass.load(log_dir + "best_model_gnn_" + location + f"_{step}.zip", env=env,
                                custom_objects=custom_objects)
else:
    model = ModelClass.load(log_dir + "best_model_" + location + f"_{step}.zip", env=env)

obs, _ = env.reset()
done = False
best_plan, best_node_list = None, None
action_history = []
total_reward = 0
while not done:
    action, _states = model.predict(obs, deterministic=True)
    action_history.append(action)

    # Gymnasium steps return 5 variables
    obs, reward, done, truncated, info = env.step(action)
    env.render()  # print out the evaluation

    # In StationPlacementEnv, done acts as terminated
    if done or truncated:
        best_node_list, best_plan = env.render()
        break

sns.countplot(x=action_history)
plt.title('Frequency of Chosen Actions')
plt.show()

output_dir = os.path.join("Results", "optimal_plan", location)
os.makedirs(output_dir, exist_ok=True)

pickle.dump(best_plan, open(os.path.join(output_dir, f"plan_RL_{step}.pkl"), "wb"))
with open(os.path.join(output_dir, f"nodes_RL_{step}.txt"), 'w') as file:
    file.write(str(best_node_list))
