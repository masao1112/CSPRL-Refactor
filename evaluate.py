from stable_baselines3 import DQN
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
location = "DongDa_partial"

graph_file = os.path.join(base_dir, "Graph", location, location + ".graphml")
node_file = os.path.join(base_dir, "Graph", location, "nodes_extended_" + location + ".txt")
plan_file = os.path.join(base_dir, "Graph", location, "existingplan_" + location + ".pkl")

use_gnn = True  # Set to True if evaluating a GNN model
obs_type = "gnn" if use_gnn else "mlp"
env = StationPlacement(graph_file, node_file, plan_file, location=location, obs_type=obs_type)

# Model directory where training was saved
model_dir = f"Results/hyperparameter_tuning/experiment_20260406_144114/run_137/"

# Evaluation output directory
eval_dir = "Results/tmp/evaluation/"

"""
Start evaluating
"""
print("Evaluation for best model")
os.makedirs(eval_dir, exist_ok=True)
env = Monitor(env, eval_dir)  # new environment for evaluation
G = ox.load_graphml(graph_file)

step = 10800
if use_gnn:
    from custom_environment.gnn_extractor import GNNFeaturesExtractor
    custom_objects = {"GNNFeaturesExtractor": GNNFeaturesExtractor}
    model = DQN.load(model_dir + "best_model_gnn_" + location + f"_{step}.zip", env=env, custom_objects=custom_objects)
else:
    model = DQN.load(model_dir + "best_model_" + location + f"_{step}.zip", env=env)

obs, _ = env.reset()
done = False
best_plan, best_node_list = None, None
action_history = []
total_reward = 0
step_count = 0
episode_rewards = []

print(f"\nStarting evaluation...")
print(f"Location: {location}")
print(f"Model step checkpoint: {step}")

while not done:
    action, _states = model.predict(obs, deterministic=True)
    action_history.append(action.item())
    
    # Gymnasium steps return 5 variables
    obs, reward, done, truncated, info = env.step(action)
    episode_rewards.append(reward)
    total_reward += reward
    step_count += 1

    print(f"Step {step_count}: Action={action.item()}, Reward={reward:.4f}, Total={total_reward:.4f}")
    
    # In StationPlacementEnv, done acts as terminated
    if done or truncated:
        best_node_list, best_plan = env.render()
        print(f"\nEpisode completed!")
        print(f"Total steps: {step_count}")
        print(f"Total reward: {total_reward:.4f}")
        print(f"Average reward per step: {total_reward/step_count:.4f}")
        print(f"Best reward in episode: {max(episode_rewards):.4f}")
        print(f"Worst reward in episode: {min(episode_rewards):.4f}")
        break

sns.countplot(x=action_history)
plt.title('Frequency of Chosen Actions')
plt.savefig(os.path.join(eval_dir, f"action_distribution.png"), dpi=100, bbox_inches='tight')
print(f"Saved action distribution plot to: {os.path.join(eval_dir, 'action_distribution.png')}")
plt.close()

output_dir = os.path.join("Results", "optimal_plan", location)
os.makedirs(output_dir, exist_ok=True)

pickle.dump(best_plan, open(os.path.join(output_dir, f"plan_RL_{step}.pkl"), "wb"))
print(f"Saved plan to: {os.path.join(output_dir, f'plan_RL_{step}.pkl')}")

with open(os.path.join(output_dir, f"nodes_RL_{step}.txt"), 'w') as file:
    file.write(str(best_node_list))
print(f"Saved nodes to: {os.path.join(output_dir, f'nodes_RL_{step}.txt')}")
print(f"\nEvaluation results saved to: {output_dir}\n")
