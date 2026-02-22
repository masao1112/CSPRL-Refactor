from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
import pickle
from custom_environment.StationPlacementEnv import StationPlacement
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

env = StationPlacement(graph_file, node_file, plan_file)
log_dir = f"Results/tmp/{location}/"

"""
Start evaluating
"""
print("Evaluation for best model")
env = Monitor(env, log_dir)  # new environment for evaluation

step = 56800
model = DQN.load(log_dir + "best_model_" + location + f"_{step}")

obs, _ = env.reset()
done = False
best_plan, best_node_list = None, None
action_history = []
while not done:
    action, _states = model.predict(obs, deterministic=True)
    action_history.append(action.item())
    obs, reward, done, info, _ = env.step(action)
    env.render()
    if done:
        best_node_list, best_plan = env.render()

sns.countplot(x=action_history)
plt.title('Frequency of Chosen Actions')
plt.show()

pickle.dump(best_plan, open("Results/" + "optimal_plan/" + location + f"/plan_RL_{step}.pkl", "wb"))
with open("Results/" + "optimal_plan/"  + location + f"/nodes_RL_{step}.txt", 'w') as file:
    file.write(str(best_node_list))
