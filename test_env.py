import os
import sys

# Ensure parent is in path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from custom_environment.StationPlacementEnv import StationPlacement

location = "DongDa_partial"
base_dir = os.path.join(current_dir, "custom_environment", "data")
graph_file = os.path.join(base_dir, "Graph", location, location + ".graphml")
node_file = os.path.join(base_dir, "Graph", location, "nodes_extended_" + location + ".txt")
plan_file = os.path.join(base_dir, "Graph", location, "existingplan_" + location + ".pkl")

print(f"Loading env for {location}...")
env = StationPlacement(graph_file, node_file, plan_file, location=location, obs_type="mlp")

obs, info = env.reset()
print(f"After reset:")
print(f"Previous score: {env.previous_score}")
print(f"Best score: {env.best_score}")

print("\nTaking step 1 (action 0)...")
obs, reward, terminated, truncated, info = env.step(1)
print(f"Reward: {reward}")
print(f"Terminated: {terminated}")
print(f"Previous score: {env.previous_score}")
print(f"Best score: {env.best_score}")
