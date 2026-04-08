import os
import sys
import numpy as np

# Add the project root to sys.path
sys.path.append(os.getcwd())

# Add custom_environment to path
project_root = r"c:\Users\Admin\Documents\CSPRL-Refactor"
if project_root not in sys.path:
    sys.path.append(project_root)

from custom_environment.StationPlacementEnv import StationPlacement

def test_no_plan():
    location = "DongDa"
    # Assuming these files exist from the context
    current_dir = os.path.join(project_root, "custom_environment")
    graph_file = os.path.join(current_dir, "data", "Graph", location, location + ".graphml")
    node_file = os.path.join(current_dir, "data", "Graph", location, "nodes_extended_" + location + ".txt")
    
    print("Testing StationPlacement with plan_file=None...")
    try:
        env = StationPlacement(graph_file, node_file, None, location=location)
        obs, info = env.reset()
        print("Reset successful!")
        
        # Try a step
        obs, reward, terminated, truncated, info = env.step(0) # Build a new station
        print(f"Step successful! Reward: {reward}")
    except Exception as e:
        # print(f"Failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_no_plan()
