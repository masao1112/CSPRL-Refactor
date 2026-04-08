import os

file_path = r"c:\Users\Admin\Documents\CSPRL-Refactor\analyze_hyperparameters.py"

with open(file_path, "r") as f:
    content = f.read()

# 1. Add imports
imports = """import glob
from stable_baselines3 import DQN
from custom_environment.StationPlacementEnv import StationPlacement
"""
content = content.replace("import argparse", imports + "\nimport argparse")

# 2. Add evaluate_model function
eval_func = """
def evaluate_model(run_dir, location, use_gnn):
    \"\"\"Evaluate the best saved model in the run directory.\"\"\"
    model_files = glob.glob(os.path.join(run_dir, "*.zip"))
    if not model_files:
        return None
    
    # Pick the most recently created best_model zip file
    best_model_file = max(model_files, key=os.path.getctime)
    
    base_dir = "custom_environment/data"
    graph_file = os.path.join(base_dir, "Graph", location, location + ".graphml")
    node_file = os.path.join(base_dir, "Graph", location, "nodes_extended_" + location + ".txt")
    plan_file = os.path.join(base_dir, "Graph", location, "existingplan_" + location + ".pkl")
    
    if not os.path.exists(plan_file):
        plan_file = None
        
    obs_type = "gnn" if use_gnn else "mlp"
    env = StationPlacement(graph_file, node_file, plan_file, location=location, obs_type=obs_type)
    
    if use_gnn:
        try:
            from custom_environment.gnn_extractor import GNNFeaturesExtractor
            custom_objects = {"GNNFeaturesExtractor": GNNFeaturesExtractor}
            model = DQN.load(best_model_file, env=env, custom_objects=custom_objects)
        except Exception as e:
            print(f" Error loading GNN model: {e}")
            return None
    else:
        try:
            model = DQN.load(best_model_file, env=env)
        except Exception as e:
            print(f" Error loading MLP model: {e}")
            return None
        
    obs, _ = env.reset()
    done = False
    total_reward = 0
    step_count = 0
    
    while not done and step_count < 2000:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        step_count += 1
        if done or truncated:
            break
            
    return total_reward

"""
# insert before load_results
content = content.replace("def load_results(experiment_dir):", eval_func + "def load_results(experiment_dir, force_eval=False):")

# 3. Patch load_results
new_load_results = """    csv_path = os.path.join(experiment_dir, "results_summary.csv")
    csv_eval_path = os.path.join(experiment_dir, "results_summary_evaluated.csv")
    detailed_path = os.path.join(experiment_dir, "results_detailed.json")
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Results file not found: {csv_path}")
    
    if os.path.exists(csv_eval_path) and not force_eval:
        print(f"Loading previously evaluated results from: {csv_eval_path}")
        df = pd.read_csv(csv_eval_path)
    else:
        df = pd.read_csv(csv_path)
        print("Evaluating models. This might take a few minutes...")
        
        eval_rewards = []
        for index, row in df.iterrows():
            run_id = int(row['run_id'])
            run_dir = os.path.join(experiment_dir, f"run_{run_id:03d}")
            
            config_path = os.path.join(run_dir, "training_config.json")
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                location = config.get("location", "DongDa_partial")
                use_gnn = config.get("use_gnn", True)
            else:
                location = "DongDa_partial"
                use_gnn = True
                
            print(f"Evaluating Run {run_id:03d}...", end="", flush=True)
            try:
                reward = evaluate_model(run_dir, location, use_gnn)
                if reward is not None:
                    print(f" Score: {reward:.4f}")
                else:
                    print(" No model found.")
                eval_rewards.append(reward)
            except Exception as e:
                print(f" Error: {e}")
                eval_rewards.append(None)
                
        df['eval_reward'] = eval_rewards
        df.to_csv(csv_eval_path, index=False)
        print(f"Saved evaluated results to {csv_eval_path}")
        
    with open(detailed_path, 'r') as f:
        detailed_results = json.load(f)
    
    return df, detailed_results"""

content_parts = content.split("def load_results(experiment_dir, force_eval=False):\n")
first_part = content_parts[0]
the_rest = content_parts[1]
end_of_func = the_rest.find("def print_summary_statistics(df):")
after_load = the_rest[end_of_func:]

content = first_part + "def load_results(experiment_dir, force_eval=False):\n" + new_load_results + "\n\n\n" + after_load

# 4. Global replacement of 'best_reward' to 'eval_reward' ONLY in the visualization functions.
# Note: we need to replace 'best_reward' with 'eval_reward' from print_summary_statistics(df): onwards
before_stats = content[:content.find("def print_summary_statistics(df):")]
after_stats = content[content.find("def print_summary_statistics(df):"):]

after_stats = after_stats.replace("'best_reward'", "'eval_reward'")
after_stats = after_stats.replace("best_reward", "eval_reward")
after_stats = after_stats.replace("Best Reward", "Eval Reward")

# Restore "best_run['eval_reward']" string format 
# Actually "best_run['eval_reward']" is correct if we replaced best_reward with eval_reward.
# The only issue is if 'best_reward' is a column name, it's now 'eval_reward'
# which is correct. "Best Reward: {best_run['eval_reward']:.4f}" is correct

content = before_stats + after_stats

# 5. Fix argument parsing to include --force_eval
new_arg = """    parser.add_argument(
        '--force_eval',
        action='store_true',
        help='Force re-evaluation of models even if evaluated results exist'
    )"""

content = content.replace("args = parser.parse_args()", new_arg + "\n    \n    args = parser.parse_args()")
content = content.replace("df, detailed_results = load_results(args.experiment_dir)", "df, detailed_results = load_results(args.experiment_dir, args.force_eval)")

with open(file_path, "w") as f:
    f.write(content)

print("Patched successfully")
