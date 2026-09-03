import argparse
import os
import pickle
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor

import custom_environment.helpers as H
from custom_environment.StationPlacementEnv import StationPlacement

"""
Generate a charging plan and evaluate performance based on trained RL model.
"""

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained RL model for station placement")
    parser.add_argument("--location", type=str, default="DongDa", help="Location/District name (default: DongDa)")
    parser.add_argument("--model_path", type=str, default=None, help="Explicit path to model .zip file")
    parser.add_argument("--log_dir", type=str, default=None, help="Directory containing saved checkpoints")
    parser.add_argument("--ns", type=str, default=None, help="Namespace / config subfolder (e.g. pcrl, config_3)")
    parser.add_argument("--step", type=int, default=None, help="Specific step number of checkpoint to load")
    parser.add_argument("--no_plot", action="store_true", help="Do not display matplotlib plot")
    return parser.parse_args()


def find_model_path(args, default_log_dir):
    if args.model_path and os.path.exists(args.model_path):
        return args.model_path

    search_dirs = []
    if args.log_dir:
        search_dirs.append(args.log_dir)
    if default_log_dir:
        search_dirs.append(default_log_dir)

    for directory in search_dirs:
        if not os.path.exists(directory):
            continue

        if args.step is not None:
            matches = glob.glob(os.path.join(directory, f"*_{args.step}.zip"))
            if matches:
                return matches[0]

        # Find latest checkpoint with highest step count
        all_zips = glob.glob(os.path.join(directory, "*.zip"))
        if all_zips:
            # Sort by step number in filename if possible
            def get_step_num(filename):
                base = os.path.splitext(os.path.basename(filename))[0]
                parts = base.split("_")
                try:
                    return int(parts[-1])
                except ValueError:
                    return 0
            all_zips.sort(key=get_step_num)
            return all_zips[-1]

    return None


def main():
    args = parse_args()
    location = args.location
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "custom_environment", "data")

    graph_file = os.path.join(base_dir, "Graph", location, f"{location}.graphml")
    
    # Check node file
    node_file = os.path.join(base_dir, "Graph", location, f"nodes_extended_{location}.txt")
    if not os.path.exists(node_file):
        alt_node_file = os.path.join(base_dir, "Graph", location, f"og_nodes_extended_{location}.txt")
        if os.path.exists(alt_node_file):
            node_file = alt_node_file

    # Check plan file
    plan_file = os.path.join(base_dir, "Graph", location, f"existingplan_{location}.pkl")
    if not os.path.exists(plan_file):
        alt_plan_file = os.path.join(base_dir, "Graph", location, f"new_existingplan_{location}.pkl")
        if os.path.exists(alt_plan_file):
            plan_file = alt_plan_file

    print(f"\n--- Initializing Evaluation for {location} ---")
    print(f"Graph file: {graph_file}")
    print(f"Node file:  {node_file}")
    print(f"Plan file:  {plan_file}")

    # Determine log_dir
    if args.log_dir:
        log_dir = args.log_dir
    elif args.ns:
        log_dir = os.path.join("Results", "tmp", location, args.ns)
    else:
        # Check standard candidate directories
        candidate_pcrl = os.path.join("Results", "tmp", location, "pcrl")
        candidate_base = os.path.join("Results", "tmp", location)
        candidate_mlp = os.path.join("Results", "tmp", location, "mlp")
        if os.path.exists(candidate_pcrl):
            log_dir = candidate_pcrl
        elif os.path.exists(candidate_mlp):
            log_dir = candidate_mlp
        else:
            log_dir = candidate_base

    model_file = find_model_path(args, log_dir)
    if model_file is None:
        print(f"\n[Warning] Could not find any checkpoint in {log_dir}.")
        print("Please specify a valid --model_path or --log_dir.")
        return

    print(f"Loading model checkpoint: {model_file}")

    # Instantiate environment
    env = StationPlacement(graph_file, node_file, plan_file, location=location)
    os.makedirs(log_dir, exist_ok=True)
    monitor_env = Monitor(env, log_dir)

    model = DQN.load(model_file, env=monitor_env)

    obs, _ = monitor_env.reset()
    done = False
    best_plan, best_node_list = None, None
    action_history = []
    total_reward = 0.0
    step_count = 0

    print("\n--- Running Evaluation Episode ---")
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        action_history.append(int(action))

        obs, reward, terminated, truncated, info = monitor_env.step(action)
        total_reward += reward
        step_count += 1
        done = terminated or truncated

        if done:
            best_node_list, best_plan = env.render()
            break

    print(f"\nEpisode finished in {step_count} steps. Total reward: {total_reward:.4f}")

    # Calculate final evaluation metrics
    if best_plan:
        if env.grid_adapter:
            station_nodes = [(s[0], s[2]["capability"]) for s in best_plan]
            dist_p, cap_p, grid_util, grid_dist = env.grid_adapter.calculate_grid_penalty(station_nodes)
            total_grid_penalty = {'dist_penalty': dist_p, 'cap_penalty': cap_p}
            final_score, benefit, cost, charg_time, wait_time, travel_cost, fairness = H.norm_score(
                best_plan, best_node_list,
                env.plan_instance.norm_benefit, env.plan_instance.norm_charg,
                env.plan_instance.norm_wait, env.plan_instance.norm_travel,
                total_grid_penalty
            )
            violations = env.grid_adapter.get_grid_violations(station_nodes)
        else:
            final_score, benefit, cost, charg_time, wait_time, travel_cost, fairness = H.norm_score(
                best_plan, best_node_list,
                env.plan_instance.norm_benefit, env.plan_instance.norm_charg,
                env.plan_instance.norm_wait, env.plan_instance.norm_travel
            )
            violations = []
            dist_p, cap_p = 0.0, 0.0

        used_budget = H.BUDGET - env.budget

        print("\n" + "=" * 60)
        print("EVALUATION METRICS SUMMARY")
        print("=" * 60)
        print(f"Total Stations:       {len(best_plan)}")
        print(f"Normalized Score:     {final_score:.4f}")
        print(f"Social Benefit:       {benefit:.4f}")
        print(f"Social Cost:          {cost:.4f}")
        print(f"Social Fairness:      {fairness:.4f}")
        print(f"Travel Cost:          {travel_cost:.4f}")
        print(f"Charging Time:        {charg_time:.4f}")
        print(f"Waiting Time:         {wait_time:.4f}")
        print(f"Used Budget:          {used_budget:,.0f} / {H.BUDGET:,.0f}")
        if env.grid_adapter:
            print(f"Grid Distance Penalty:{dist_p:.4f}")
            print(f"Grid Capacity Penalty:{cap_p:.4f}")
            print(f"Grid Violations:      {len(violations)} buses overloaded")
            if violations:
                for v in violations:
                    print(f"  - Bus {v['bus_idx']} ({v['bus_name']}): shortage = {v['shortage_mw']:.3f} MW")
        print("=" * 60)

        # Save optimal plan
        output_dir = os.path.join("Results", "optimal_plan", location)
        os.makedirs(output_dir, exist_ok=True)
        step_tag = args.step if args.step is not None else "best"

        plan_out = os.path.join(output_dir, f"plan_RL_{step_tag}.pkl")
        nodes_out = os.path.join(output_dir, f"nodes_RL_{step_tag}.txt")

        with open(plan_out, "wb") as f:
            pickle.dump(best_plan, f)
        with open(nodes_out, "w") as f:
            f.write(str(best_node_list))

        print(f"\nSaved optimal plan to: {plan_out}")
        print(f"Saved optimal nodes to: {nodes_out}")

    # Plot action frequency
    if not args.no_plot and action_history:
        try:
            plt.figure(figsize=(8, 5))
            sns.countplot(x=action_history)
            plt.title(f"Frequency of Chosen Actions - {location}")
            plt.xlabel("Action (0: build benefit, 1: build demand, 2: add benefit, 3: add demand, 4: move)")
            plt.ylabel("Count")
            plot_path = os.path.join("Results", "optimal_plan", location, "action_frequency.png")
            plt.savefig(plot_path)
            print(f"Saved action frequency plot to: {plot_path}")
            plt.show()
        except Exception as e:
            print(f"Could not render plot: {e}")


if __name__ == '__main__':
    main()
