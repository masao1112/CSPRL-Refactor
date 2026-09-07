import os
import sys
import csv
import json
import pickle
import argparse
from math import ceil

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import osmnx as ox
from stable_baselines3 import DQN, PPO

# Add project root to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

import custom_environment.helpers as H
from custom_environment.StationPlacementEnv import StationPlacement
from algorithms.ga.ga_utils import GAPolicy, unflatten_weights


def prepare_existing_plan(my_plan, my_node_list, graph):
    my_cost_dict = {}
    my_node_dict = {}
    for my_node in my_node_list:
        my_node_dict[my_node[0]] = {}  # prepare node_dict
        my_node[1]["charging station"] = None
        my_node[1]["distance"] = None

    for index in range(len(my_plan)):
        my_plan[index] = H.s_dictionnary(my_plan[index], my_node_list)

    my_node_list, _, _ = H.station_seeking(my_plan, my_node_list, my_node_dict, my_cost_dict, graph)

    for index in range(len(my_plan)):
        my_plan[index] = H.s_dictionnary(my_plan[index], my_node_list)

    return my_node_list, my_plan


def eci_test(
    my_plan,
    my_node_list,
    my_norm_benefit,
    my_norm_charging,
    my_norm_waiting,
    my_norm_travel,
    grid_penalty=None,
):
    score, benefit, cost, charg_time, wait_time, cost_travel, _ = H.norm_score(
        my_plan,
        my_node_list,
        my_norm_benefit,
        my_norm_charging,
        my_norm_waiting,
        my_norm_travel,
        grid_penalty,
    )
    return score


def run_episode(agent, env, agent_type="rl", max_steps=None, eval_grid_penalty_weight=None):
    """Run one episode. max_steps cuts it short for quick behaviour probes on the
    large districts -- the resulting scores are NOT comparable to a full run.

    eval_grid_penalty_weight rescores the resulting plan under a different grid
    penalty weight than the episode ran with. This is what an ablation needs: a
    policy trained at w_g=0 must ACT under w_g=0 (the weight reaches its
    observation through global_state[2] = best_score - starting_score, so running
    it at w_g=1 feeds it a signal it never saw), but must be REPORTED under the
    full metric, otherwise the overloading it causes stays hidden. Leave it None
    to score with whatever weight the episode used.
    """
    obs, _ = env.reset(seed=1)
    total_reward = 0
    terminated = False
    truncated = False
    overloaded = False
    n_steps = 0

    while not (terminated or truncated):
        if max_steps is not None and n_steps >= max_steps:
            print(f"  [TRUNCATED] stopped at {max_steps} steps with "
                  f"{len(env.plan_instance.plan)} stations; scores below are a probe, not a result.")
            break
        n_steps += 1
        if agent_type in ["rl", "dqn", "ppo"]:
            action, _ = agent.predict(obs, deterministic=True)
        elif agent_type == "ga":
            action = agent.select_action(obs)
        elif agent_type in ["greedy_benefit", "greedy_demand"]:
            station_list = [s[0][0] for s in env.plan_instance.plan]
            free_list = [node for node in env.node_list if node[0] not in station_list]
            
            # Reset overloaded at each step
            overloaded = False
            current_wait_metric = 0.0

            if env.plan_instance.plan:
                score, benefit, cost, charg_time, wait_time, cost_travel, _ = H.norm_score(
                    env.plan_instance.plan,
                    env.node_list,
                    env.plan_instance.norm_benefit,
                    env.plan_instance.norm_charg,
                    env.plan_instance.norm_wait,
                    env.plan_instance.norm_travel,
                    None
                )
                current_wait_metric = wait_time
                # If any relevant metric exceeds 100%, we start using 'add' action (overloaded = True)
                if wait_time > 1.0 or charg_time > 1.0:
                    overloaded = True

            # Tracking the waiting time dynamically
            # print(f"[{agent_type}] Current Wait (%): {current_wait_metric*100:.2f}% | Stations: {len(env.plan_instance.plan)}")

            if agent_type == "greedy_benefit":
                if overloaded:
                    action = 2  # Add charger
                elif free_list:
                    action = 0  # New station
                else:
                    action = 2
            else:
                if overloaded:
                    action = 3  # Add charger
                elif free_list:
                    action = 1  # New station
                else:
                    action = 3

        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward

    best_node_list, best_plan = env.render()

    # Calculate all metrics
    norm_benefit = env.plan_instance.norm_benefit
    norm_charg = env.plan_instance.norm_charg
    norm_wait = env.plan_instance.norm_wait
    norm_travel = env.plan_instance.norm_travel

    grid_penalty = None
    dist_penalty = 0.0
    cap_penalty = 0.0
    grid_violations = []
    if env.grid_adapter:
        station_nodes = [(s[0], s[2]["capability"]) for s in best_plan]
        dist_penalty, cap_penalty, _, _ = env.grid_adapter.calculate_grid_penalty(station_nodes)
        grid_penalty = {'dist_penalty': dist_penalty, 'cap_penalty': cap_penalty}
        grid_violations = env.grid_adapter.get_grid_violations(station_nodes)

    # Rescore only; best_plan itself was already chosen under the episode's own
    # weight, which is the plan this agent would actually deploy.
    episode_wg = H.GRID_PENALTY_WEIGHT
    if eval_grid_penalty_weight is not None:
        H.GRID_PENALTY_WEIGHT = eval_grid_penalty_weight
    try:
        score, benefit, cost, charg_time, wait_time, cost_travel, fairness = H.norm_score(
            best_plan,
            best_node_list,
            norm_benefit,
            norm_charg,
            norm_wait,
            norm_travel,
            grid_penalty,
        )
    finally:
        H.GRID_PENALTY_WEIGHT = episode_wg

    # if env.grid_adapter and cap_penalty < 0:
    #     score -= 100

    travel_max = H.travel_metric(best_node_list)
    wait_max = H.waiting_metric(best_plan)

    # Budget calculation (consistent with run_metrics.py) -- basic_cost is the
    # snapshot taken at reset(); the fees on extend_existing_plan grow in place as
    # the agent upgrades existing stations, so they cannot serve as the baseline.
    basic_cost = env.plan_instance.basic_cost
    total_inst_cost = (sum(station[2]["fee"] for station in best_plan) - basic_cost) / H.BUDGET

    metrics = {
        "reward": total_reward,
        "score": score * 100,
        "benefit": benefit * 100,
        "cost": cost * 100,
        "fairness": fairness * 100,

        "charg_time": charg_time * 100,
        "wait_time": wait_time * 100,
        "cost_travel": cost_travel * 100,
        "travel_max": travel_max,
        "wait_max": wait_max,
        "num_stations": len(best_plan),
        "used_budget": total_inst_cost * 100,
        "dist_penalty": dist_penalty,
        "cap_penalty": cap_penalty,
        "grid_violations": grid_violations,
    }

    return metrics, best_plan, best_node_list


# ── Visualization helpers (from visualise.py) ──────────────────────────


def nodesize(station_list, my_graph, my_plan):
    ns = []
    for node in my_graph.nodes():
        if node not in station_list:
            ns.append(2)
        else:
            i = station_list.index(node)
            station = my_plan[i]
            try:
                capacity = station[2]["capability"]
            except (KeyError, IndexError):
                capacity = np.sum(H.CHARGING_POWER * station[1])
                
            if capacity < 100:
                ns.append(6)
            elif 100 <= capacity < 200:
                ns.append(11)
            elif 200 <= capacity < 300:
                ns.append(16)
            else:
                ns.append(21)
    return ns


def visualise_stations(my_graph, my_plan, my_filepath, title=None):
    """Create plot of the charging station distribution."""
    station_list = [station[0][0] for station in my_plan]
    colours = ["r", "grey"]
    nc = [colours[0] if node in station_list else colours[1] for node in my_graph.nodes()]
    labels = ["Charging station", "Normal road junction"]
    
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", lw=0, markerfacecolor=colours[0], markersize=7),
        Line2D([0], [0], marker="o", color="w", lw=0, markerfacecolor=colours[1], markersize=4),
    ]
    ns = nodesize(station_list, my_graph, my_plan)
    
    fig, ax = ox.plot_graph(
        my_graph,
        node_color=nc,
        save=False,
        node_size=ns,
        edge_linewidth=0.2,
        edge_alpha=0.8,
        show=False,
        close=False,
    )
    ax.legend(legend_elements, labels, loc=2, prop={"size": 10})
    
    if title:
        ax.set_title(title, fontsize=14, fontweight="bold", color="white")
        
    os.makedirs(os.path.dirname(my_filepath), exist_ok=True)
    plt.savefig(my_filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Map saved: {my_filepath}")


# ── Model loading ──────────────────────────────────────────────────────
def extractor_custom_objects(obs_type):
    """SB3 cannot unpickle a policy whose custom features extractor class is not
    importable at load time, so hand it the class for the obs_types that use one."""
    if obs_type == "gnn":
        from custom_environment.gnn_extractor import GNNFeaturesExtractor
        return {"GNNFeaturesExtractor": GNNFeaturesExtractor}
    if obs_type == "attention":
        from custom_environment.attention_extractor import AttentionFeaturesExtractor
        return {"AttentionFeaturesExtractor": AttentionFeaturesExtractor}
    return None


def resolve_rl_model(rl_log_dir, location, obs_type, ns="", step=None, algo="dqn"):
    """Locate the checkpoint to compare against; returns (path_or_None, step).

    train.py names checkpoints:
      DQN: best_model_{obs_type}_{location}_{ns}_{step}.zip (or without ns)
      PPO: best_model_ppo_{obs_type}_{location}_{ns}_{step}.zip (or without ns)

    With no explicit step, prefer the ranking evaluate_all.py already wrote into
    evaluation_results.csv (it sorts best-first).
    """
    algo_dir = "" if algo == "dqn" else "ppo"
    algo_prefix = "" if algo == "dqn" else "ppo_"

    # Resolve base directory for the algorithm
    if algo_dir and not rl_log_dir.rstrip("/\\").endswith(algo_dir):
        base_dir = os.path.join(rl_log_dir, algo_dir)
    else:
        base_dir = rl_log_dir

    # Check run directory with ns, falling back to base_dir if ns is empty or dir with ns doesn't exist
    run_dir = os.path.join(base_dir, ns) if ns else base_dir
    prefix = f"best_model_{algo_prefix}{obs_type}_{location}_{ns}_" if ns else f"best_model_{algo_prefix}{obs_type}_{location}_"

    if not os.path.isdir(run_dir):
        # If ns was provided but run_dir doesn't exist, check base_dir
        if ns and os.path.isdir(base_dir):
            run_dir = base_dir
            prefix = f"best_model_{algo_prefix}{obs_type}_{location}_"
        else:
            print(f"Warning: run directory not found: {run_dir}")
            return None, step

    if step is not None:
        path = os.path.join(run_dir, f"{prefix}{step}.zip")
        if os.path.exists(path):
            return path, step
        # Also check without ns in prefix if not found
        alt_prefix = f"best_model_{algo_prefix}{obs_type}_{location}_"
        alt_path = os.path.join(run_dir, f"{alt_prefix}{step}.zip")
        if os.path.exists(alt_path):
            return alt_path, step
        print(f"Warning: requested checkpoint not found: {path}")
        return None, step

    eval_csv = os.path.join(run_dir, "evaluation_results.csv")
    if os.path.exists(eval_csv):
        with open(eval_csv, newline="") as f:
            rows = list(csv.DictReader(f))
        if rows:
            best = rows[0]
            path = os.path.join(run_dir, best["model_file"])
            if os.path.exists(path):
                print(f"Selected best-scoring [{algo.upper()}] checkpoint from evaluation_results.csv "
                      f"(step {best['step']}, score {float(best['score']):.4f})")
                return path, int(best["step"])

    checkpoints = {}
    for fname in os.listdir(run_dir):
        if fname.endswith(".zip") and (fname.startswith(prefix) or (algo == "ppo" and "ppo" in fname)):
            try:
                # Extract trailing step number before .zip
                base_name = fname[:-len(".zip")]
                step_val = int(base_name.split("_")[-1])
                checkpoints[step_val] = fname
            except ValueError:
                continue
    if not checkpoints:
        print(f"Warning: no checkpoints matching {prefix}*.zip in {run_dir}")
        return None, step

    latest = max(checkpoints)
    print(f"Warning: no evaluation_results.csv in {run_dir}. Falling back to the LATEST "
          f"[{algo.upper()}] checkpoint (step {latest}), which is not necessarily the best-scoring one. "
          f"Run evaluate_all.py on this run first for a fair comparison.")
    return os.path.join(run_dir, checkpoints[latest]), latest


def load_config_for_run(location, obs_type, ns="", algo="dqn"):
    """Load config.json for a training run to replicate ablation parameters."""
    algo_dir = "" if algo == "dqn" else "ppo"
    paths_to_check = []
    if ns:
        if algo_dir:
            paths_to_check.append(os.path.join("Results", "tmp", location, obs_type, algo_dir, ns, "config.json"))
        paths_to_check.append(os.path.join("Results", "tmp", location, obs_type, ns, "config.json"))
    if algo_dir:
        paths_to_check.append(os.path.join("Results", "tmp", location, obs_type, algo_dir, "config.json"))
    paths_to_check.append(os.path.join("Results", "tmp", location, obs_type, "config.json"))

    for p in paths_to_check:
        if os.path.exists(p):
            with open(p, "r") as f:
                return json.load(f), p
    return {}, None


# ── Main comparison ────────────────────────────────────────────────────


def compare(location="DongDa", obs_type="mlp_graph", ns="config_2", step=None, max_steps=None,
            eval_grid_penalty_weight=None,
            ga_ns="",
            algo="dqn",
            ppo_ns="",
            ppo_step=None,
            device="cpu"):
    # Base directory and paths
    base_dir = os.path.join(project_root, "custom_environment", "data")
    graph_file = os.path.join(base_dir, "Graph", location, f"{location}.graphml")
    node_file = os.path.join(base_dir, "Graph", location, f"nodes_extended_{location}.txt")
    plan_file = os.path.join(base_dir, "Graph", location, f"new_existingplan_{location}.pkl")
    if not os.path.exists(plan_file):
        plan_file = os.path.join(base_dir, "Graph", location, f"existingplan_{location}.pkl")

    # Adopt the reward parameters the RL run was trained with, before the env is
    # built: reset() already scores the plan and builds the first observation.
    primary_algo = "ppo" if algo == "ppo" else "dqn"
    primary_ns = ppo_ns if (algo == "ppo" and ppo_ns) else ns
    run_cfg, cfg_path = load_config_for_run(location, obs_type, primary_ns, primary_algo)
    if cfg_path:
        print(f"Loaded config from {cfg_path}")
    else:
        print(f"Warning: config.json not found for {location}/{obs_type}; running with module defaults.")

    for key, attr in (("grid_penalty_weight", "GRID_PENALTY_WEIGHT"),
                      ("eta", "DEMAND_ETA"), ("beta", "DEMAND_BETA")):
        if run_cfg.get(key) is not None:
            setattr(H, attr, float(run_cfg[key]))
    print(f"[EPISODE] grid_penalty_weight={H.GRID_PENALTY_WEIGHT}, "
          f"eta={H.DEMAND_ETA}, beta={H.DEMAND_BETA}")
    if eval_grid_penalty_weight is not None:
        print(f"[SCORING] plans rescored at grid_penalty_weight={eval_grid_penalty_weight}")

    # Env for testing
    env = StationPlacement(graph_file, node_file, plan_file, location=location, obs_type=obs_type)

    # 0. Baseline calculation (as in run_metrics.py)
    with open(node_file, "r") as file:
        baseline_node_list = eval(file.readline())
        
    with open(plan_file, "rb") as f:
        baseline_plan = pickle.load(f)
        
    baseline_node_list, baseline_plan = prepare_existing_plan(baseline_plan, baseline_node_list, env.graph)
    
    (
        b_norm_benefit,
        b_norm_cost,
        b_norm_charging,
        b_norm_waiting,
        b_norm_travel,
        b_norm_fairness
    ) = H.existing_score(baseline_plan, baseline_node_list)
    
    baseline_grid_penalty = None
    b_dist_penalty = 0.0
    b_cap_penalty = 0.0
    baseline_violations = []
    if env.grid_adapter:
        station_nodes = [(s[0], s[2]["capability"]) for s in baseline_plan]
        b_dist_penalty, b_cap_penalty, _, _ = env.grid_adapter.calculate_grid_penalty(station_nodes)
        baseline_grid_penalty = {'dist_penalty': b_dist_penalty, 'cap_penalty': b_cap_penalty}
        baseline_violations = env.grid_adapter.get_grid_violations(station_nodes)
    
    norm_score_baseline = eci_test(
        baseline_plan,
        baseline_node_list,
        b_norm_benefit,
        b_norm_charging,
        b_norm_waiting,
        b_norm_travel,
        baseline_grid_penalty,
    )
    print(f"Baseline (Existing Plan) Norm Score: {norm_score_baseline:.6f}")
    if env.grid_adapter:
        print(f"  Baseline Grid Penalties:")
        print(f"    Distance Penalty: {b_dist_penalty:.4f}")
        print(f"    Capacity Penalty: {b_cap_penalty:.4f}")
        if baseline_violations:
            print(f"    Violating Buses:")
            for violation in baseline_violations:
                print(f"      - Bus {violation['bus_name']} (Index {violation['bus_idx']}):")
                print(f"        Voltage: {violation['voltage_kv']:.1f} kV")
                print(f"        Required: {violation['required_mw']:.3f} MW")
                print(f"        Available: {violation['available_mw']:.3f} MW")
                print(f"        Shortage: {violation['shortage_mw']:.3f} MW")
        print()

    # Load graph for visualization
    G = ox.load_graphml(graph_file)

    # 1. Load RL Models (DQN and/or PPO)
    print()
    rl_log_dir = os.path.join("Results", "tmp", location, obs_type)
    rl_agents = {}  # name -> (agent, step)

    run_dqn = algo in ["dqn", "both", "all"]
    run_ppo = algo in ["ppo", "both", "all"]

    if run_dqn:
        best_dqn_model, dqn_resolved_step = resolve_rl_model(rl_log_dir, location, obs_type, ns, step, algo="dqn")
        if best_dqn_model:
            label = "DQN" if (run_ppo or algo == "dqn") else "RL"
            print(f"Loading DQN model from {best_dqn_model} on device {device}")
            agent = DQN.load(best_dqn_model, env=env, device=device, custom_objects=extractor_custom_objects(obs_type))
            rl_agents[label] = (agent, dqn_resolved_step)
        else:
            print("Warning: DQN model not found.")

    if run_ppo:
        resolved_ppo_ns = ppo_ns if ppo_ns else (ns if algo == "ppo" else "")
        resolved_ppo_step = ppo_step if ppo_step is not None else (step if algo == "ppo" else None)
        best_ppo_model, ppo_resolved_step = resolve_rl_model(rl_log_dir, location, obs_type, resolved_ppo_ns, resolved_ppo_step, algo="ppo")
        if best_ppo_model:
            label = "PPO"
            print(f"Loading PPO model from {best_ppo_model} on device {device}")
            agent = PPO.load(best_ppo_model, env=env, device=device, custom_objects=extractor_custom_objects(obs_type))
            rl_agents[label] = (agent, ppo_resolved_step)
        else:
            print("Warning: PPO model not found.")

    # 2. Load GA Model (Only compatible with 'mlp')
    ga_agent = None
    if obs_type == "mlp":
        ga_dir = os.path.join("Results", "ga", location)
        if ga_ns:
            ga_dir = os.path.join(ga_dir, ga_ns)
        ga_model_path = os.path.join(ga_dir, f"best_ga_model_{location}.pt")
        if os.path.exists(ga_model_path):
            print(f"Loading GA model from {ga_model_path}")
            chromosome = torch.load(ga_model_path, weights_only=False)
            input_dim = env.observation_space.shape[0]
            output_dim = env.action_space.n
            ga_agent = GAPolicy(input_dim, output_dim, hidden_dim=256)
            unflatten_weights(ga_agent, chromosome)
        else:
            print("Warning: GA model not found.")
    else:
        print(f"Note: GA comparison skipped (Incompatible with {obs_type} observation format).")

    # Run comparisons
    results = {}
    plans = {}
    node_lists = {}
    step_map = {}

    for label, (agent, agent_step) in rl_agents.items():
        print(f"Running {label} evaluation...")
        metrics, plan, node_list = run_episode(agent, env, "rl", max_steps=max_steps, eval_grid_penalty_weight=eval_grid_penalty_weight)
        results[label] = metrics
        plans[label] = plan
        node_lists[label] = node_list
        step_map[label] = agent_step

    if ga_agent:
        print("Running GA evaluation...")
        metrics, plan, node_list = run_episode(ga_agent, env, "ga", max_steps=max_steps, eval_grid_penalty_weight=eval_grid_penalty_weight)
        results["GA"] = metrics
        plans["GA"] = plan
        node_lists["GA"] = node_list

    print("Running Greedy Benefit evaluation...")
    metrics, plan, node_list = run_episode(None, env, "greedy_benefit", max_steps=max_steps, eval_grid_penalty_weight=eval_grid_penalty_weight)
    results["G-Benefit"] = metrics
    plans["G-Benefit"] = plan
    node_lists["G-Benefit"] = node_list

    print("Running Greedy Demand evaluation...")
    metrics, plan, node_list = run_episode(None, env, "greedy_demand", max_steps=max_steps, eval_grid_penalty_weight=eval_grid_penalty_weight)
    results["G-Demand"] = metrics
    plans["G-Demand"] = plan
    node_lists["G-Demand"] = node_list

    # Display results
    print("\n--- Comparison Results ---")
    for alg, m in results.items():
        rel_score = m["score"] / (norm_score_baseline + 1e-9)

        print(f"{alg}:")
        print(f"  Score (raw): {m['score']:.2f}")
        print(f"  Score (relative to baseline): {rel_score:.2f}%")
        print(f"  Benefit: {m['benefit']:.2f}%")
        print(f"  Fairness: {m['fairness']:.2f}%")

        print(
            f"  Waiting time: {m['wait_time']:.2f}%, "
            f"Travel time: {m['cost_travel']:.2f}%, "
            f"Charging time: {m['charg_time']:.2f}%"
        )
        print(f"  Max Travel time: {m['travel_max']:.2f} min, Max Waiting time: {m['wait_max']:.2f} min")
        print(f"  Used budget: {m['used_budget']:.2f}%")
        print(f"  Nodes covered (Stations): {m['num_stations']}")
        if env.grid_adapter:
            print(f"  Grid Penalties:")
            print(f"    Distance Penalty: {m.get('dist_penalty', 0.0):.4f}")
            print(f"    Capacity Penalty: {m.get('cap_penalty', 0.0):.4f}")
            violations = m.get('grid_violations', [])
            if violations:
                print(f"    Violating Buses:")
                for violation in violations:
                    print(f"      - Bus {violation['bus_name']} (Index {violation['bus_idx']}):")
                    print(f"        Voltage: {violation['voltage_kv']:.1f} kV")
                    print(f"        Required: {violation['required_mw']:.3f} MW")
                    print(f"        Available: {violation['available_mw']:.3f} MW")
                    print(f"        Shortage: {violation['shortage_mw']:.3f} MW")
        print(f"  Total Reward: {m['reward']:.2f}\n")

    # ── Bar/Line chart ──
    if results:
        algorithms = list(results.keys())
        rewards = [m["reward"] for m in results.values()]
        scores = [m["score"] for m in results.values()]

        fig, ax1 = plt.subplots(figsize=(10, 6))

        color = "tab:blue"
        ax1.set_xlabel("Algorithm")
        ax1.set_ylabel("Total Reward", color=color)
        ax1.bar(algorithms, rewards, color=color, alpha=0.6, label="Reward")
        ax1.tick_params(axis="y", labelcolor=color)

        ax2 = ax1.twinx()
        color = "tab:red"
        ax2.set_ylabel("Final Score", color=color)
        ax2.plot(
            algorithms, scores, color=color, marker="o", label="Score", linewidth=2, markersize=8
        )
        ax2.tick_params(axis="y", labelcolor=color)

        plt.title(f"Algorithm Performance Comparison ({location})")
        fig.tight_layout()
        chart_path = os.path.join("Results", f"comparison_{location}.png")
        plt.savefig(chart_path, dpi=150)
        plt.close(fig)
        print(f"Comparison chart saved to {chart_path}")

    # ── Station maps ──
    print("\n--- Generating Station Maps ---")
    map_dir = os.path.join("Results", "maps", location)
    for alg_name, plan in plans.items():
        safe_name = alg_name.replace("-", "_").replace(" ", "_").replace("(", "").replace(")", "")
        filepath = os.path.join(map_dir, f"map_{safe_name}_{location}.png")
        m = results[alg_name]
        title = f"{alg_name} — Score: {m['score']:.2f}% — Stations: {m['num_stations']}"
        visualise_stations(G, plan, filepath, title=title)

    print(f"\nAll maps saved to {map_dir}{os.sep}")

    # ── Save best plans (mirrors evaluate.py's Results/optimal_plan output) ──
    print("\n--- Saving Optimal Plans ---")
    plan_dir = os.path.join("Results", "optimal_plan", location)
    os.makedirs(plan_dir, exist_ok=True)
    for alg_name, plan in plans.items():
        safe_name = alg_name.replace("-", "_").replace(" ", "_").replace("(", "").replace(")", "")
        cur_step = step_map.get(alg_name)
        suffix = f"_{cur_step}" if cur_step is not None else ""

        plan_path = os.path.join(plan_dir, f"plan_{safe_name}{suffix}.pkl")
        with open(plan_path, "wb") as f:
            pickle.dump(plan, f)

        nodes_path = os.path.join(plan_dir, f"nodes_{safe_name}{suffix}.txt")
        with open(nodes_path, "w") as f:
            f.write(str(node_lists[alg_name]))

        print(f"  {alg_name}: saved {plan_path} and {nodes_path}")

    print(f"\nAll optimal plans saved to {plan_dir}{os.sep}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare RL policies (DQN/PPO) against GA and greedy baselines on one district.")
    parser.add_argument("--location", type=str, default="DongDa")
    parser.add_argument("--algo", type=str, choices=["dqn", "ppo", "both", "all"], default="dqn",
                        help="RL algorithm to evaluate: 'dqn', 'ppo', or 'both'/'all' (default: dqn)")
    parser.add_argument("--obs_type", type=str, default="mlp",
                        choices=["mlp", "mlp_graph", "gnn", "attention"],
                        help="Observation type; must match the RL run being loaded")
    parser.add_argument("--ns", type=str, default="config_2",
                        help="Run namespace under Results/tmp/<location>/<obs_type>/")
    parser.add_argument("--step", type=int, default=None,
                        help="Checkpoint step; default picks the best-scoring one from evaluation_results.csv")
    parser.add_argument("--ppo_ns", type=str, default="",
                        help="Namespace for PPO run (if different from --ns or when --algo both)")
    parser.add_argument("--ppo_step", type=int, default=None,
                        help="Checkpoint step for PPO (if different from --step or when --algo both)")
    parser.add_argument("--max_steps", type=int, default=None,
                        help="Cap episode length for a quick behaviour probe on the large districts. "
                             "Scores from a capped run are NOT comparable to a full one")
    parser.add_argument("--ga_ns", type=str, default="",
                        help="Namespace of the GA run to load, matching train_ga.py --ns. "
                             "Omit for the flat Results/ga/<location>/ layout")
    parser.add_argument("--eval_grid_penalty_weight", type=float, default=None,
                        help="Rescore the resulting plans at this grid penalty weight. "
                             "Pass 1.0 for an ablation trained with --grid_penalty_weight 0, "
                             "so it acts as trained but is reported on the full metric. "
                             "Omit to score with the weight the run was trained at")
    parser.add_argument("--device", type=str, default="cpu", choices=["auto", "cpu", "cuda"],
                         help="Device to load model on (default: cpu)")

    args = parser.parse_args()
    compare(location=args.location, obs_type=args.obs_type, ns=args.ns,
            step=args.step, max_steps=args.max_steps, ga_ns=args.ga_ns,
            eval_grid_penalty_weight=args.eval_grid_penalty_weight,
            algo=args.algo, ppo_ns=args.ppo_ns, ppo_step=args.ppo_step,
            device=args.device)



