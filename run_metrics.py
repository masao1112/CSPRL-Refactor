import argparse
import os
import pickle
from math import ceil
import numpy as np
import osmnx as ox

import custom_environment.helpers as H
from custom_environment.power_grid.csprl_adapter import create_adapter_for_location


def travel_metric(my_node_list):
    """Max travel time in minutes."""
    big_travel_list = []
    for my_node in my_node_list:
        travel = my_node[1]["distance"] / H.VELOCITY * 60
        times = ceil(10 * H.weak_demand(my_node))
        for _ in range(times):
            big_travel_list.append(travel)
    return max(big_travel_list) if big_travel_list else 0.0


def waiting_metric(my_plan):
    """Max waiting time in minutes."""
    big_waiting_list = []
    for my_station in my_plan:
        times = ceil(my_station[2]["D_s"])
        for _ in range(times):
            big_waiting_list.append(my_station[2]["W_s"] * 60)
    return max(big_waiting_list) if big_waiting_list else 0.0


def prepare_existing_plan(my_plan, my_node_list, graph):
    my_cost_dict = {}
    my_node_dict = {}
    for my_node in my_node_list:
        my_node_dict[my_node[0]] = {}
        my_node[1]["charging station"] = None
        my_node[1]["distance"] = None

    for index in range(len(my_plan)):
        my_plan[index] = H.s_dictionnary(my_plan[index], my_node_list)
    my_node_list, _, _ = H.station_seeking(my_plan, my_node_list, my_node_dict, my_cost_dict, graph)
    for index in range(len(my_plan)):
        my_plan[index] = H.s_dictionnary(my_plan[index], my_node_list)
    return my_node_list, my_plan


def evaluate_plan(my_plan, my_node_list, my_basic_cost, norm_constants, grid_adapter=None):
    norm_benefit, norm_cost, norm_charg, norm_wait, norm_travel, norm_fairness, norm_score_val = norm_constants

    dist_penalty = 0.0
    cap_penalty = 0.0
    grid_utilization = 0.0
    grid_distance = 0.0
    grid_penalty_tuple = None

    if grid_adapter is not None:
        try:
            station_nodes = [(s[0], s[2]["capability"]) for s in my_plan]
            dist_penalty, cap_penalty, grid_utilization, grid_distance = grid_adapter.calculate_grid_penalty(station_nodes)
            grid_penalty_tuple = (dist_penalty, cap_penalty)
        except Exception as e:
            print(f"[GRID] Warning calculating grid penalty: {e}")

    score, benefit, cost, charg_time, wait_time, cost_travel, fairness = H.norm_score(
        my_plan, my_node_list, norm_benefit, norm_charg, norm_wait, norm_travel, grid_penalty=grid_penalty_tuple
    )

    travel_max = travel_metric(my_node_list)
    wait_max = waiting_metric(my_plan)

    total_inst_cost = (sum([station[2]["fee"] for station in my_plan]) - my_basic_cost) / H.BUDGET
    rel_score = (score / (norm_score_val + 1e-9)) * 100.0 if norm_score_val != 0 else score * 100.0

    return {
        "score_raw": score * 100.0,
        "score_rel": rel_score,
        "benefit": benefit * 100.0,
        "cost": cost * 100.0,
        "fairness": fairness * 100.0,
        "wait_time": wait_time * 100.0,
        "cost_travel": cost_travel * 100.0,
        "charg_time": charg_time * 100.0,
        "travel_max": travel_max,
        "wait_max": wait_max,
        "used_budget": total_inst_cost * 100.0,
        "num_stations": len(my_plan),
        "dist_penalty": abs(dist_penalty),
        "cap_penalty": abs(cap_penalty),
        "grid_utilization": grid_utilization,
        "grid_distance": grid_distance,
    }


def print_metrics(name, m):
    print(f"\n================ {name} ================")
    print(f"  Score (relative to baseline): {m['score_rel']:.2f}%")
    print(f"  Score (raw):                 {m['score_raw']:.2f}%")
    print(f"  Social Benefit:              {m['benefit']:.2f}%")
    print(f"  Social Cost:                 {m['cost']:.2f}%")
    print(f"  Social Fairness:             {m['fairness']:.2f}%")
    print(f"  Waiting Time:                {m['wait_time']:.2f}%")
    print(f"  Travel Time:                 {m['cost_travel']:.2f}%")
    print(f"  Charging Time:               {m['charg_time']:.2f}%")
    print(f"  Max Travel Time:             {m['travel_max']:.2f} min")
    print(f"  Max Waiting Time:            {m['wait_max']:.2f} min")
    print(f"  Used Budget:                 {m['used_budget']:.2f}%")
    print(f"  Number of Stations:          {m['num_stations']}")
    if m["dist_penalty"] > 0 or m["cap_penalty"] > 0:
        print(f"  Grid Dist Penalty:           {m['dist_penalty']:.4f}")
        print(f"  Grid Cap Penalty:            {m['cap_penalty']:.4f}")
        print(f"  Grid Avg Utilization:        {m['grid_utilization']:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate EV charging placement plan with benchmark metrics.")
    parser.add_argument("--location", type=str, default="DongDa", help="District location name")
    parser.add_argument("--step", type=int, default=61608, help="Checkpoint step for RL plan")
    parser.add_argument("--plan_file", type=str, default=None, help="Path to custom plan .pkl file")
    parser.add_argument("--node_file", type=str, default=None, help="Path to custom nodes .txt file")
    parser.add_argument("--grid_penalty_weight", type=float, default=1.0, help="Weight for grid penalties in score")
    args = parser.parse_args()

    location = args.location
    H.GRID_PENALTY_WEIGHT = args.grid_penalty_weight

    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "custom_environment", "data")
    graph_file = os.path.join(base_dir, "Graph", location, f"{location}.graphml")
    base_node_file = os.path.join(base_dir, "Graph", location, f"nodes_extended_{location}.txt")
    if not os.path.exists(base_node_file):
        alt = os.path.join(base_dir, "Graph", location, f"og_nodes_extended_{location}.txt")
        if os.path.exists(alt):
            base_node_file = alt

    existing_plan_file = os.path.join(base_dir, "Graph", location, f"new_existingplan_{location}.pkl")
    if not os.path.exists(existing_plan_file):
        alt = os.path.join(base_dir, "Graph", location, f"existingplan_{location}.pkl")
        if os.path.exists(alt):
            existing_plan_file = alt

    # Load graph and baseline existing plan
    print(f"Loading graph: {graph_file}")
    graph = ox.load_graphml(graph_file)

    with open(base_node_file, "r") as f:
        baseline_node_list = eval(f.readline(), {"np": np, "numpy": np})
    with open(existing_plan_file, "rb") as f:
        baseline_plan = pickle.load(f)

    # Load power grid adapter and set district scope
    grid_adapter = None
    try:
        grid_adapter = create_adapter_for_location(location)
        if grid_adapter is not None:
            bus_indices = [
                grid_adapter._get_bus_info(attrs.get('y', 0), attrs.get('x', 0)).get('bus_idx', -1)
                for _, attrs in baseline_node_list
            ]
            grid_adapter.set_district_scope(bus_indices)
    except Exception as e:
        print(f"[GRID] Note: Grid adapter not available or skipped for {location}: {e}")

    baseline_node_list, baseline_plan = prepare_existing_plan(baseline_plan, baseline_node_list, graph)
    basic_cost = sum([station[2]["fee"] for station in baseline_plan])

    norm_benefit, norm_cost, norm_charg, norm_wait, norm_travel, norm_fairness = H.existing_score(
        baseline_plan, baseline_node_list
    )

    # Compute baseline score
    baseline_dist_p, baseline_cap_p = 0.0, 0.0
    baseline_penalty_tuple = None
    if grid_adapter is not None:
        station_nodes = [(s[0], s[2]["capability"]) for s in baseline_plan]
        baseline_dist_p, baseline_cap_p, _, _ = grid_adapter.calculate_grid_penalty(station_nodes)
        baseline_penalty_tuple = (baseline_dist_p, baseline_cap_p)

    norm_score_val, b_ben, b_cost, b_charg, b_wait, b_travel, b_fair = H.norm_score(
        baseline_plan, baseline_node_list, norm_benefit, norm_charg, norm_wait, norm_travel,
        grid_penalty=baseline_penalty_tuple
    )

    norm_constants = (norm_benefit, norm_cost, norm_charg, norm_wait, norm_travel, norm_fairness, norm_score_val)

    # Evaluate baseline existing plan
    baseline_metrics = evaluate_plan(baseline_plan, baseline_node_list, basic_cost, norm_constants, grid_adapter)
    print_metrics(f"Baseline Existing Plan ({location})", baseline_metrics)

    # Determine RL plan and node files
    plan_file = args.plan_file
    node_file = args.node_file
    if plan_file is None:
        plan_file = os.path.join("Results", "optimal_plan", location, f"plan_RL_{args.step}.pkl")
    if node_file is None:
        node_file = os.path.join("Results", "optimal_plan", location, f"nodes_RL_{args.step}.txt")

    if not os.path.exists(plan_file):
        print(f"\n[ERROR] Plan file not found: {plan_file}")
        return
    if not os.path.exists(node_file):
        print(f"\n[ERROR] Node file not found: {node_file}")
        return

    print(f"\nEvaluating Plan from: {plan_file}")
    with open(plan_file, "rb") as f:
        rl_plan = pickle.load(f)
    with open(node_file, "r") as f:
        rl_node_list = eval(f.readline(), {"np": np, "numpy": np})

    # Recompute station dictionary parameters under current branch rules (M/M/1/N queuing model)
    for index in range(len(rl_plan)):
        rl_plan[index] = H.s_dictionnary(rl_plan[index], rl_node_list)

    rl_metrics = evaluate_plan(rl_plan, rl_node_list, basic_cost, norm_constants, grid_adapter)
    print_metrics(f"RL Plan (Step {args.step})", rl_metrics)


if __name__ == "__main__":
    main()
