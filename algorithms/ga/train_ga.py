import os
import sys
import torch
import numpy as np
import random
import argparse
import time
import csv
import json
import copy

# Add project root to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.append(project_root)

# Now we can import from project root
from custom_environment.StationPlacementEnv import StationPlacement
from algorithms.ga.ga_utils import GAPolicy, flatten_weights, unflatten_weights, crossover, mutate

def evaluate_agent(policy, env, seed):
    obs, _ = env.reset(seed=seed)
    total_reward = 0
    terminated = False
    truncated = False
    while not (terminated or truncated):
        action = policy.select_action(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
    return total_reward, env.best_score

def train_ga(args):
    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Instantiate the env (GA uses flat MLP observations only)
    location = args.location
    base_dir = os.path.join(project_root, "custom_environment", "data")
    graph_file = os.path.join(base_dir, "Graph", location, location + ".graphml")
    node_file = os.path.join(base_dir, "Graph", location, "nodes_extended_" + location + ".txt")
    plan_file = os.path.join(base_dir, "Graph", location, "new_existingplan_" + location + ".pkl")

    env = StationPlacement(graph_file, node_file, plan_file, location=location, obs_type="mlp")
    
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n
    hidden_dim = args.hidden_dim

    # Population initialization
    population = []
    for _ in range(args.pop_size):
        model = GAPolicy(input_dim, output_dim, hidden_dim)
        population.append(flatten_weights(model))

    best_overall_score = -np.inf
    best_overall_chromosome = None

    # Use absolute path for results directory. --ns namespaces the run the same way
    # train.py does, so several seeds on one district no longer overwrite each
    # other's config.json, training_history.csv and best_ga_model_*.pt.
    log_dir = os.path.join(project_root, "Results", "ga", location)
    if args.ns:
        log_dir = os.path.join(log_dir, args.ns)
    os.makedirs(log_dir, exist_ok=True)
    print(f"GA results -> {log_dir}")

    # Save training config
    config_data = vars(args).copy()
    config_data["input_dim"] = int(input_dim)
    config_data["output_dim"] = int(output_dim)
    with open(os.path.join(log_dir, "config.json"), "w") as f:
        json.dump(config_data, f, indent=2)
    print(f"Config saved to {os.path.join(log_dir, 'config.json')}")

    # Training history tracking
    history = []

    start_time = time.time()

    for gen in range(args.generations):
        gen_start = time.time()
        fitness_rewards = []
        fitness_scores = []
        
        # Evaluation step
        for i, chromosome in enumerate(population):
            print(f"  Gen {gen} | Evaluating agent {i+1}/{args.pop_size}...", end="\r")
            model = GAPolicy(input_dim, output_dim, hidden_dim)
            unflatten_weights(model, chromosome)
            # Use fixed seed for evaluation to reduce variance within generation
            reward, final_score = evaluate_agent(model, env, args.seed + gen)
            fitness_rewards.append(reward)
            fitness_scores.append(final_score)
        print() # New line after generation evaluation

        fitness_rewards = np.array(fitness_rewards)
        fitness_scores = np.array(fitness_scores)
        
        # Sort by score (descending)
        idx = np.argsort(fitness_scores)[::-1]
        population = [population[i] for i in idx]
        
        current_best_reward = fitness_rewards[idx[0]]
        current_best_score = fitness_scores[idx[0]]
        gen_elapsed = time.time() - gen_start

        if current_best_score > best_overall_score:
            best_overall_score = current_best_score
            best_overall_chromosome = population[0].copy()
            # Save the best model
            torch.save(best_overall_chromosome, os.path.join(log_dir, f"best_ga_model_{location}.pt"))
            print(f"Gen {gen}: * New best score: {best_overall_score:.4f} (saved)")

        print(f"Gen {gen}: Best Score: {current_best_score:.4f}, Best Reward: {current_best_reward:.2f}, "
              f"Avg Score: {np.mean(fitness_scores):.4f}, Avg Reward: {np.mean(fitness_rewards):.2f} "
              f"({gen_elapsed:.1f}s)")

        # Record history
        history.append({
            "generation": gen,
            "best_score": float(current_best_score),
            "best_reward": float(current_best_reward),
            "avg_score": float(np.mean(fitness_scores)),
            "avg_reward": float(np.mean(fitness_rewards)),
            "best_overall_score": float(best_overall_score),
            "elapsed_sec": gen_elapsed,
        })

        # Selection (Elitism) — deepcopy to prevent mutation from corrupting elites
        new_population = [chrom.copy() for chrom in population[:args.elitism]]
        
        # Produce offspring
        while len(new_population) < args.pop_size:
            p1, p2 = random.sample(population[:args.pop_size // 2], 2)
            c1, c2 = crossover(p1, p2, rate=0.5)
            new_population.append(mutate(c1, rate=args.mutation_rate, sigma=args.mutation_sigma))
            if len(new_population) < args.pop_size:
                new_population.append(mutate(c2, rate=args.mutation_rate, sigma=args.mutation_sigma))
        
        population = new_population

    total_elapsed = time.time() - start_time
    print(f"\nGA Training finished. Best global score: {best_overall_score:.4f}")
    print(f"Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f}min)")

    # Save training history to CSV
    csv_path = os.path.join(log_dir, "training_history.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    print(f"Training history saved to {csv_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a GA agent for station placement.")
    parser.add_argument("--location", type=str, default="DongDa")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--ns", type=str, default="",
                        help="Namespace for this run, e.g. ga_s1. Results go to "
                             "Results/ga/<location>/<ns>/; omit for the legacy flat layout")
    parser.add_argument("--pop_size", type=int, default=50)
    parser.add_argument("--generations", type=int, default=100)
    parser.add_argument("--elitism", type=int, default=5)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--mutation_rate", type=float, default=0.05)
    parser.add_argument("--mutation_sigma", type=float, default=0.1)
    
    args = parser.parse_args()
    train_ga(args)
