"""
Verification for the hard grid constraint.

The claim being tested is narrow and absolute: in grid_mode="hard", NO bus is
ever over its capacity, at ANY step, under ANY action sequence -- including
adversarial ones that do nothing but build. A random policy is the right test
because the guarantee must not depend on the policy: if it held only for a
trained agent it would be a tendency, not a constraint.

It also reports what the constraint costs, which is the number the paper needs:
"hard" should reach a comparable station count and score to "off", NOT collapse
to the near-empty plan that the soft penalty produces.

Usage:
    python tests/test_grid_constraint.py --location DongDa --episodes 3
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import custom_environment.helpers as H
from custom_environment.StationPlacementEnv import StationPlacement


def build_env(location, grid_mode, base_dir):
    graph_file = os.path.join(base_dir, "Graph", location, location + ".graphml")
    node_file = os.path.join(base_dir, "Graph", location, "nodes_extended_" + location + ".txt")
    plan_file = os.path.join(base_dir, "Graph", location, "new_existingplan_" + location + ".pkl")
    if not os.path.exists(plan_file):
        plan_file = os.path.join(base_dir, "Graph", location, "existingplan_" + location + ".pkl")
    env = StationPlacement(graph_file, node_file, plan_file,
                           location=location, obs_type="mlp", grid_mode=grid_mode)
    env.verify_grid = (grid_mode == "hard")
    return env


def run_episodes(env, episodes, seed, policy="random", max_steps=None):
    """Roll out and return per-episode summaries plus the worst violation seen at
    ANY step (not just at the end -- an intermediate overload is still a violation,
    and only a per-step check can catch one that a later relocation hides)."""
    rng = np.random.default_rng(seed)
    out = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        assert env.observation_space.contains(obs.astype(np.float32)), \
            "reset() returned an observation outside the declared space"
        done = False
        steps = 0
        worst_util = 0.0
        max_overload = 0.0
        while not done:
            if policy == "random":
                action = int(rng.integers(env.action_space.n))
            else:  # "build": only the station-building actions, the adversarial case
                action = int(rng.choice([0, 1, 5] if env.action_space.n > 5 else [0, 1]))
            obs, reward, done, truncated, _ = env.step(action)
            steps += 1
            if env.feasibility:
                worst_util = max(worst_util, env.feasibility.worst_utilization())
                for v in env.feasibility.violated_buses():
                    max_overload = max(max_overload, v["overload_mw"])
            if max_steps and steps >= max_steps:
                break
        assert env.observation_space.contains(obs.astype(np.float32)), \
            "step() returned an observation outside the declared space"
        out.append({
            "steps": steps,
            "stations": len(env.plan_instance.plan),
            "score": env.best_score,
            "worst_util": worst_util,
            "max_overload_mw": max_overload,
            "blocked": env.feasibility.blocked_actions if env.feasibility else 0,
            "redirected": env.feasibility.redirected_actions if env.feasibility else 0,
        })
    return out


def summarize(name, rows):
    def m(k):
        return float(np.mean([r[k] for r in rows]))
    print(f"{name:>22} | stations {m('stations'):6.1f} | score {m('score'):7.4f} | "
          f"worst bus util {m('worst_util'):6.3f} | max overload {m('max_overload_mw'):7.3f} MW | "
          f"blocked {m('blocked'):5.1f} | redirected {m('redirected'):5.1f}")
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--location", default="DongDa")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_steps", type=int, default=None,
                    help="cut episodes short to keep the check quick")
    args = ap.parse_args()

    base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "custom_environment", "data")

    results = {}
    for mode in ("off", "soft", "hard"):
        env = build_env(args.location, mode, base_dir)
        rows = run_episodes(env, args.episodes, args.seed, "random", args.max_steps)
        results[mode] = rows

    print("\n" + "=" * 118)
    print(f"RANDOM POLICY, {args.episodes} episodes, {args.location}")
    print("=" * 118)
    for mode in ("off", "soft", "hard"):
        summarize(f"grid_mode={mode}", results[mode])

    # Adversarial: build-only actions, the sequence most likely to overload a bus.
    env = build_env(args.location, "hard", base_dir)
    build_rows = run_episodes(env, args.episodes, args.seed, "build", args.max_steps)
    print("-" * 118)
    summarize("hard / build-only", build_rows)
    print("=" * 118)

    failures = []
    for rows, label in ((results["hard"], "hard/random"), (build_rows, "hard/build-only")):
        for i, r in enumerate(rows):
            if r["max_overload_mw"] > 1e-9:
                failures.append(f"{label} ep{i}: {r['max_overload_mw']:.4f} MW overload")
            if r["worst_util"] > 1.0 + 1e-9:
                failures.append(f"{label} ep{i}: bus utilization {r['worst_util']:.4f} > 1.0")

    if failures:
        print("\nFAIL -- the hard constraint was broken:")
        for f in failures:
            print("  " + f)
        sys.exit(1)

    soft_st = np.mean([r["stations"] for r in results["soft"]])
    off_st = np.mean([r["stations"] for r in results["off"]])
    hard_st = np.mean([r["stations"] for r in results["hard"]])
    print("\nPASS -- no bus exceeded its capacity at any step, in any hard-mode episode.")
    print(f"Cost of the constraint: {hard_st:.1f} stations vs {off_st:.1f} unconstrained "
          f"({100 * hard_st / max(off_st, 1e-9):.0f}% retained).")
    print(f"For contrast, the soft penalty reaches {soft_st:.1f} stations -- note that a "
          f"LOW number there is the failure mode, not a success.")


if __name__ == "__main__":
    main()
