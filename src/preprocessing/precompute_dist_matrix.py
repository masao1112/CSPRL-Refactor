"""
9.1 - Precompute all-pairs shortest-path distance matrix.

Run ONCE before training to generate the lookup file:
    python src/preprocessing/precompute_dist_matrix.py

The output is saved to:
    custom_environment/data/Graph/<location>/dist_matrix_<location>.pkl

After this file exists, helpers.calculate_distance() uses O(1) dict lookups
instead of running Dijkstra on every call (~10-50x speed-up).
"""
import os
import pickle
import sys

import networkx as nx
import osmnx as ox

# Ensure project root is on path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.append(project_root)


def precompute(location: str = "DongDa") -> None:
    data_dir  = os.path.join(project_root, "custom_environment", "data", "Graph", location)
    graph_file = os.path.join(data_dir, f"{location}.graphml")
    out_file   = os.path.join(data_dir, f"dist_matrix_{location}.pkl")

    if not os.path.exists(graph_file):
        print(f"ERROR: Graph file not found: {graph_file}")
        sys.exit(1)

    if os.path.exists(out_file):
        print(f"Distance matrix already exists at {out_file}. Delete it to recompute.")
        return

    print(f"Loading graph from {graph_file} ...")
    G = ox.load_graphml(graph_file)
    n_nodes = G.number_of_nodes()
    print(f"Graph has {n_nodes} nodes and {G.number_of_edges()} edges.")
    print("Computing all-pairs shortest paths (this may take a few minutes) ...")

    dist_matrix = dict(nx.all_pairs_dijkstra_path_length(G, weight='length'))

    print(f"Saving distance matrix to {out_file} ...")
    with open(out_file, "wb") as f:
        pickle.dump(dist_matrix, f, protocol=pickle.HIGHEST_PROTOCOL)

    size_mb = os.path.getsize(out_file) / (1024 ** 2)
    print(f"Done. File size: {size_mb:.1f} MB")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Precompute distance matrix for a location.")
    parser.add_argument("--location", type=str, default="DongDa")
    args = parser.parse_args()
    precompute(args.location)
