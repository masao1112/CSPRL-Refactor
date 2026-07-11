import numpy as np


class GraphFeatureAugmentor:
    """
    Precomputes graph-derived features to augment MLP observations.

    Uses the graph adjacency to produce neighborhood-aggregated features
    without requiring a GNN at training time. This gives the MLP policy
    access to relational/spatial information that would otherwise require
    message-passing layers.

    Features produced:
      - Per-node: 1-hop neighborhood mean of all node features (via row-normalized adjacency)
      - Graph-level: 5 summary statistics capturing coverage, demand distribution,
        station utilization, spatial mismatch, and station density
    """

    N_GRAPH_SUMMARIES = 5

    def __init__(self, node_list, graph, node_id_to_idx):
        self.n_nodes = len(node_list)
        self.node_id_to_idx = node_id_to_idx

        # Build weighted adjacency matrix from the graph
        adj = np.zeros((self.n_nodes, self.n_nodes), dtype=np.float32)
        for u, v, data in graph.edges(data=True):
            if u in node_id_to_idx and v in node_id_to_idx:
                i, j = node_id_to_idx[u], node_id_to_idx[v]
                length = data.get('length', 10.0)
                # Distance-decay weighting: closer nodes get higher weight
                weight = np.exp(-length / 1000.0)
                adj[i, j] = weight
                adj[j, i] = weight  # ensure symmetry

        # Add self-loops so each node includes itself in the aggregation
        adj += np.eye(self.n_nodes, dtype=np.float32)

        # Row-normalize: D^{-1} A — gives a weighted mean of neighbors
        deg = adj.sum(axis=1, keepdims=True)
        self.norm_adj = adj / np.maximum(deg, 1e-8)

    def compute_hop1_features(self, node_features):
        """
        Compute 1-hop neighborhood aggregation of node features.

        Args:
            node_features: (N, F) array of per-node features (already scaled to [-1, 1])

        Returns:
            (N, F) array of neighborhood-averaged features
        """
        return self.norm_adj @ node_features

    def compute_graph_summaries(self, node_features):
        """
        Compute graph-level summary statistics that help the agent decide
        between the 5 high-level strategies.

        The summaries capture:
          1. Coverage gap: how under-served is the network overall
          2. Demand spread: how varied dynamic demand is (concentrated vs uniform)
          3. Utilization imbalance: how varied station capacity is
          4. Spatial mismatch: correlation between neighborhood demand and supply
          5. Station density: what fraction of nodes have stations

        Args:
            node_features: (N, F) array of per-node features (scaled to [-1, 1])
                Feature indices:
                  3 = dynamic demand
                  6 = station capability
                  9 = nearest station distance

        Returns:
            (5,) array of summary features, clipped to [-1, 1]
        """
        dynamic_demand = node_features[:, 3]
        capability = node_features[:, 6]
        nearest_dist = node_features[:, 9]

        # 1. Coverage gap: mean distance to nearest station
        #    High value → poor coverage → build new stations
        coverage_gap = float(np.mean(nearest_dist))

        # 2. Demand spread: std of dynamic demand, scaled to ~[-1, 1]
        #    High value → demand is heterogeneous → demand-based actions help
        demand_std = float(np.std(dynamic_demand))
        demand_spread = np.clip(2.0 * demand_std - 1.0, -1.0, 1.0)

        # 3. Utilization imbalance: std of capability among stations
        #    High value → uneven station sizes → expand underserved ones
        station_mask = capability > -0.9  # nodes with stations
        n_stations = int(np.sum(station_mask))
        if n_stations > 1:
            cap_std = float(np.std(capability[station_mask]))
            util_imbalance = np.clip(2.0 * cap_std - 1.0, -1.0, 1.0)
        else:
            util_imbalance = -1.0

        # 4. Spatial mismatch: correlation between neighborhood demand and supply
        #    Negative → demand and supply are spatially misaligned → relocate
        hop1_demand = self.norm_adj @ dynamic_demand
        hop1_capability = self.norm_adj @ np.clip(capability, -1.0, 1.0)
        d_std = float(np.std(hop1_demand))
        c_std = float(np.std(hop1_capability))
        if d_std > 1e-8 and c_std > 1e-8:
            corr = float(np.corrcoef(hop1_demand, hop1_capability)[0, 1])
            spatial_mismatch = np.clip(corr, -1.0, 1.0) if not np.isnan(corr) else 0.0
        else:
            spatial_mismatch = 0.0

        # 5. Station density: fraction of nodes with stations, scaled to [-1, 1]
        station_fraction = float(np.mean(station_mask.astype(np.float32)))
        station_density = np.clip(2.0 * station_fraction - 1.0, -1.0, 1.0)

        return np.array([
            coverage_gap,
            demand_spread,
            util_imbalance,
            spatial_mismatch,
            station_density
        ], dtype=np.float32)
