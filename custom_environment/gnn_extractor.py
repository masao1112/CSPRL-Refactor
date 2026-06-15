import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class GNNFeaturesExtractor(BaseFeaturesExtractor):
    """
    Custom Feature Extractor for SB3 to process graph observations.
    Expects observation_space to be a spaces.Dict with:
      - node_features: (N, F)
      - edge_index: (2, E)
      - global_state: (1,)
    """
    def __init__(self, observation_space, features_dim=256):
        super(GNNFeaturesExtractor, self).__init__(observation_space, features_dim)
        
        n_node_features = observation_space.spaces["node_features"].shape[1]
        self.n_nodes = observation_space.spaces["node_features"].shape[0]
        n_global_features = observation_space.spaces["global_state"].shape[0]

        self._cached_norm_adj = None
        self._cached_device = None

        self.node_features_mlp = nn.Sequential(
            nn.Linear(n_node_features, 32),
            nn.ReLU()
        )

        # Stream 1: Local Node Features (GCN)
        self.gcn1_weight = nn.Parameter(torch.Tensor(32, 64))
        self.gcn1_bias = nn.Parameter(torch.Tensor(64))
        
        self.gcn2_weight = nn.Parameter(torch.Tensor(64, 128))
        self.gcn2_bias = nn.Parameter(torch.Tensor(128))

        self.gcn3_weight = nn.Parameter(torch.Tensor(128, 256))
        self.gcn3_bias = nn.Parameter(torch.Tensor(256))
        
        nn.init.xavier_uniform_(self.gcn1_weight)
        nn.init.zeros_(self.gcn1_bias)
        nn.init.xavier_uniform_(self.gcn2_weight)
        nn.init.zeros_(self.gcn2_bias)
        nn.init.xavier_uniform_(self.gcn3_weight)
        nn.init.zeros_(self.gcn3_bias)

        # Stream 2: Global State (MLP)
        self.global_mlp = nn.Sequential(
            nn.Linear(n_global_features, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # Fusion Layer
        self.linear = nn.Linear(256 + 64, features_dim)

    def _build_norm_adj(self, edge_index, edge_weight, device):
        """Compute and cache the normalized adjacency matrix (static graph)."""
        edges = edge_index[0].long()  # (2, E)

        adj = torch.zeros((self.n_nodes, self.n_nodes), device=device)
        adj[edges[0], edges[1]] = edge_weight
        print(adj.shape)
        adj += torch.eye(self.n_nodes, device=device)

        deg = adj.sum(dim=1)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0
        norm_adj = deg_inv_sqrt.view(-1, 1) * adj * deg_inv_sqrt.view(1, -1)

        self._cached_norm_adj = norm_adj
        self._cached_device = device

    def forward(self, observations):
        x = observations["node_features"]      # (B, N, F)
        edge_index = observations["edge_index"] # (B, 2, E)
        edge_attr = observations["edge_attr"]   # (B, E, 1)
        global_state = observations["global_state"]    # (B, G)
        
        B = x.shape[0]
        device = x.device

        # Build norm_adj once (graph is static); rebuild only on device change
        if self._cached_norm_adj is None or self._cached_device != device:
            edge_distance = edge_attr[0, :, 0] # [-1, 1]
            # Convert scaled distance to positive affinity weight. 
            # Closer nodes (lower distance) will have higher edge weights.
            edge_weight = torch.exp(-edge_distance)
            self._build_norm_adj(edge_index, edge_weight, device)

        # --- Node Features (MLP) ---
        embs = self.node_features_mlp(x) # (B, 32)

        # --- Local Stream (GNN) ---
        norm_adj = self._cached_norm_adj.unsqueeze(0).expand(B, -1, -1)

        support1 = torch.matmul(embs, self.gcn1_weight)
        out1 = torch.relu(torch.bmm(norm_adj, support1) + self.gcn1_bias)

        support2 = torch.matmul(out1, self.gcn2_weight)
        out2 = torch.relu(torch.bmm(norm_adj, support2) + self.gcn2_bias)

        support3 = torch.matmul(out2, self.gcn3_weight)
        out3 = torch.relu(torch.bmm(norm_adj, support3) + self.gcn3_bias)

        graph_embed = out3.mean(dim=1) # (B, 256)
        
        # --- Global Stream (MLP) ---
        global_embed = self.global_mlp(global_state) # (B, 128)
        
        # --- Fusion ---
        combined = torch.cat([graph_embed, global_embed], dim=1) # (B, 384)
        
        return torch.relu(self.linear(combined))
