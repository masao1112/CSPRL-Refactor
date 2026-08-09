"""
Set/attention-based feature extractor for SB3 DQN policies -- an alternative
to `gnn_extractor.py`'s message-passing GCN.

Motivation (see chat discussion): StationPlacementEnv's policy only ever
picks among 5 *meta*-actions (build-by-benefit, build-by-demand, add-by-
benefit, add-by-demand, relocate) -- node selection itself is always done
by a fixed heuristic (`StationPlacementEnv._control_action` calls
`H.choose_node_new_benefit` / `H.choose_node_bydemand` / ...), never learned.
So the encoder's job is to produce ONE fixed-size summary of the whole
road-network state for a 5-way decision, not a per-node embedding. The
existing GCN extractor flattens all N node embeddings into the output
(`n_nodes * k_features + 32`), which (a) ties the network to one district's
node count, (b) squeezes a 256-dim per-node representation down to
`k_features` scalars right before the policy head, and (c) only has a
3-hop receptive field (3 stacked graph-conv layers) -- not obviously enough
to summarize a several-hundred-node city graph for a *global* decision.

This module sidesteps all three: linear-embed each node, run a few
self-attention layers (global receptive field from layer 1, since every
node attends to every other node), then pool to a FIXED-size vector
regardless of how many nodes there are.

Architecture, in short::

    node_features (N, F) --> linear embed --> [self-attention block] x n_layers
                          --> pool (PMA or mean, see `use_pma`) --> fixed-size vector
    global_state (budget, step, score_delta) --> small MLP
    concat(pooled, global) --> fed to SB3's Q-head (policy_kwargs.net_arch)

Relationship to PPO-Attention
------------------------------
The encoder shape (linear embed -> stacked self-attention blocks) is
borrowed from PPO-Attention (Liu, Sun & Qi, "Optimal Placement of Charging
Stations in Road Networks: A Reinforcement Learning Approach with Attention
Mechanism", Applied Sciences 13:8473, 2023) -- the closest published
precedent, itself extending the same PCRL problem/action-space family this
repo is built on. Two deliberate departures from it:

  1. Self-attention here is UNMASKED (every node attends to every other
     node). PPO-Attention's Eq. (8) restricts attention to graph edges
     (u_ij = -inf if i is not adjacent to j) -- that's really a Graph
     Attention Network (GAT) layer, not a Set-Transformer, and it still
     depends on road-graph adjacency, the exact dependency this module is
     trying to drop. Dropping the mask trades a graph-locality prior for a
     global receptive field in a single layer, and for one fewer moving
     part (no adjacency matrix to build/cache, unlike `gnn_extractor.py`).
     Reintroducing an edge-distance attention *bias* (soft, not a hard
     mask) would be a natural follow-up if pure set-attention underperforms.
  2. LayerNorm is used instead of PPO-Attention's BatchNorm for the
     residual sublayers. SB3 calls `forward()` on batch size 1 during
     environment rollout (action selection) and on `batch_size` during
     gradient updates; BatchNorm's running statistics behave inconsistently
     across that mismatch (batch size 1 breaks batch statistics in train
     mode entirely). LayerNorm has no batch-size dependence.
  3. PPO-Attention's decoder is an MLP applied directly to the per-node
     encoder output (`v = MLP(h_i^N)`) -- as written, its output still
     appears to scale with N (see chat discussion; the paper is ambiguous
     here). This module adds an explicit pooling step (PMA or mean) so the
     output is *always* a fixed size, independent of N -- this is the part
     PPO-Attention does not clearly do, and the main reason to prefer this
     over just copying their decoder.

The `use_pma` flag toggles the pooling step so the two modes can be compared
head-to-head with everything else (encoder weights init aside) held fixed:

  - use_pma=True (default): Pooling by Multi-head Attention (PMA), from
    Set Transformer (Lee et al., "Set Transformer: A Framework for
    Attention-based Permutation-Invariant Neural Networks", ICML 2019,
    arXiv:1810.00825). A small set of *learned* query vectors ("seeds")
    attend over the node embeddings, producing `pma_seeds` fixed-size
    summary vectors regardless of how many nodes there are.
  - use_pma=False: plain mean-pool over node embeddings. No learned
    parameters in the pooling step -- isolates whether the *learned*
    attention-pooling is earning its keep over a naive average, holding
    the encoder (embedding + self-attention layers) identical. This is
    the ablation baseline, not a second architecture to fight over.

Unlike `gnn_extractor.py`, this module ignores the `edge_index`/`edge_attr`
entries of the observation Dict entirely -- it treats the road network as
an unordered set of nodes, not a graph, so it needs no adjacency matrix and
no per-instance caching of one.
"""
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class SelfAttentionBlock(nn.Module):
    """One pre-norm Transformer encoder block: multi-head self-attention
    sublayer + feed-forward sublayer, each wrapped in a residual connection
    with LayerNorm. Unmasked -- every node attends to every other node."""

    def __init__(self, embed_dim, n_heads, ff_mult=4, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * ff_mult),
            nn.ReLU(),
            nn.Linear(embed_dim * ff_mult, embed_dim),
        )

    def forward(self, h):
        # h: (B, N, D)
        h_norm = self.norm1(h)
        attn_out, _ = self.attn(h_norm, h_norm, h_norm, need_weights=False)
        h = h + attn_out
        h = h + self.ff(self.norm2(h))
        return h


class PMAPooling(nn.Module):
    """Pooling by Multi-head Attention (Lee et al., ICML 2019). `n_seeds`
    learned query vectors attend over the (variable-length) node set,
    producing `n_seeds` fixed-size summary vectors -- output size never
    depends on the number of nodes."""

    def __init__(self, embed_dim, n_heads, n_seeds=1, dropout=0.0):
        super().__init__()
        self.n_seeds = n_seeds
        self.seeds = nn.Parameter(torch.empty(1, n_seeds, embed_dim))
        nn.init.xavier_uniform_(self.seeds)
        self.attn = nn.MultiheadAttention(embed_dim, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, h):
        # h: (B, N, D) -> (B, n_seeds * D)
        B = h.shape[0]
        seeds = self.seeds.expand(B, -1, -1)  # (B, n_seeds, D)
        attn_out, _ = self.attn(seeds, h, h, need_weights=False)
        pooled = self.norm1(seeds + attn_out)
        pooled = self.norm2(pooled + self.ff(pooled))
        return pooled.reshape(B, -1)


class AttentionFeaturesExtractor(BaseFeaturesExtractor):
    """
    Encoder-only, permutation-invariant feature extractor for SB3.
    Expects the same Dict observation space `GNNFeaturesExtractor` uses
    (node_features, edge_index, edge_attr, global_state) -- edge_index/
    edge_attr are accepted for interface compatibility but never read.

    Args:
        embed_dim: per-node embedding width used throughout the encoder.
        n_heads: attention heads (must divide embed_dim).
        n_layers: number of stacked self-attention blocks.
        pma_seeds: number of PMA seed (query) vectors -- i.e. how many
            summary vectors the pooling step produces when use_pma=True.
            Ignored when use_pma=False.
        use_pma: True -> learned PMA pooling. False -> mean-pool ablation
            baseline (see module docstring).
        ff_mult: feed-forward hidden-layer expansion factor.
        dropout: attention/dropout rate (0.0 = deterministic, easiest to
            compare like-for-like against the mean-pool baseline).
    """

    def __init__(self, observation_space, embed_dim=128, n_heads=4, n_layers=2,
                 pma_seeds=1, use_pma=True, ff_mult=4, dropout=0.0):
        if embed_dim % n_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by n_heads ({n_heads})")

        n_global_features = observation_space.spaces["global_state"].shape[0]
        global_mlp_dim = 32
        pooled_dim = (pma_seeds * embed_dim) if use_pma else embed_dim
        # Unlike GNNFeaturesExtractor's actual_features_dim (n_nodes * k_features + 32),
        # this is independent of n_nodes -- the whole point of pooling.
        actual_features_dim = pooled_dim + global_mlp_dim
        super().__init__(observation_space, actual_features_dim)

        n_node_features = observation_space.spaces["node_features"].shape[1]
        self.use_pma = use_pma

        self.embed = nn.Linear(n_node_features, embed_dim)
        self.embed_norm = nn.LayerNorm(embed_dim)

        self.layers = nn.ModuleList([
            SelfAttentionBlock(embed_dim, n_heads, ff_mult=ff_mult, dropout=dropout)
            for _ in range(n_layers)
        ])

        if use_pma:
            self.pool = PMAPooling(embed_dim, n_heads, n_seeds=pma_seeds, dropout=dropout)
        else:
            self.pool = None  # mean-pool has no learnable parameters

        self.global_mlp = nn.Sequential(
            nn.Linear(n_global_features, global_mlp_dim),
            nn.ReLU(),
        )

    def forward(self, observations):
        x = observations["node_features"]              # (B, N, F)
        global_state = observations["global_state"]     # (B, G)

        h = self.embed_norm(self.embed(x))               # (B, N, D)
        for layer in self.layers:
            h = layer(h)

        if self.use_pma:
            pooled = self.pool(h)                        # (B, pma_seeds * D)
        else:
            pooled = h.mean(dim=1)                        # (B, D)  -- ablation baseline

        global_embed = self.global_mlp(global_state)      # (B, 32)
        return torch.cat([pooled, global_embed], dim=1)
