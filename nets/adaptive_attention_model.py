import math

import torch
from torch import nn

from nets.attention_model import AttentionModel


class AdaptiveHierarchicalAttentionRouter(AttentionModel):
    """
    TSP-only prototype that keeps the original attention solver, but:
    - adds size conditioning to the decoder query
    - restricts attention to a sparse local candidate set after the first move

    This is a pragmatic implementation of the proposed novel model. It is not a
    full hierarchical clustering stack, but it captures the key research idea:
    use local routing structure to reduce computation while preserving the
    original model's inductive bias.
    """

    def __init__(
        self,
        embedding_dim,
        hidden_dim,
        problem,
        n_encode_layers=2,
        tanh_clipping=10.0,
        mask_inner=True,
        mask_logits=True,
        normalization="batch",
        n_heads=8,
        checkpoint_encoder=False,
        shrink_size=None,
        local_topk=None,
        sparse_after_step=3,
        adaptive_topk_fraction=0.25,
        adaptive_topk_min=8,
        size_conditioning=True,
        max_graph_size=None,
    ):
        assert problem.NAME == "tsp", "AdaptiveHierarchicalAttentionRouter currently supports TSP only"
        super().__init__(
            embedding_dim,
            hidden_dim,
            problem,
            n_encode_layers=n_encode_layers,
            tanh_clipping=tanh_clipping,
            mask_inner=mask_inner,
            mask_logits=mask_logits,
            normalization=normalization,
            n_heads=n_heads,
            checkpoint_encoder=checkpoint_encoder,
            shrink_size=shrink_size,
        )

        self.size_conditioning = size_conditioning
        self.max_graph_size = max_graph_size
        self.local_topk = local_topk
        self.sparse_after_step = sparse_after_step
        self.adaptive_topk_fraction = adaptive_topk_fraction
        self.adaptive_topk_min = adaptive_topk_min

        if self.size_conditioning:
            self.size_encoder = nn.Sequential(
                nn.Linear(1, embedding_dim),
                nn.ReLU(),
                nn.Linear(embedding_dim, 2 * embedding_dim),
            )
            nn.init.zeros_(self.size_encoder[-1].weight)
            nn.init.zeros_(self.size_encoder[-1].bias)
        else:
            self.size_encoder = None

    def _get_size_context(self, fixed):
        if not self.size_conditioning:
            return None

        graph_size = fixed.node_embeddings.size(1)
        scale = float(graph_size)
        if self.max_graph_size is not None and self.max_graph_size > 0:
            scale = scale / float(self.max_graph_size)
        size_token = fixed.node_embeddings.new_full((fixed.node_embeddings.size(0), 1), scale)
        size_context = self.size_encoder(size_token)
        size_scale, size_shift = size_context.chunk(2, dim=-1)
        return size_scale[:, None, :], size_shift[:, None, :]

    def _get_candidate_indices(self, state):
        batch_size = state.ids.size(0)
        n_nodes = state.loc.size(-2)
        sparse_after_step = max(self.sparse_after_step, n_nodes // 2)
        if n_nodes >= 100:
            sparse_after_step = max(sparse_after_step, (2 * n_nodes) // 3)

        # Keep the first few decisions global, then move to local candidate routing.
        if state.i.item() < sparse_after_step:
            return torch.arange(n_nodes, device=state.loc.device).expand(batch_size, n_nodes)

        current = state.prev_a[:, 0]
        batch_ids = torch.arange(batch_size, device=state.loc.device)
        dist = state.dist[batch_ids, current]
        dist = dist + state.get_mask().float()[:, 0, :] * 1e6

        remaining = n_nodes - state.i.item()
        if self.local_topk is None:
            topk_fraction = self.adaptive_topk_fraction
            if n_nodes >= 50:
                topk_fraction = max(topk_fraction, 0.9)
            if n_nodes >= 100:
                topk_fraction = max(topk_fraction, 0.95)
            k = max(self.adaptive_topk_min, int(math.ceil(n_nodes * topk_fraction)))
        else:
            k = self.local_topk
        k = min(k, remaining, n_nodes)

        return dist.topk(k, dim=-1, largest=False)[1]

    def _sparse_log_p(self, fixed, state, query):
        batch_size = state.ids.size(0)
        n_nodes = state.loc.size(-2)
        candidate_idx = self._get_candidate_indices(state)
        candidate_k = candidate_idx.size(-1)

        if candidate_k >= n_nodes:
            return None

        batch_idx = torch.arange(batch_size, device=state.loc.device)[:, None]
        candidate_embeddings = fixed.node_embeddings[batch_idx, candidate_idx]

        candidate_node_proj = self.project_node_embeddings(candidate_embeddings[:, None, :, :])
        glimpse_key, glimpse_val, logit_key = candidate_node_proj.chunk(3, dim=-1)
        glimpse_K = self._make_heads(glimpse_key, 1)
        glimpse_V = self._make_heads(glimpse_val, 1)

        candidate_mask = torch.gather(state.get_mask(), 2, candidate_idx[:, None, :])
        log_p, _ = self._one_to_many_logits(query, glimpse_K, glimpse_V, logit_key, candidate_mask)

        full_log_p = log_p.new_full((batch_size, 1, n_nodes), -math.inf)
        full_log_p.scatter_(2, candidate_idx[:, None, :], log_p)
        return full_log_p

    def _get_log_p(self, fixed, state, normalize=True):
        query = fixed.context_node_projected + self.project_step_context(
            self._get_parallel_step_context(fixed.node_embeddings, state)
        )

        if self.size_conditioning:
            # Allow the graph-size signal to shift the query without changing the rest of the solver.
            size_scale, size_shift = self._get_size_context(fixed)
            query = query * (1 + torch.tanh(size_scale)) + size_shift

        sparse_log_p = self._sparse_log_p(fixed, state, query)
        if sparse_log_p is not None:
            log_p = sparse_log_p
            mask = state.get_mask()
            if normalize:
                log_p = torch.log_softmax(log_p / self.temp, dim=-1)
            return log_p, mask

        glimpse_K, glimpse_V, logit_K = self._get_attention_node_data(fixed, state)
        mask = state.get_mask()
        log_p, _ = self._one_to_many_logits(query, glimpse_K, glimpse_V, logit_K, mask)

        if normalize:
            log_p = torch.log_softmax(log_p / self.temp, dim=-1)

        assert not torch.isnan(log_p).any()
        return log_p, mask
