from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import math
import torch
from torch import nn
import torch.nn.functional as F


class TransitRouteOrderer(nn.Module):
    def __init__(self, hidden_dim: int = 96, role_dim: int = 8):
        super().__init__()
        self.role_embedding = nn.Embedding(2, role_dim)

        context_dim = 2 + 2 + 1 + 1 + 1 + role_dim
        cand_dim = 2 + 1 + 2 + 2 + 2

        self.context_net = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.candidate_net = nn.Sequential(
            nn.Linear(cand_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.scorer = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def _context_features(
        self,
        hub: torch.Tensor,
        current: torch.Tensor,
        remaining_frac: torch.Tensor,
        step_frac: torch.Tensor,
        capacity: torch.Tensor,
        role: torch.Tensor,
    ) -> torch.Tensor:
        role_emb = self.role_embedding(role)
        return torch.cat((hub, current, remaining_frac, step_frac, capacity, role_emb), dim=-1)

    def _candidate_features(
        self,
        coords: torch.Tensor,
        demands: torch.Tensor,
        hub: torch.Tensor,
        current: torch.Tensor,
    ) -> torch.Tensor:
        delta_current = coords - current[:, None, :]
        delta_hub = coords - hub[:, None, :]
        dist_current = torch.norm(delta_current, dim=-1, keepdim=True)
        dist_hub = torch.norm(delta_hub, dim=-1, keepdim=True)
        return torch.cat(
            (
                coords,
                demands[:, :, None],
                delta_current,
                delta_hub,
                dist_current,
                dist_hub,
            ),
            dim=-1,
        )

    def score_candidates(
        self,
        hub: torch.Tensor,
        current: torch.Tensor,
        coords: torch.Tensor,
        demands: torch.Tensor,
        remaining_frac: torch.Tensor,
        step_frac: torch.Tensor,
        capacity: torch.Tensor,
        role: torch.Tensor,
    ) -> torch.Tensor:
        context = self._context_features(hub, current, remaining_frac, step_frac, capacity, role)
        context_emb = self.context_net(context)
        cand_emb = self.candidate_net(self._candidate_features(coords, demands, hub, current))
        context_expanded = context_emb[:, None, :].expand(-1, cand_emb.size(1), -1)
        logits = self.scorer(torch.cat((cand_emb, context_expanded), dim=-1)).squeeze(-1)
        return logits

    def greedy_order(
        self,
        hub: torch.Tensor,
        coords: torch.Tensor,
        demands: torch.Tensor,
        capacity: torch.Tensor,
        role: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, n, _ = coords.size()
        assert batch_size == 1, "Greedy ordering currently expects a single route"

        remaining = torch.arange(n, device=coords.device)
        current = hub
        order = []

        for step in range(n):
            cand_coords = coords[:, remaining, :]
            cand_demands = demands[:, remaining]
            remaining_frac = torch.full((1, 1), float(len(remaining)) / float(n), device=coords.device)
            step_frac = torch.full((1, 1), float(step) / float(max(n, 1)), device=coords.device)
            logits = self.score_candidates(
                hub=hub,
                current=current,
                coords=cand_coords,
                demands=cand_demands,
                remaining_frac=remaining_frac,
                step_frac=step_frac,
                capacity=capacity,
                role=role,
            )
            pick = int(torch.argmax(logits, dim=-1).item())
            chosen = int(remaining[pick].item())
            order.append(chosen)
            current = coords[:, chosen, :]
            remaining = torch.cat((remaining[:pick], remaining[pick + 1 :]))

        return torch.tensor(order, device=coords.device, dtype=torch.long)

    def step_loss(
        self,
        hub: torch.Tensor,
        coords: torch.Tensor,
        demands: torch.Tensor,
        target_order: Sequence[int],
        capacity: torch.Tensor,
        role: torch.Tensor,
    ) -> torch.Tensor:
        remaining = list(range(coords.size(1)))
        current = hub
        total_loss = coords.new_tensor(0.0)
        n = coords.size(1)

        for step, target_idx in enumerate(target_order):
            cand_coords = coords[:, remaining, :]
            cand_demands = demands[:, remaining]
            remaining_frac = coords.new_tensor([[float(len(remaining)) / float(n)]])
            step_frac = coords.new_tensor([[float(step) / float(max(n, 1))]])
            logits = self.score_candidates(
                hub=hub,
                current=current,
                coords=cand_coords,
                demands=cand_demands,
                remaining_frac=remaining_frac,
                step_frac=step_frac,
                capacity=capacity,
                role=role,
            )
            target_pos = remaining.index(int(target_idx))
            total_loss = total_loss + F.cross_entropy(logits, torch.tensor([target_pos], device=coords.device))
            current = coords[:, int(target_idx), :]
            remaining.pop(target_pos)

        return total_loss / float(n)

