from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List

import numpy as np
import torch

from .clustering import cluster_requests
from .fleet import VehicleSpec, assign_fleet_to_hubs, expand_vehicle_inventory
from .learned_policy import TransitPlannerPolicy, choose_oracle_hubs, default_vehicle_specs, extract_selected_vehicles
from .models import HubProposal
from .pipeline import build_day_ahead_plan_from_hubs, reassign_requests_to_hubs
from .validate_clustering import _make_synthetic_requests


def _clusters_to_hubs(clusters) -> List[HubProposal]:
    return [
        HubProposal(
            hub_id=f"hub_{cluster.cluster_id}",
            cluster_id=cluster.cluster_id,
            location=cluster.center,
            request_ids=list(cluster.request_ids),
            demand=cluster.demand,
            radius=cluster.radius,
            score=cluster.score,
            access_cost=cluster.compactness + 0.5 * cluster.temporal_span,
        )
        for cluster in clusters
    ]


def _mean_or_zero(values: List[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _validate_split(
    policy: TransitPlannerPolicy,
    num_instances: int,
    base_seed: int,
    num_candidates: int,
    num_hubs: int,
    fleet_specs: List[VehicleSpec],
    learned_order_path: str,
) -> dict:
    policy.eval()
    hub_losses: List[float] = []
    fleet_losses: List[float] = []
    proxy_costs: List[float] = []

    with torch.no_grad():
        for offset in range(num_instances):
            seed = base_seed + offset
            requests = _make_synthetic_requests(seed=seed)
            candidate_clusters = cluster_requests(requests, num_clusters=num_candidates, seed=seed)
            oracle_idx = choose_oracle_hubs(candidate_clusters, requests, fleet_specs, num_hubs, learned_model_path=learned_order_path)
            oracle_clusters = [candidate_clusters[idx] for idx in oracle_idx]
            oracle_hubs = reassign_requests_to_hubs(requests, _clusters_to_hubs(oracle_clusters))

            hub_losses.append(float(policy.hub_bce_loss(candidate_clusters, [cluster.cluster_id for cluster in oracle_clusters]).item()))

            inventory = expand_vehicle_inventory(fleet_specs)
            assignment_lookup = extract_selected_vehicles(assign_fleet_to_hubs(oracle_hubs, requests, fleet_specs))
            fleet_loss = 0.0
            for hub in oracle_hubs:
                fleet_loss += float(policy.vehicle_bce_loss(hub, inventory, assignment_lookup.get(hub.hub_id, [])).item())
            fleet_losses.append(fleet_loss)

            proxy_plan = build_day_ahead_plan_from_hubs(
                oracle_hubs,
                requests,
                fleet_specs,
                fleet_strategy="heuristic",
                order_strategy="learned",
                learned_model_path=learned_order_path,
            )
            proxy_costs.append(sum(route.route_length for route in proxy_plan.routes))

    return {
        "hub_loss": _mean_or_zero(hub_losses),
        "fleet_loss": _mean_or_zero(fleet_losses),
        "proxy_cost": _mean_or_zero(proxy_costs),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the joint hub and fleet planner")
    parser.add_argument("--train-instances", type=int, default=200)
    parser.add_argument("--val-instances", type=int, default=40)
    parser.add_argument("--base-seed", type=int, default=7000)
    parser.add_argument("--num-hubs", type=int, default=3)
    parser.add_argument("--num-candidates", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--save-path", type=str, default="outputs/transit_joint_policy.pt")
    parser.add_argument("--learned-order-path", type=str, default="outputs/transit_orderer.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = TransitPlannerPolicy(hidden_dim=args.hidden_dim).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)
    fleet_specs = default_vehicle_specs()

    os.makedirs(Path(args.save_path).parent, exist_ok=True)
    best_score = None

    for epoch in range(args.epochs):
        policy.train()
        train_hub_losses: List[float] = []
        train_fleet_losses: List[float] = []
        train_proxy_costs: List[float] = []

        for offset in range(args.train_instances):
            seed = args.base_seed + offset
            requests = _make_synthetic_requests(seed=seed)
            candidate_clusters = cluster_requests(requests, num_clusters=args.num_candidates, seed=seed)
            oracle_idx = choose_oracle_hubs(
                candidate_clusters,
                requests,
                fleet_specs,
                args.num_hubs,
                learned_model_path=args.learned_order_path,
            )
            oracle_clusters = [candidate_clusters[idx] for idx in oracle_idx]
            oracle_hubs = reassign_requests_to_hubs(requests, _clusters_to_hubs(oracle_clusters))

            hub_loss = policy.hub_bce_loss(candidate_clusters, [cluster.cluster_id for cluster in oracle_clusters])
            inventory = expand_vehicle_inventory(fleet_specs)
            assignment_lookup = extract_selected_vehicles(assign_fleet_to_hubs(oracle_hubs, requests, fleet_specs))

            fleet_loss = torch.tensor(0.0, device=device)
            for hub in oracle_hubs:
                fleet_loss = fleet_loss + policy.vehicle_bce_loss(hub, inventory, assignment_lookup.get(hub.hub_id, []))

            loss = hub_loss + 0.5 * fleet_loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()

            train_hub_losses.append(float(hub_loss.item()))
            train_fleet_losses.append(float(fleet_loss.item()))

            with torch.no_grad():
                proxy_plan = build_day_ahead_plan_from_hubs(
                    oracle_hubs,
                    requests,
                    fleet_specs,
                    fleet_strategy="heuristic",
                    order_strategy="learned",
                    learned_model_path=args.learned_order_path,
                )
                train_proxy_costs.append(sum(route.route_length for route in proxy_plan.routes))

        val_metrics = _validate_split(
            policy,
            num_instances=args.val_instances,
            base_seed=args.base_seed + 10000,
            num_candidates=args.num_candidates,
            num_hubs=args.num_hubs,
            fleet_specs=fleet_specs,
            learned_order_path=args.learned_order_path,
        )

        summary = {
            "epoch": epoch,
            "train_hub_loss": round(_mean_or_zero(train_hub_losses), 6),
            "train_fleet_loss": round(_mean_or_zero(train_fleet_losses), 6),
            "train_proxy_cost": round(_mean_or_zero(train_proxy_costs), 6),
            "val_hub_loss": round(val_metrics["hub_loss"], 6),
            "val_fleet_loss": round(val_metrics["fleet_loss"], 6),
            "val_proxy_cost": round(val_metrics["proxy_cost"], 6),
        }
        print(summary)

        score = val_metrics["hub_loss"] + val_metrics["fleet_loss"] + 0.01 * val_metrics["proxy_cost"]
        if best_score is None or score < best_score:
            best_score = score
            torch.save(
                {
                    "model": policy.state_dict(),
                    "args": vars(args),
                },
                args.save_path,
            )

    print(f"Saved best joint planner checkpoint to {args.save_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
