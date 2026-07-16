from __future__ import annotations

from itertools import combinations
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from .clustering import cluster_requests
from .fleet import (
    VehicleInstance,
    VehicleSpec,
    assign_fleet_to_hubs,
    expand_vehicle_inventory,
    pack_requests_into_vehicles,
)
from .models import HubProposal, PassengerRequest, RequestCluster
from .fleet import VehicleAssignment


_POLICY_CACHE: Dict[str, "TransitPlannerPolicy"] = {}


def _mlp(in_dim: int, hidden_dim: int, out_dim: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, out_dim),
    )


def _cluster_features(cluster: RequestCluster, device: Optional[torch.device] = None) -> torch.Tensor:
    min_x, min_y, max_x, max_y = cluster.bbox
    area = max((max_x - min_x) * (max_y - min_y), 1e-6)
    return torch.tensor(
        [
            float(cluster.center[0]),
            float(cluster.center[1]),
            float(cluster.demand),
            float(cluster.radius),
            float(cluster.compactness),
            float(cluster.temporal_span),
            float(area),
            float(cluster.score),
        ],
        dtype=torch.float32,
        device=device,
    )


def _hub_features(hub: HubProposal, device: Optional[torch.device] = None) -> torch.Tensor:
    return torch.tensor(
        [
            float(hub.location[0]),
            float(hub.location[1]),
            float(hub.demand),
            float(hub.radius),
            float(hub.score),
            float(hub.access_cost),
        ],
        dtype=torch.float32,
        device=device,
    )


def _vehicle_features(
    vehicle: VehicleInstance,
    demand: float,
    hub_feat: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    role = 1.0 if vehicle.role == "trunk" else 0.0
    return torch.tensor(
        [
            float(vehicle.capacity) / 40.0,
            role,
            float(vehicle.fixed_cost),
            float(vehicle.per_km_cost),
            float(demand) / max(float(vehicle.capacity), 1.0),
            float(demand),
        ],
        dtype=torch.float32,
        device=device,
    )


class TransitPlannerPolicy(nn.Module):
    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.hub_scorer = _mlp(8, hidden_dim, 1)
        self.vehicle_scorer = _mlp(6 + 6, hidden_dim, 1)

    def device(self) -> torch.device:
        return next(self.parameters()).device

    def score_hubs(self, cluster_feats: torch.Tensor) -> torch.Tensor:
        return self.hub_scorer(cluster_feats).squeeze(-1)

    def score_vehicles(self, hub_feat: torch.Tensor, vehicle_feats: torch.Tensor) -> torch.Tensor:
        hub_expanded = hub_feat[:, None, :].expand(-1, vehicle_feats.size(1), -1)
        feats = torch.cat((hub_expanded, vehicle_feats), dim=-1)
        return self.vehicle_scorer(feats).squeeze(-1)

    def select_hubs(self, clusters: Sequence[RequestCluster], num_hubs: int) -> List[RequestCluster]:
        device = self.device()
        feats = torch.stack([_cluster_features(cluster, device=device) for cluster in clusters], dim=0)
        with torch.no_grad():
            scores = self.score_hubs(feats)
        order = torch.argsort(scores, descending=True).tolist()
        return [clusters[idx] for idx in order[:num_hubs]]

    def select_vehicles(
        self,
        hub: HubProposal,
        available: Sequence[VehicleInstance],
        demand: float,
    ) -> List[VehicleInstance]:
        device = self.device()
        hub_feat = _hub_features(hub, device=device)[None, :]
        vehicle_feats = torch.stack(
            [_vehicle_features(vehicle, demand, hub_feat[0], device=device) for vehicle in available],
            dim=0,
        )[
            None, :, :
        ]
        with torch.no_grad():
            scores = self.score_vehicles(hub_feat, vehicle_feats)[0]
        order = torch.argsort(scores, descending=True).tolist()
        selected: List[VehicleInstance] = []
        covered = 0
        for idx in order:
            vehicle = available[idx]
            selected.append(vehicle)
            covered += vehicle.capacity
            if covered >= demand:
                break
        return selected

    def hub_bce_loss(
        self,
        clusters: Sequence[RequestCluster],
        target_cluster_ids: Sequence[int],
    ) -> torch.Tensor:
        device = self.device()
        feats = torch.stack([_cluster_features(cluster, device=device) for cluster in clusters], dim=0)
        logits = self.score_hubs(feats)
        targets = torch.tensor(
            [1.0 if cluster.cluster_id in target_cluster_ids else 0.0 for cluster in clusters],
            dtype=torch.float32,
            device=device,
        )
        return F.binary_cross_entropy_with_logits(logits, targets)

    def vehicle_bce_loss(
        self,
        hub: HubProposal,
        available: Sequence[VehicleInstance],
        target_vehicle_ids: Sequence[str],
    ) -> torch.Tensor:
        demand = max(float(hub.demand), 1.0)
        device = self.device()
        hub_feat = _hub_features(hub, device=device)[None, :]
        vehicle_feats = torch.stack(
            [_vehicle_features(vehicle, demand, hub_feat[0], device=device) for vehicle in available],
            dim=0,
        )[
            None, :, :
        ]
        logits = self.score_vehicles(hub_feat, vehicle_feats)[0]
        targets = torch.tensor(
            [1.0 if vehicle.vehicle_id in target_vehicle_ids else 0.0 for vehicle in available],
            dtype=torch.float32,
            device=device,
        )
        return F.binary_cross_entropy_with_logits(logits, targets)


def _evaluate_subset_cost(
    selected_clusters: Sequence[RequestCluster],
    requests: Sequence[PassengerRequest],
    vehicle_specs: Sequence[VehicleSpec],
    learned_order_strategy: str = "learned",
    learned_model_path: str = "outputs/transit_orderer.pt",
) -> float:
    hubs = [
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
    for cluster in selected_clusters
    ]
    from .pipeline import build_day_ahead_plan_from_hubs
    from .pipeline import reassign_requests_to_hubs

    hubs = reassign_requests_to_hubs(requests, hubs)
    try:
        plan = build_day_ahead_plan_from_hubs(
            hubs,
            requests,
            vehicle_specs,
            order_strategy=learned_order_strategy,
            learned_model_path=learned_model_path,
        )
    except ValueError:
        return float("inf")
    return float(sum(route.route_length for route in plan.routes))


def choose_oracle_hubs(
    candidate_clusters: Sequence[RequestCluster],
    requests: Sequence[PassengerRequest],
    vehicle_specs: Sequence[VehicleSpec],
    num_hubs: int,
    learned_model_path: str = "outputs/transit_orderer.pt",
) -> List[int]:
    candidate_clusters = list(candidate_clusters)
    assert len(candidate_clusters) >= num_hubs
    best_subset: Optional[Tuple[int, ...]] = None
    best_cost: Optional[float] = None

    for subset in combinations(range(len(candidate_clusters)), num_hubs):
        clusters = [candidate_clusters[idx] for idx in subset]
        cost = _evaluate_subset_cost(clusters, requests, vehicle_specs, learned_model_path=learned_model_path)
        if best_cost is None or cost < best_cost:
            best_cost = cost
            best_subset = subset

    assert best_subset is not None
    return list(best_subset)


def load_policy(model_path: str, hidden_dim: int = 64) -> TransitPlannerPolicy:
    return _load_policy_from_checkpoint(model_path, hidden_dim=hidden_dim)


def _load_policy_from_checkpoint(model_path: str, hidden_dim: int = 64) -> TransitPlannerPolicy:
    if model_path in _POLICY_CACHE:
        return _POLICY_CACHE[model_path]
    from pathlib import Path

    if not Path(model_path).exists():
        raise FileNotFoundError(f"Transit planner checkpoint not found: {model_path}")
    ckpt = torch.load(model_path, map_location="cpu")
    args = ckpt.get("args", {})
    model = TransitPlannerPolicy(hidden_dim=int(args.get("hidden_dim", hidden_dim)))
    model.load_state_dict(ckpt["model"])
    model.eval()
    _POLICY_CACHE[model_path] = model
    return model


def select_learned_hubs(
    clusters: Sequence[RequestCluster],
    num_hubs: int,
    model_path: str,
    hidden_dim: int = 64,
) -> List[RequestCluster]:
    policy = _load_policy_from_checkpoint(model_path, hidden_dim=hidden_dim)
    return policy.select_hubs(clusters, num_hubs)


def select_learned_vehicle_ids(
    hub: HubProposal,
    available: Sequence[VehicleInstance],
    demand: float,
    model_path: str,
    hidden_dim: int = 64,
) -> List[VehicleInstance]:
    policy = _load_policy_from_checkpoint(model_path, hidden_dim=hidden_dim)
    return policy.select_vehicles(hub, available, demand)


def assign_fleet_learned(
    hubs: Sequence[HubProposal],
    requests: Sequence[PassengerRequest],
    vehicle_specs: Sequence[VehicleSpec],
    model_path: str,
    hidden_dim: int = 64,
) -> List[VehicleAssignment]:
    policy = _load_policy_from_checkpoint(model_path, hidden_dim=hidden_dim)
    available = expand_vehicle_inventory(vehicle_specs)
    selected_by_hub: Dict[str, List[str]] = {}
    for hub in sorted(hubs, key=lambda h: (-h.score, h.hub_id)):
        selected = policy.select_vehicles(hub, available, max(float(hub.demand), 1.0))
        if not selected:
            continue
        selected_by_hub[hub.hub_id] = [vehicle.vehicle_id for vehicle in selected]
        used = {vehicle.vehicle_id for vehicle in selected}
        available = [vehicle for vehicle in available if vehicle.vehicle_id not in used]
    return assign_fleet_to_hubs_with_selection(hubs, requests, vehicle_specs, selected_by_hub)


def assign_fleet_to_hubs_with_selection(
    hubs: Sequence[HubProposal],
    requests: Sequence[PassengerRequest],
    vehicle_specs: Sequence[VehicleSpec],
    selected_vehicle_ids_by_hub: Dict[str, List[str]],
) -> List[VehicleAssignment]:
    request_lookup = {req.request_id: req for req in requests}
    available = expand_vehicle_inventory(vehicle_specs)
    assignments: List[VehicleAssignment] = []

    for hub in sorted(hubs, key=lambda h: (-h.score, h.hub_id)):
        cluster_requests = [request_lookup[rid] for rid in hub.request_ids]
        total_load = sum(req.passenger_count for req in cluster_requests)
        if total_load <= 0:
            continue

        selected_ids = selected_vehicle_ids_by_hub.get(hub.hub_id, [])
        chosen = [vehicle for vehicle in available if vehicle.vehicle_id in selected_ids]

        if not chosen:
            preferred = "trunk" if total_load >= 24 else "feeder"
            chosen = sorted(
                [vehicle for vehicle in available if vehicle.role == preferred] + [vehicle for vehicle in available if vehicle.role != preferred],
                key=lambda v: (0 if v.role == preferred else 1, -v.capacity, v.vehicle_id),
            )
            if not chosen:
                raise ValueError("No vehicles available for learned assignment")

        chosen = sorted(chosen, key=lambda v: (-v.capacity, v.vehicle_id))
        selected: List[VehicleInstance] = []
        covered = 0
        for veh in chosen:
            selected.append(veh)
            covered += veh.capacity
            if covered >= total_load:
                break
        if covered < total_load:
            raise ValueError(f"Insufficient fleet capacity for hub {hub.hub_id}")

        packed = pack_requests_into_vehicles(cluster_requests, [veh.capacity for veh in selected])
        for idx, (veh, reqs) in enumerate(zip(selected, packed)):
            load = sum(req.passenger_count for req in reqs)
            assignments.append(
                VehicleAssignment(
                    assignment_id=f"{hub.hub_id}_{veh.vehicle_id}_{idx}",
                    hub_id=hub.hub_id,
                    cluster_id=hub.cluster_id,
                    vehicle_id=veh.vehicle_id,
                    vehicle_type=veh.vehicle_type,
                    vehicle_role=veh.role,
                    capacity=veh.capacity,
                    load=load,
                    utilization=float(load) / float(veh.capacity),
                    request_ids=[req.request_id for req in reqs],
                )
            )
        used_ids = {vehicle.vehicle_id for vehicle in selected}
        available = [vehicle for vehicle in available if vehicle.vehicle_id not in used_ids]

    return assignments


def default_vehicle_specs() -> List[VehicleSpec]:
    return [
        VehicleSpec(vehicle_type="feeder_8", capacity=8, role="feeder", available_count=3),
        VehicleSpec(vehicle_type="feeder_12", capacity=12, role="feeder", available_count=2),
        VehicleSpec(vehicle_type="trunk_24", capacity=24, role="trunk", available_count=2),
        VehicleSpec(vehicle_type="trunk_40", capacity=40, role="trunk", available_count=1),
    ]


def extract_selected_vehicles(assignments) -> Dict[str, List[str]]:
    selected: Dict[str, List[str]] = {}
    for assignment in assignments:
        selected.setdefault(assignment.hub_id, []).append(assignment.vehicle_id)
    return selected
