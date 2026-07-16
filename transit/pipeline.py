from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import math

import numpy as np

from .clustering import build_hub_proposals
from .clustering import cluster_requests
from .fleet import VehicleSpec, assign_fleet_to_hubs
from .learned_policy import assign_fleet_learned, select_learned_hubs
from .models import HubProposal, PassengerRequest
from .routing import RoutePlan, plan_routes


@dataclass(frozen=True)
class DayAheadPlan:
    hubs: List[HubProposal]
    route_assignments: List
    routes: List[RoutePlan]


def _distance(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _build_service_hub(hub: HubProposal, assigned_requests: Sequence[PassengerRequest]) -> HubProposal:
    if not assigned_requests:
        return HubProposal(
            hub_id=hub.hub_id,
            cluster_id=hub.cluster_id,
            location=hub.location,
            request_ids=[],
            demand=0,
            radius=0.0,
            score=hub.score,
            access_cost=hub.access_cost,
        )
    points = np.asarray([req.origin for req in assigned_requests], dtype=np.float64)
    weights = np.asarray([max(int(req.passenger_count), 1) for req in assigned_requests], dtype=np.float64)
    center = np.average(points, axis=0, weights=weights)
    radius = float(max(_distance(hub.location, tuple(point)) for point in points)) if len(points) else 0.0
    compactness = float(np.average([_distance(hub.location, tuple(point)) for point in points], weights=weights))
    temporal_span = float(max(req.request_time for req in assigned_requests) - min(req.request_time for req in assigned_requests))
    return HubProposal(
        hub_id=hub.hub_id,
        cluster_id=hub.cluster_id,
        location=(float(hub.location[0]), float(hub.location[1])),
        request_ids=[req.request_id for req in assigned_requests],
        demand=int(sum(req.passenger_count for req in assigned_requests)),
        radius=radius,
        score=hub.score,
        access_cost=float(compactness + 0.5 * temporal_span),
    )


def reassign_requests_to_hubs(
    requests: Sequence[PassengerRequest],
    hubs: Sequence[HubProposal],
) -> List[HubProposal]:
    if not hubs:
        return []
    assigned = {hub.hub_id: [] for hub in hubs}
    for request in requests:
        best_hub = min(hubs, key=lambda hub: _distance(request.origin, hub.location))
        assigned[best_hub.hub_id].append(request)
    return [_build_service_hub(hub, assigned[hub.hub_id]) for hub in hubs]


def build_day_ahead_plan(
    requests: Sequence[PassengerRequest],
    vehicle_specs: Sequence[VehicleSpec],
    num_hubs: int = 3,
    seed: int = 1234,
    hub_strategy: str = "heuristic",
    fleet_strategy: str = "heuristic",
    order_strategy: str = "learned",
    attention_model_path: str = "pretrained/tsp_20",
    learned_model_path: str = "outputs/transit_orderer.pt",
    joint_policy_path: str = "outputs/transit_joint_policy.pt",
    joint_policy_hidden_dim: int = 64,
    hub_candidate_multiplier: int = 2,
) -> DayAheadPlan:
    if hub_strategy == "learned":
        candidate_count = max(num_hubs * hub_candidate_multiplier, num_hubs)
        candidate_clusters = cluster_requests(requests, num_clusters=candidate_count, seed=seed)
        selected_clusters = select_learned_hubs(
            candidate_clusters,
            num_hubs=num_hubs,
            model_path=joint_policy_path,
            hidden_dim=joint_policy_hidden_dim,
        )
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
        hubs = reassign_requests_to_hubs(requests, hubs)
    else:
        hubs = build_hub_proposals(requests, num_hubs=num_hubs, seed=seed)
    return build_day_ahead_plan_from_hubs(
        hubs,
        requests,
        vehicle_specs,
        fleet_strategy=fleet_strategy,
        order_strategy=order_strategy,
        attention_model_path=attention_model_path,
        learned_model_path=learned_model_path,
        joint_policy_path=joint_policy_path,
        joint_policy_hidden_dim=joint_policy_hidden_dim,
    )


def build_day_ahead_plan_from_hubs(
    hubs: Sequence[HubProposal],
    requests: Sequence[PassengerRequest],
    vehicle_specs: Sequence[VehicleSpec],
    fleet_strategy: str = "heuristic",
    order_strategy: str = "learned",
    attention_model_path: str = "pretrained/tsp_20",
    learned_model_path: str = "outputs/transit_orderer.pt",
    joint_policy_path: str = "outputs/transit_joint_policy.pt",
    joint_policy_hidden_dim: int = 64,
) -> DayAheadPlan:
    if fleet_strategy == "learned":
        assignments = assign_fleet_learned(
            hubs,
            requests,
            vehicle_specs,
            model_path=joint_policy_path,
            hidden_dim=joint_policy_hidden_dim,
        )
    else:
        assignments = assign_fleet_to_hubs(hubs, requests, vehicle_specs)
    routes = plan_routes(
        assignments,
        requests,
        hubs,
        order_strategy=order_strategy,
        attention_model_path=attention_model_path,
        learned_model_path=learned_model_path,
    )
    return DayAheadPlan(hubs=list(hubs), route_assignments=list(assignments), routes=routes)
