from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import math
import os
import torch

from nets.attention_model import set_decode_type
from utils.functions import load_model

from .order_model import TransitRouteOrderer
from .fleet import VehicleAssignment
from .models import HubProposal, PassengerRequest


Coordinate = Tuple[float, float]


@dataclass(frozen=True)
class RoutePlan:
    assignment_id: str
    hub_id: str
    vehicle_id: str
    ordered_request_ids: List[str] = field(default_factory=list)
    route_length: float = 0.0
    load: int = 0
    stop_count: int = 0


_ATTENTION_MODEL_CACHE = {}
_LEARNED_MODEL_CACHE = {}


def _load_attention_orderer(model_path: str):
    if model_path not in _ATTENTION_MODEL_CACHE:
        model, _ = load_model(model_path)
        set_decode_type(model, "greedy")
        model.eval()
        _ATTENTION_MODEL_CACHE[model_path] = model
    return _ATTENTION_MODEL_CACHE[model_path]


def _load_learned_orderer(model_path: str):
    if model_path not in _LEARNED_MODEL_CACHE:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Learned route-ordering checkpoint not found: {model_path}")
        ckpt = torch.load(model_path, map_location="cpu")
        args = ckpt.get("args", {})
        model = TransitRouteOrderer(hidden_dim=int(args.get("hidden_dim", 96)))
        model.load_state_dict(ckpt["model"])
        model.eval()
        _LEARNED_MODEL_CACHE[model_path] = model
    return _LEARNED_MODEL_CACHE[model_path]


def _distance(a: Coordinate, b: Coordinate) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _nearest_neighbor_order(points: Dict[str, Coordinate], start: Coordinate) -> List[str]:
    remaining = set(points.keys())
    order: List[str] = []
    current = start

    while remaining:
        next_id = min(remaining, key=lambda rid: _distance(current, points[rid]))
        order.append(next_id)
        current = points[next_id]
        remaining.remove(next_id)

    return order


def _attention_order(points: Dict[str, Coordinate], model_path: str) -> List[str]:
    if len(points) <= 1:
        return list(points.keys())

    model = _load_attention_orderer(model_path)
    keys = list(points.keys())
    coords = torch.tensor([points[key] for key in keys], dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        _, _, pi = model(coords, return_pi=True)
    ordering = pi[0].tolist()
    return [keys[idx] for idx in ordering]


def _learned_order(
    points: Dict[str, Coordinate],
    demands: Dict[str, float],
    hub: Coordinate,
    role: int,
    capacity: int,
    model_path: str,
) -> List[str]:
    if len(points) <= 1:
        return list(points.keys())

    model = _load_learned_orderer(model_path)
    keys = sorted(points.keys())
    coords = torch.tensor([points[key] for key in keys], dtype=torch.float32).unsqueeze(0)
    demand_t = torch.tensor([[float(demands[key]) for key in keys]], dtype=torch.float32)
    hub_t = torch.tensor([hub], dtype=torch.float32)
    capacity_t = torch.tensor([[float(capacity) / 40.0]], dtype=torch.float32)
    role_t = torch.tensor([role], dtype=torch.long)
    with torch.no_grad():
        order = model.greedy_order(hub_t, coords, demand_t, capacity_t, role_t).tolist()
    return [keys[idx] for idx in order]


def build_route_plan(
    assignment: VehicleAssignment,
    requests: Sequence[PassengerRequest],
    hub: HubProposal,
    order_strategy: str = "nearest_neighbor",
    attention_model_path: str = "pretrained/tsp_20",
    learned_model_path: str = "outputs/transit_orderer.pt",
) -> RoutePlan:
    request_lookup = {req.request_id: req for req in requests}
    points = {rid: request_lookup[rid].origin for rid in assignment.request_ids}
    demands = {rid: request_lookup[rid].passenger_count for rid in assignment.request_ids}
    if not points:
        return RoutePlan(
            assignment_id=assignment.assignment_id,
            hub_id=assignment.hub_id,
            vehicle_id=assignment.vehicle_id,
            ordered_request_ids=[],
            route_length=0.0,
            load=0,
            stop_count=0,
        )

    if order_strategy == "attention":
        order = _attention_order(points, attention_model_path)
    elif order_strategy == "learned":
        order = _learned_order(
            points,
            demands,
            hub.location,
            0 if assignment.vehicle_role == "feeder" else 1,
            assignment.capacity,
            learned_model_path,
        )
    else:
        order = _nearest_neighbor_order(points, hub.location)

    route_length = 0.0
    current = hub.location
    for rid in order:
        route_length += _distance(current, points[rid])
        current = points[rid]
    route_length += _distance(current, hub.location)

    load = sum(request_lookup[rid].passenger_count for rid in order)
    return RoutePlan(
        assignment_id=assignment.assignment_id,
        hub_id=assignment.hub_id,
        vehicle_id=assignment.vehicle_id,
        ordered_request_ids=order,
        route_length=float(route_length),
        load=int(load),
        stop_count=len(order),
    )


def plan_routes(
    assignments: Sequence[VehicleAssignment],
    requests: Sequence[PassengerRequest],
    hubs: Sequence[HubProposal],
    order_strategy: str = "nearest_neighbor",
    attention_model_path: str = "pretrained/tsp_20",
    learned_model_path: str = "outputs/transit_orderer.pt",
) -> List[RoutePlan]:
    hub_lookup = {hub.hub_id: hub for hub in hubs}
    plans: List[RoutePlan] = []
    for assignment in assignments:
        plans.append(
            build_route_plan(
                assignment,
                requests,
                hub_lookup[assignment.hub_id],
                order_strategy=order_strategy,
                attention_model_path=attention_model_path,
                learned_model_path=learned_model_path,
            )
        )
    return plans
