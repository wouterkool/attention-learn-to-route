from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .fleet import VehicleSpec
from .order_utils import route_length, two_opt_route
from .pipeline import build_day_ahead_plan
from .validate_clustering import _make_synthetic_requests


@dataclass(frozen=True)
class OrderingSample:
    hub: np.ndarray
    role: int
    capacity: float
    coords: np.ndarray
    demands: np.ndarray
    target_order: List[int]
    heuristic_length: float
    optimized_length: float
    route_id: str


def default_fleet() -> List[VehicleSpec]:
    return [
        VehicleSpec(vehicle_type="feeder_8", capacity=8, role="feeder", available_count=3),
        VehicleSpec(vehicle_type="feeder_12", capacity=12, role="feeder", available_count=2),
        VehicleSpec(vehicle_type="trunk_24", capacity=24, role="trunk", available_count=2),
        VehicleSpec(vehicle_type="trunk_40", capacity=40, role="trunk", available_count=1),
    ]


def build_ordering_samples(
    num_instances: int,
    base_seed: int = 1234,
    num_hubs: int = 3,
) -> List[OrderingSample]:
    fleet = default_fleet()
    samples: List[OrderingSample] = []

    for offset in range(num_instances):
        seed = base_seed + offset
        requests = _make_synthetic_requests(seed=seed)
        request_lookup = {req.request_id: req for req in requests}
        plan = build_day_ahead_plan(
            requests,
            fleet,
            num_hubs=num_hubs,
            seed=seed,
            order_strategy="nearest_neighbor",
        )
        hub_lookup = {hub.hub_id: hub for hub in plan.hubs}
        assignment_lookup = {assignment.assignment_id: assignment for assignment in plan.route_assignments}

        for route in plan.routes:
            if len(route.ordered_request_ids) <= 1:
                continue

            sorted_ids = sorted(route.ordered_request_ids)
            coords = np.asarray([request_lookup[rid].origin for rid in sorted_ids], dtype=np.float64)
            demands = np.asarray([request_lookup[rid].passenger_count for rid in sorted_ids], dtype=np.float64)
            initial_order = [sorted_ids.index(rid) for rid in route.ordered_request_ids]
            target_order, optimized_length = two_opt_route(coords, hub_lookup[route.hub_id].location, initial_order)
            heuristic_length = route.route_length
            assignment = assignment_lookup[route.assignment_id]

            samples.append(
                OrderingSample(
                    hub=np.asarray(hub_lookup[route.hub_id].location, dtype=np.float64),
                    role=0 if route.vehicle_id.startswith("feeder") else 1,
                    capacity=float(assignment.capacity),
                    coords=coords,
                    demands=demands,
                    target_order=target_order,
                    heuristic_length=float(heuristic_length),
                    optimized_length=float(optimized_length),
                    route_id=route.assignment_id,
                )
            )

    return samples


class OrderingDataset(Dataset):
    def __init__(self, samples: Sequence[OrderingSample]):
        self.samples = list(samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> OrderingSample:
        return self.samples[idx]


def collate_ordering_samples(batch: Sequence[OrderingSample]) -> List[OrderingSample]:
    return list(batch)
