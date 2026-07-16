from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .models import HubProposal, PassengerRequest


@dataclass(frozen=True)
class VehicleSpec:
    vehicle_type: str
    capacity: int
    role: str
    available_count: int = 1
    fixed_cost: float = 0.0
    per_km_cost: float = 0.0


@dataclass(frozen=True)
class VehicleInstance:
    vehicle_id: str
    vehicle_type: str
    capacity: int
    role: str
    fixed_cost: float = 0.0
    per_km_cost: float = 0.0


@dataclass(frozen=True)
class VehicleAssignment:
    assignment_id: str
    hub_id: str
    cluster_id: int
    vehicle_id: str
    vehicle_type: str
    vehicle_role: str
    capacity: int
    load: int
    utilization: float
    request_ids: List[str] = field(default_factory=list)


def expand_vehicle_inventory(specs: Sequence[VehicleSpec]) -> List[VehicleInstance]:
    expanded: List[VehicleInstance] = []
    for spec in specs:
        for idx in range(spec.available_count):
            expanded.append(
                VehicleInstance(
                    vehicle_id=f"{spec.vehicle_type}_{idx}",
                    vehicle_type=spec.vehicle_type,
                    capacity=spec.capacity,
                    role=spec.role,
                    fixed_cost=spec.fixed_cost,
                    per_km_cost=spec.per_km_cost,
                )
            )
    expanded.sort(key=lambda v: (v.role, -v.capacity, v.vehicle_id))
    return expanded


def _pack_requests(requests: List[PassengerRequest], capacities: Sequence[int]) -> List[List[PassengerRequest]]:
    remaining = sorted(requests, key=lambda req: (req.passenger_count, req.priority), reverse=True)
    bins: List[List[PassengerRequest]] = [[] for _ in capacities]
    loads = [0 for _ in capacities]

    for req in remaining:
        best_bin = None
        best_slack = None
        for idx, cap in enumerate(capacities):
            slack = cap - (loads[idx] + req.passenger_count)
            if slack < 0:
                continue
            if best_slack is None or slack < best_slack:
                best_slack = slack
                best_bin = idx
        if best_bin is None:
            raise ValueError(f"Cannot place request {req.request_id} into any available vehicle")
        bins[best_bin].append(req)
        loads[best_bin] += req.passenger_count

    return bins


def pack_requests_into_vehicles(
    requests: Sequence[PassengerRequest],
    capacities: Sequence[int],
) -> List[List[PassengerRequest]]:
    return _pack_requests(list(requests), list(capacities))


def assign_fleet_to_hubs(
    hubs: Sequence[HubProposal],
    requests: Sequence[PassengerRequest],
    vehicle_specs: Sequence[VehicleSpec],
    trunk_demand_threshold: int = 24,
) -> List[VehicleAssignment]:
    request_lookup: Dict[str, PassengerRequest] = {req.request_id: req for req in requests}
    vehicles = expand_vehicle_inventory(vehicle_specs)
    available = list(vehicles)
    assignments: List[VehicleAssignment] = []

    hubs_sorted = sorted(hubs, key=lambda h: (-h.score, h.hub_id))
    for hub in hubs_sorted:
        cluster_requests = [request_lookup[rid] for rid in hub.request_ids]
        total_load = sum(req.passenger_count for req in cluster_requests)
        if total_load <= 0:
            continue

        preferred_role = "trunk" if total_load >= trunk_demand_threshold else "feeder"

        eligible = [veh for veh in available if veh.role == preferred_role]
        fallback = [veh for veh in available if veh.role != preferred_role]
        chosen = sorted(
            eligible + fallback,
            key=lambda v: (0 if v.role == preferred_role else 1, -v.capacity, v.vehicle_id),
        )
        if not chosen:
            raise ValueError("No vehicles available for assignment")

        chosen.sort(key=lambda v: (-v.capacity, v.vehicle_id))

        # Greedily choose the smallest prefix that can cover total load.
        selected: List[VehicleInstance] = []
        covered = 0
        for veh in chosen:
            selected.append(veh)
            covered += veh.capacity
            if covered >= total_load:
                break
        if covered < total_load:
            raise ValueError(f"Insufficient fleet capacity for hub {hub.hub_id}")

        selected_capacities = [veh.capacity for veh in selected]
        packed = pack_requests_into_vehicles(cluster_requests, selected_capacities)

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

        used_ids = {veh.vehicle_id for veh in selected}
        available = [veh for veh in available if veh.vehicle_id not in used_ids]

    return assignments
