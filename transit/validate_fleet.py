from __future__ import annotations

import argparse
from collections import Counter
from typing import List

import numpy as np

from .clustering import build_hub_proposals
from .fleet import VehicleSpec, assign_fleet_to_hubs
from .models import PassengerRequest
from .validate_clustering import _make_synthetic_requests


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test for mixed-fleet assignment")
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    requests = _make_synthetic_requests(seed=args.seed)
    hubs = build_hub_proposals(requests, num_hubs=3, seed=args.seed)

    fleet = [
        VehicleSpec(vehicle_type="feeder_8", capacity=8, role="feeder", available_count=3),
        VehicleSpec(vehicle_type="feeder_12", capacity=12, role="feeder", available_count=2),
        VehicleSpec(vehicle_type="trunk_24", capacity=24, role="trunk", available_count=2),
        VehicleSpec(vehicle_type="trunk_40", capacity=40, role="trunk", available_count=1),
    ]

    assignments = assign_fleet_to_hubs(hubs, requests, fleet, trunk_demand_threshold=22)

    request_lookup = {req.request_id: req for req in requests}
    assigned = Counter()
    for assignment in assignments:
        load = sum(request_lookup[rid].passenger_count for rid in assignment.request_ids)
        assert load == assignment.load, "Assignment load mismatch"
        assert load <= assignment.capacity, "Vehicle capacity exceeded"
        assert 0.0 < assignment.utilization <= 1.0, "Invalid utilization"
        for rid in assignment.request_ids:
            assigned[rid] += 1

    assert len(assigned) == len(requests), "Not all requests were assigned"
    assert all(count == 1 for count in assigned.values()), "Some requests were assigned multiple times"

    total_load = sum(req.passenger_count for req in requests)
    total_assigned_load = sum(a.load for a in assignments)
    assert total_load == total_assigned_load, "Total load was not conserved"

    print(f"requests={len(requests)} total_load={total_load} assignments={len(assignments)}")
    print("Assignments:")
    for assignment in assignments:
        print(
            {
                "assignment_id": assignment.assignment_id,
                "hub_id": assignment.hub_id,
                "vehicle_id": assignment.vehicle_id,
                "vehicle_role": assignment.vehicle_role,
                "capacity": assignment.capacity,
                "load": assignment.load,
                "utilization": round(assignment.utilization, 3),
                "request_ids": assignment.request_ids,
            }
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

