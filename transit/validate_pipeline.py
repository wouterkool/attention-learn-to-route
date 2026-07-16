from __future__ import annotations

import argparse
from collections import Counter

from .fleet import VehicleSpec
from .pipeline import build_day_ahead_plan
from .validate_clustering import _make_synthetic_requests


def main() -> int:
    parser = argparse.ArgumentParser(description="End-to-end smoke test for the day-ahead transit planner")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--num-hubs", type=int, default=3)
    args = parser.parse_args()

    requests = _make_synthetic_requests(seed=args.seed)
    fleet = [
        VehicleSpec(vehicle_type="feeder_8", capacity=8, role="feeder", available_count=3),
        VehicleSpec(vehicle_type="feeder_12", capacity=12, role="feeder", available_count=2),
        VehicleSpec(vehicle_type="trunk_24", capacity=24, role="trunk", available_count=2),
        VehicleSpec(vehicle_type="trunk_40", capacity=40, role="trunk", available_count=1),
    ]

    heuristic_plan = build_day_ahead_plan(
        requests,
        fleet,
        num_hubs=args.num_hubs,
        seed=args.seed,
        order_strategy="nearest_neighbor",
    )
    attention_plan = build_day_ahead_plan(
        requests,
        fleet,
        num_hubs=args.num_hubs,
        seed=args.seed,
        order_strategy="attention",
        attention_model_path="pretrained/tsp_20",
    )

    request_counts = Counter()
    for route in heuristic_plan.routes:
        assert route.route_length >= 0.0, "Route length must be non-negative"
        assert route.load > 0, "Route load must be positive"
        for rid in route.ordered_request_ids:
            request_counts[rid] += 1

    assert len(request_counts) == len(requests), "Not all requests are covered by routes"
    assert all(count == 1 for count in request_counts.values()), "Requests were assigned multiple times"

    total_route_load = sum(route.load for route in heuristic_plan.routes)
    total_request_load = sum(req.passenger_count for req in requests)
    assert total_route_load == total_request_load, "Route loads do not match total demand"

    heuristic_length = sum(route.route_length for route in heuristic_plan.routes)
    attention_length = sum(route.route_length for route in attention_plan.routes)

    print(f"hubs={len(heuristic_plan.hubs)} assignments={len(heuristic_plan.route_assignments)} routes={len(heuristic_plan.routes)}")
    print(f"total_demand={total_request_load} total_route_load={total_route_load}")
    print(f"heuristic_total_route_length={heuristic_length:.4f}")
    print(f"attention_total_route_length={attention_length:.4f}")
    for route in heuristic_plan.routes:
        print(
            {
                "assignment_id": route.assignment_id,
                "vehicle_id": route.vehicle_id,
                "hub_id": route.hub_id,
                "stop_count": route.stop_count,
                "load": route.load,
                "route_length": round(route.route_length, 4),
                "ordered_request_ids": route.ordered_request_ids,
            }
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
