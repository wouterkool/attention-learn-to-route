from __future__ import annotations

import argparse
import statistics
from dataclasses import dataclass
from typing import Dict, List

from .custom_instance import generate_custom_instance, load_instance, _to_requests, _to_vehicle_specs
from .pipeline import build_day_ahead_plan


@dataclass(frozen=True)
class JointMetrics:
    total_route_length: float
    total_load: int
    total_capacity: int
    utilization: float
    route_count: int
    hub_count: int


def summarize_plan(plan) -> JointMetrics:
    total_route_length = sum(route.route_length for route in plan.routes)
    total_load = sum(route.load for route in plan.routes)
    total_capacity = sum(assignment.capacity for assignment in plan.route_assignments)
    utilization = float(total_load) / float(total_capacity) if total_capacity > 0 else 0.0
    return JointMetrics(
        total_route_length=float(total_route_length),
        total_load=int(total_load),
        total_capacity=int(total_capacity),
        utilization=float(utilization),
        route_count=len(plan.routes),
        hub_count=len(plan.hubs),
    )


def _run_strategies(requests, vehicle_specs, num_hubs, seed, joint_policy_path, learned_order_path):
    return {
        "heuristic": build_day_ahead_plan(
            requests,
            vehicle_specs,
            num_hubs=num_hubs,
            seed=seed,
            hub_strategy="heuristic",
            fleet_strategy="heuristic",
            order_strategy="nearest_neighbor",
        ),
        "learned_order": build_day_ahead_plan(
            requests,
            vehicle_specs,
            num_hubs=num_hubs,
            seed=seed,
            hub_strategy="heuristic",
            fleet_strategy="heuristic",
            order_strategy="learned",
            learned_model_path=learned_order_path,
        ),
        "joint_learned": build_day_ahead_plan(
            requests,
            vehicle_specs,
            num_hubs=num_hubs,
            seed=seed,
            hub_strategy="learned",
            fleet_strategy="learned",
            order_strategy="learned",
            learned_model_path=learned_order_path,
            joint_policy_path=joint_policy_path,
        ),
    }


def _print_metrics(name: str, metrics: JointMetrics) -> None:
    print(
        {
            "strategy": name,
            "total_route_length": round(metrics.total_route_length, 6),
            "total_load": metrics.total_load,
            "total_capacity": metrics.total_capacity,
            "utilization": round(metrics.utilization, 6),
            "route_count": metrics.route_count,
            "hub_count": metrics.hub_count,
        }
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark the full joint transit planner")
    parser.add_argument("--instances", type=int, default=20)
    parser.add_argument("--base-seed", type=int, default=1234)
    parser.add_argument("--num-hubs", type=int, default=3)
    parser.add_argument("--instance-json", type=str, default=None, help="Benchmark a single custom instance from JSON")
    parser.add_argument("--joint-policy-path", type=str, default="outputs/transit_joint_policy.pt")
    parser.add_argument("--learned-order-path", type=str, default="outputs/transit_orderer.pt")
    args = parser.parse_args()

    if args.instance_json is not None:
        payload = load_instance(args.instance_json)
        requests = _to_requests(payload)
        vehicle_specs = _to_vehicle_specs(payload)
        num_hubs = int(payload.get("num_hubs", args.num_hubs))
        seed = int(payload.get("seed", args.base_seed))
        plans = _run_strategies(requests, vehicle_specs, num_hubs, seed, args.joint_policy_path, args.learned_order_path)
        total_request_load = sum(req.passenger_count for req in requests)
        for name, plan in plans.items():
            metrics = summarize_plan(plan)
            assert metrics.total_load == total_request_load, f"{name} did not cover total demand"
            _print_metrics(name, metrics)
        return 0

    results: Dict[str, List[JointMetrics]] = {"heuristic": [], "learned_order": [], "joint_learned": []}
    for offset in range(args.instances):
        seed = args.base_seed + offset
        payload = generate_custom_instance(seed=seed, num_clusters=args.num_hubs, requests_per_cluster=12)
        requests = _to_requests(payload)
        vehicle_specs = _to_vehicle_specs(payload)
        plans = _run_strategies(requests, vehicle_specs, args.num_hubs, seed, args.joint_policy_path, args.learned_order_path)
        total_request_load = sum(req.passenger_count for req in requests)
        for name, plan in plans.items():
            metrics = summarize_plan(plan)
            assert metrics.total_load == total_request_load, f"{name} did not cover total demand"
            results[name].append(metrics)

    def mean_metric(name: str, attr: str) -> float:
        return statistics.mean(getattr(m, attr) for m in results[name])

    print(f"instances={args.instances} num_hubs={args.num_hubs} base_seed={args.base_seed}")
    for name in ["heuristic", "learned_order", "joint_learned"]:
        print(
            {
                "strategy": name,
                "mean_route_length": round(mean_metric(name, "total_route_length"), 6),
                "mean_utilization": round(mean_metric(name, "utilization"), 6),
                "mean_route_count": round(mean_metric(name, "route_count"), 3),
                "mean_hub_count": round(mean_metric(name, "hub_count"), 3),
            }
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
