from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import dataclass
from typing import Dict, List, Sequence

from .custom_instance import generate_custom_instance, load_instance, _to_requests, _to_vehicle_specs
from .pipeline import build_day_ahead_plan


@dataclass(frozen=True)
class PipelineMetrics:
    total_route_length: float
    total_load: int
    total_capacity: int
    utilization: float
    mean_route_utilization: float
    route_count: int
    hub_count: int


def summarize_plan(plan) -> PipelineMetrics:
    total_route_length = sum(route.route_length for route in plan.routes)
    total_load = sum(route.load for route in plan.routes)
    total_capacity = sum(assignment.capacity for assignment in plan.route_assignments)
    utilization = float(total_load) / float(total_capacity) if total_capacity > 0 else 0.0
    route_utils = [route.load / assignment.capacity for route, assignment in zip(plan.routes, plan.route_assignments)]
    mean_route_utilization = statistics.mean(route_utils) if route_utils else 0.0
    return PipelineMetrics(
        total_route_length=float(total_route_length),
        total_load=int(total_load),
        total_capacity=int(total_capacity),
        utilization=float(utilization),
        mean_route_utilization=float(mean_route_utilization),
        route_count=len(plan.routes),
        hub_count=len(plan.hubs),
    )


def _run_strategies(requests, vehicle_specs, num_hubs, seed):
    return {
        "heuristic": build_day_ahead_plan(
            requests,
            vehicle_specs,
            num_hubs=num_hubs,
            seed=seed,
            order_strategy="nearest_neighbor",
        ),
        "attention": build_day_ahead_plan(
            requests,
            vehicle_specs,
            num_hubs=num_hubs,
            seed=seed,
            order_strategy="attention",
            attention_model_path="pretrained/tsp_20",
        ),
        "learned": build_day_ahead_plan(
            requests,
            vehicle_specs,
            num_hubs=num_hubs,
            seed=seed,
            order_strategy="learned",
            learned_model_path="outputs/transit_orderer.pt",
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark full transit pipeline metrics")
    parser.add_argument("--instances", type=int, default=20)
    parser.add_argument("--base-seed", type=int, default=1234)
    parser.add_argument("--num-hubs", type=int, default=3)
    parser.add_argument("--instance-json", type=str, default=None, help="Benchmark a single custom instance from JSON")
    args = parser.parse_args()

    if args.instance_json is not None:
        payload = load_instance(args.instance_json)
        requests = _to_requests(payload)
        vehicle_specs = _to_vehicle_specs(payload)
        num_hubs = int(payload.get("num_hubs", args.num_hubs))
        seed = int(payload.get("seed", args.base_seed))
        plans = _run_strategies(requests, vehicle_specs, num_hubs, seed)
        for name, plan in plans.items():
            metrics = summarize_plan(plan)
            print(name, metrics)
        return 0

    results: Dict[str, List[PipelineMetrics]] = {"heuristic": [], "attention": [], "learned": []}
    for offset in range(args.instances):
        seed = args.base_seed + offset
        payload = generate_custom_instance(seed=seed, num_clusters=args.num_hubs, requests_per_cluster=12)
        requests = _to_requests(payload)
        vehicle_specs = _to_vehicle_specs(payload)
        plans = _run_strategies(requests, vehicle_specs, args.num_hubs, seed)
        for name, plan in plans.items():
            results[name].append(summarize_plan(plan))

    def mean_metric(name: str, attr: str) -> float:
        return statistics.mean(getattr(m, attr) for m in results[name])

    print(f"instances={args.instances} num_hubs={args.num_hubs} base_seed={args.base_seed}")
    for name in ["heuristic", "attention", "learned"]:
        print(
            {
                "strategy": name,
                "mean_route_length": round(mean_metric(name, "total_route_length"), 6),
                "mean_utilization": round(mean_metric(name, "utilization"), 6),
                "mean_route_utilization": round(mean_metric(name, "mean_route_utilization"), 6),
                "mean_route_count": round(mean_metric(name, "route_count"), 3),
            }
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
