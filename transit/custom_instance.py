from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from typing import Dict, List

import numpy as np

from .fleet import VehicleSpec
from .models import PassengerRequest


def default_vehicle_specs() -> List[VehicleSpec]:
    return [
        VehicleSpec(vehicle_type="feeder_8", capacity=8, role="feeder", available_count=3),
        VehicleSpec(vehicle_type="feeder_12", capacity=12, role="feeder", available_count=2),
        VehicleSpec(vehicle_type="trunk_24", capacity=24, role="trunk", available_count=2),
        VehicleSpec(vehicle_type="trunk_40", capacity=40, role="trunk", available_count=1),
    ]


def generate_custom_instance(
    seed: int = 1234,
    num_clusters: int = 3,
    requests_per_cluster: int = 12,
    cluster_spread: float = 0.02,
) -> Dict:
    rng = np.random.default_rng(seed)
    centers = rng.uniform(0.15, 0.85, size=(num_clusters, 2))
    requests: List[Dict] = []
    request_id = 0
    for cluster_idx, center in enumerate(centers):
        for _ in range(requests_per_cluster):
            origin = np.clip(center + rng.normal(scale=cluster_spread, size=2), 0.0, 1.0)
            destination = np.clip(center + rng.normal(scale=cluster_spread * 4.0, size=2), 0.0, 1.0)
            requests.append(
                {
                    "request_id": f"r{request_id}",
                    "origin": origin.tolist(),
                    "destination": destination.tolist(),
                    "request_time": float(rng.uniform(0.0, 60.0)),
                    "passenger_count": int(rng.integers(1, 4)),
                    "time_window": None,
                    "priority": float(rng.uniform(0.5, 1.5)),
                }
            )
            request_id += 1
    return {
        "seed": seed,
        "num_hubs": num_clusters,
        "requests": requests,
        "vehicle_specs": [asdict(spec) for spec in default_vehicle_specs()],
        "cluster_spread": cluster_spread,
    }


def load_instance(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_instance(instance: Dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(instance, f, indent=2)


def _to_requests(payload: Dict) -> List[PassengerRequest]:
    requests: List[PassengerRequest] = []
    for item in payload["requests"]:
        requests.append(
            PassengerRequest(
                request_id=item["request_id"],
                origin=tuple(item["origin"]),
                destination=tuple(item["destination"]),
                request_time=float(item.get("request_time", 0.0)),
                passenger_count=int(item.get("passenger_count", 1)),
                time_window=tuple(item["time_window"]) if item.get("time_window") else None,
                priority=float(item.get("priority", 1.0)),
            )
        )
    return requests


def _to_vehicle_specs(payload: Dict) -> List[VehicleSpec]:
    specs: List[VehicleSpec] = []
    for item in payload["vehicle_specs"]:
        specs.append(
            VehicleSpec(
                vehicle_type=item["vehicle_type"],
                capacity=int(item["capacity"]),
                role=item["role"],
                available_count=int(item.get("available_count", 1)),
                fixed_cost=float(item.get("fixed_cost", 0.0)),
                per_km_cost=float(item.get("per_km_cost", 0.0)),
            )
        )
    return specs


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a custom synthetic mixed-fleet transit instance")
    parser.add_argument("--output", type=str, default="custom_transit_instance.json")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--num-clusters", type=int, default=3)
    parser.add_argument("--requests-per-cluster", type=int, default=12)
    parser.add_argument("--cluster-spread", type=float, default=0.02)
    args = parser.parse_args()

    instance = generate_custom_instance(
        seed=args.seed,
        num_clusters=args.num_clusters,
        requests_per_cluster=args.requests_per_cluster,
        cluster_spread=args.cluster_spread,
    )
    save_instance(instance, args.output)
    print(f"saved {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

