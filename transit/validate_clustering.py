from __future__ import annotations

import argparse
from dataclasses import asdict
from typing import List

import numpy as np

from .clustering import build_hub_proposals, cluster_requests
from .models import PassengerRequest


def _make_synthetic_requests(seed: int = 1234) -> List[PassengerRequest]:
    rng = np.random.default_rng(seed)
    centers = np.array(
        [
            [0.18, 0.22],
            [0.72, 0.25],
            [0.55, 0.78],
        ],
        dtype=np.float64,
    )
    requests: List[PassengerRequest] = []
    idx = 0
    for cluster_idx, center in enumerate(centers):
        count = 10 + cluster_idx * 2
        for _ in range(count):
            origin = center + rng.normal(scale=0.015 + 0.005 * cluster_idx, size=2)
            destination = np.clip(center + rng.normal(scale=0.08, size=2), 0.0, 1.0)
            requests.append(
                PassengerRequest(
                    request_id=f"r{idx}",
                    origin=(float(origin[0]), float(origin[1])),
                    destination=(float(destination[0]), float(destination[1])),
                    request_time=float(rng.uniform(0.0, 60.0)),
                    passenger_count=int(rng.integers(1, 4)),
                    priority=float(rng.uniform(0.5, 1.5)),
                )
            )
            idx += 1
    return requests


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test for demand clustering and hub proposals")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--num-hubs", type=int, default=3)
    args = parser.parse_args()

    requests = _make_synthetic_requests(seed=args.seed)
    clusters = cluster_requests(requests, num_clusters=args.num_hubs, seed=args.seed)
    hubs = build_hub_proposals(requests, num_hubs=args.num_hubs, seed=args.seed)

    total_demand = sum(req.passenger_count for req in requests)
    clustered_demand = sum(cluster.demand for cluster in clusters)
    hub_demand = sum(hub.demand for hub in hubs)
    assigned_requests = sum(len(cluster.request_ids) for cluster in clusters)

    print(f"requests={len(requests)} total_demand={total_demand}")
    print(f"clusters={len(clusters)} hubs={len(hubs)}")
    print(f"clustered_demand={clustered_demand} hub_demand={hub_demand}")
    print(f"assigned_requests={assigned_requests}")

    assert len(clusters) == args.num_hubs, "Unexpected cluster count"
    assert len(hubs) == args.num_hubs, "Unexpected hub count"
    assert clustered_demand == total_demand, "Demand was not conserved across clusters"
    assert hub_demand == total_demand, "Demand was not conserved across hub proposals"
    assert assigned_requests == len(requests), "Some requests were not assigned to any cluster"
    assert all(cluster.radius >= 0.0 for cluster in clusters), "Cluster radius must be non-negative"
    assert all(np.isfinite(cluster.score) for cluster in clusters), "Cluster score must be finite"
    assert all(np.isfinite(hub.score) for hub in hubs), "Hub score must be finite"

    print("Top clusters:")
    for cluster in clusters:
        print(asdict(cluster))

    print("Top hub proposals:")
    for hub in hubs:
        print(asdict(hub))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

