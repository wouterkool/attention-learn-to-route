from __future__ import annotations

import argparse
import statistics
from dataclasses import dataclass
from typing import Dict, List

from .fleet import VehicleSpec
from .pipeline import build_day_ahead_plan
from .validate_clustering import _make_synthetic_requests


@dataclass(frozen=True)
class BenchmarkResult:
    seed: int
    heuristic_length: float
    attention_length: float
    learned_length: float
    delta: float
    delta_pct: float
    learned_delta: float
    learned_delta_pct: float


def _make_fleet() -> List[VehicleSpec]:
    return [
        VehicleSpec(vehicle_type="feeder_8", capacity=8, role="feeder", available_count=3),
        VehicleSpec(vehicle_type="feeder_12", capacity=12, role="feeder", available_count=2),
        VehicleSpec(vehicle_type="trunk_24", capacity=24, role="trunk", available_count=2),
        VehicleSpec(vehicle_type="trunk_40", capacity=40, role="trunk", available_count=1),
    ]


def run_benchmark(num_instances: int, base_seed: int, num_hubs: int) -> List[BenchmarkResult]:
    fleet = _make_fleet()
    results: List[BenchmarkResult] = []

    for offset in range(num_instances):
        seed = base_seed + offset
        requests = _make_synthetic_requests(seed=seed)

        heuristic_plan = build_day_ahead_plan(
            requests,
            fleet,
            num_hubs=num_hubs,
            seed=seed,
            order_strategy="nearest_neighbor",
        )
        attention_plan = build_day_ahead_plan(
            requests,
            fleet,
            num_hubs=num_hubs,
            seed=seed,
            order_strategy="attention",
            attention_model_path="pretrained/tsp_20",
        )
        learned_plan = build_day_ahead_plan(
            requests,
            fleet,
            num_hubs=num_hubs,
            seed=seed,
            order_strategy="learned",
            learned_model_path="outputs/transit_orderer.pt",
        )

        heuristic_length = sum(route.route_length for route in heuristic_plan.routes)
        attention_length = sum(route.route_length for route in attention_plan.routes)
        learned_length = sum(route.route_length for route in learned_plan.routes)
        delta = attention_length - heuristic_length
        delta_pct = (delta / heuristic_length * 100.0) if heuristic_length > 0 else 0.0
        learned_delta = learned_length - heuristic_length
        learned_delta_pct = (learned_delta / heuristic_length * 100.0) if heuristic_length > 0 else 0.0

        results.append(
            BenchmarkResult(
                seed=seed,
                heuristic_length=float(heuristic_length),
                attention_length=float(attention_length),
                learned_length=float(learned_length),
                delta=float(delta),
                delta_pct=float(delta_pct),
                learned_delta=float(learned_delta),
                learned_delta_pct=float(learned_delta_pct),
            )
        )

    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark heuristic vs attention route ordering")
    parser.add_argument("--num-instances", type=int, default=20)
    parser.add_argument("--base-seed", type=int, default=1234)
    parser.add_argument("--num-hubs", type=int, default=3)
    args = parser.parse_args()

    results = run_benchmark(args.num_instances, args.base_seed, args.num_hubs)

    heuristic_lengths = [r.heuristic_length for r in results]
    attention_lengths = [r.attention_length for r in results]
    learned_lengths = [r.learned_length for r in results]
    deltas = [r.delta for r in results]
    delta_pcts = [r.delta_pct for r in results]
    learned_deltas = [r.learned_delta for r in results]
    learned_delta_pcts = [r.learned_delta_pct for r in results]
    wins = sum(1 for r in results if r.delta < 0)
    ties = sum(1 for r in results if abs(r.delta) < 1e-9)
    learned_wins = sum(1 for r in results if r.learned_delta < 0)
    learned_ties = sum(1 for r in results if abs(r.learned_delta) < 1e-9)

    print(f"instances={len(results)} num_hubs={args.num_hubs} base_seed={args.base_seed}")
    print(f"heuristic_mean={statistics.mean(heuristic_lengths):.6f}")
    print(f"attention_mean={statistics.mean(attention_lengths):.6f}")
    print(f"learned_mean={statistics.mean(learned_lengths):.6f}")
    print(f"delta_mean={statistics.mean(deltas):.6f}")
    print(f"delta_pct_mean={statistics.mean(delta_pcts):.3f}")
    print(f"delta_median={statistics.median(deltas):.6f}")
    print(f"wins={wins} ties={ties} losses={len(results) - wins - ties}")
    print(f"learned_delta_mean={statistics.mean(learned_deltas):.6f}")
    print(f"learned_delta_pct_mean={statistics.mean(learned_delta_pcts):.3f}")
    print(f"learned_wins={learned_wins} learned_ties={learned_ties} learned_losses={len(results) - learned_wins - learned_ties}")

    print("per_instance:")
    for r in results:
        print(
            {
                "seed": r.seed,
                "heuristic_length": round(r.heuristic_length, 6),
                "attention_length": round(r.attention_length, 6),
                "learned_length": round(r.learned_length, 6),
                "delta": round(r.delta, 6),
                "delta_pct": round(r.delta_pct, 3),
                "learned_delta": round(r.learned_delta, 6),
                "learned_delta_pct": round(r.learned_delta_pct, 3),
            }
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
