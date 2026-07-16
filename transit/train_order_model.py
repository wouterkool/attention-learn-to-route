from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from typing import List

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from .order_dataset import OrderingDataset, build_ordering_samples, collate_ordering_samples
from .order_model import TransitRouteOrderer
from .order_utils import route_length


def _evaluate(model: TransitRouteOrderer, dataset: OrderingDataset, device: torch.device) -> dict:
    model.eval()
    heuristic_lengths = []
    learned_lengths = []
    oracle_lengths = []
    with torch.no_grad():
        for sample in dataset.samples:
            hub = torch.tensor(sample.hub[None, :], dtype=torch.float32, device=device)
            coords = torch.tensor(sample.coords[None, :, :], dtype=torch.float32, device=device)
            demands = torch.tensor(sample.demands[None, :], dtype=torch.float32, device=device)
            capacity = torch.tensor([[sample.capacity / 40.0]], dtype=torch.float32, device=device)
            role = torch.tensor([sample.role], dtype=torch.long, device=device)
            order = model.greedy_order(hub, coords, demands, capacity, role).cpu().tolist()
            learned_lengths.append(route_length(sample.coords, order, tuple(sample.hub)))
            heuristic_lengths.append(sample.heuristic_length)
            oracle_lengths.append(sample.optimized_length)
    return {
        "heuristic_mean": float(np.mean(heuristic_lengths)),
        "learned_mean": float(np.mean(learned_lengths)),
        "oracle_mean": float(np.mean(oracle_lengths)),
        "delta_mean": float(np.mean(np.asarray(learned_lengths) - np.asarray(heuristic_lengths))),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a transit-specific learned route-ordering model")
    parser.add_argument("--train-instances", type=int, default=300)
    parser.add_argument("--val-instances", type=int, default=60)
    parser.add_argument("--base-seed", type=int, default=5000)
    parser.add_argument("--num-hubs", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--save-path", type=str, default="outputs/transit_orderer.pt")
    args = parser.parse_args()

    train_samples = build_ordering_samples(args.train_instances, base_seed=args.base_seed, num_hubs=args.num_hubs)
    val_samples = build_ordering_samples(args.val_instances, base_seed=args.base_seed + 10000, num_hubs=args.num_hubs)
    train_dataset = OrderingDataset(train_samples)
    val_dataset = OrderingDataset(val_samples)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransitRouteOrderer(hidden_dim=args.hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val = None
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_ordering_samples)
        running_loss = 0.0
        steps = 0
        for batch in loader:
            loss = 0.0
            for sample in batch:
                hub = torch.tensor(sample.hub[None, :], dtype=torch.float32, device=device)
                coords = torch.tensor(sample.coords[None, :, :], dtype=torch.float32, device=device)
                demands = torch.tensor(sample.demands[None, :], dtype=torch.float32, device=device)
                capacity = torch.tensor([[sample.capacity / 40.0]], dtype=torch.float32, device=device)
                role = torch.tensor([sample.role], dtype=torch.long, device=device)
                loss = loss + model.step_loss(hub, coords, demands, sample.target_order, capacity, role)
            loss = loss / float(len(batch))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running_loss += float(loss.item())
            steps += 1

        val_metrics = _evaluate(model, val_dataset, device)
        avg_train_loss = running_loss / max(steps, 1)
        print(
            {
                "epoch": epoch,
                "train_loss": round(avg_train_loss, 6),
                **{k: round(v, 6) for k, v in val_metrics.items()},
            }
        )

        score = val_metrics["learned_mean"]
        if best_val is None or score < best_val:
            best_val = score
            torch.save(
                {
                    "model": model.state_dict(),
                    "args": vars(args),
                    "val_metrics": val_metrics,
                },
                args.save_path,
            )

    print(f"Saved best checkpoint to {args.save_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

