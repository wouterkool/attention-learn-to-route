from __future__ import annotations

import argparse
import torch

from .order_dataset import OrderingDataset, build_ordering_samples
from .order_model import TransitRouteOrderer
from .train_order_model import _evaluate


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the learned route-ordering model")
    parser.add_argument("--checkpoint", type=str, default="outputs/transit_orderer.pt")
    parser.add_argument("--instances", type=int, default=20)
    parser.add_argument("--base-seed", type=int, default=9000)
    parser.add_argument("--num-hubs", type=int, default=3)
    args = parser.parse_args()

    samples = build_ordering_samples(args.instances, base_seed=args.base_seed, num_hubs=args.num_hubs)
    dataset = OrderingDataset(samples)

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model = TransitRouteOrderer(hidden_dim=ckpt.get("args", {}).get("hidden_dim", 96))
    model.load_state_dict(ckpt["model"])
    device = torch.device("cpu")
    metrics = _evaluate(model, dataset, device)
    print(metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

