from __future__ import annotations

import math
from typing import Iterable, List, Sequence, Tuple

import numpy as np


Coordinate = Tuple[float, float]


def route_length(coords: np.ndarray, order: Sequence[int], hub: Coordinate) -> float:
    if len(order) == 0:
        return 0.0
    current = np.asarray(hub, dtype=np.float64)
    total = 0.0
    for idx in order:
        nxt = coords[idx]
        total += float(np.linalg.norm(nxt - current))
        current = nxt
    total += float(np.linalg.norm(np.asarray(hub, dtype=np.float64) - current))
    return total


def two_opt_route(
    coords: np.ndarray,
    hub: Coordinate,
    initial_order: Sequence[int],
) -> Tuple[List[int], float]:
    order = list(initial_order)
    if len(order) <= 2:
        return order, route_length(coords, order, hub)

    best_length = route_length(coords, order, hub)
    improved = True
    while improved:
        improved = False
        for i in range(len(order) - 1):
            for j in range(i + 1, len(order)):
                candidate = order[:i] + list(reversed(order[i : j + 1])) + order[j + 1 :]
                cand_length = route_length(coords, candidate, hub)
                if cand_length + 1e-12 < best_length:
                    order = candidate
                    best_length = cand_length
                    improved = True
                    break
            if improved:
                break
    return order, best_length

