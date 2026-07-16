import math
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .models import HubProposal, PassengerRequest, RequestCluster


def _request_matrix(requests: Sequence[PassengerRequest]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    origins = np.asarray([req.origin for req in requests], dtype=np.float64)
    weights = np.asarray([max(int(req.passenger_count), 1) for req in requests], dtype=np.float64)
    times = np.asarray([float(req.request_time) for req in requests], dtype=np.float64)
    ids = np.asarray([req.request_id for req in requests], dtype=object)
    return origins, weights, times, ids


def _pairwise_sq_dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    diff = a[:, None, :] - b[None, :, :]
    return np.sum(diff * diff, axis=-1)


def _weighted_kmeans(points: np.ndarray, weights: np.ndarray, k: int, seed: int = 1234, max_iter: int = 50) -> np.ndarray:
    if len(points) == 0:
        raise ValueError("Cannot cluster an empty request set")

    k = max(1, min(int(k), len(points)))
    rng = np.random.default_rng(seed)

    centers = np.empty((k, points.shape[1]), dtype=np.float64)
    first_idx = int(rng.choice(len(points), p=weights / weights.sum()))
    centers[0] = points[first_idx]

    closest_sq = _pairwise_sq_dist(points, centers[:1]).reshape(-1)
    for i in range(1, k):
        probs = closest_sq * weights
        if float(probs.sum()) <= 0.0:
            idx = int(rng.integers(0, len(points)))
        else:
            probs = probs / probs.sum()
            idx = int(rng.choice(len(points), p=probs))
        centers[i] = points[idx]
        closest_sq = np.minimum(closest_sq, _pairwise_sq_dist(points, centers[i : i + 1]).reshape(-1))

    for _ in range(max_iter):
        dist = _pairwise_sq_dist(points, centers)
        assign = np.argmin(dist, axis=1)
        new_centers = centers.copy()
        for i in range(k):
            mask = assign == i
            if not np.any(mask):
                new_centers[i] = points[int(rng.integers(0, len(points)))]
                continue
            w = weights[mask][:, None]
            new_centers[i] = np.sum(points[mask] * w, axis=0) / np.sum(w)
        if np.allclose(new_centers, centers):
            break
        centers = new_centers

    return centers


def estimate_cluster_count(num_requests: int, average_cluster_size: int = 18) -> int:
    if num_requests <= 0:
        return 0
    return max(1, int(math.ceil(num_requests / float(average_cluster_size))))


def cluster_requests(
    requests: Sequence[PassengerRequest],
    num_clusters: Optional[int] = None,
    seed: int = 1234,
) -> List[RequestCluster]:
    origins, weights, times, ids = _request_matrix(requests)
    if num_clusters is None:
        num_clusters = estimate_cluster_count(len(requests))

    centers = _weighted_kmeans(origins, weights, num_clusters, seed=seed)
    dist = _pairwise_sq_dist(origins, centers)
    assign = np.argmin(dist, axis=1)

    clusters: List[RequestCluster] = []
    for cluster_id in range(len(centers)):
        mask = assign == cluster_id
        if not np.any(mask):
            continue

        cluster_points = origins[mask]
        cluster_weights = weights[mask]
        cluster_times = times[mask]
        cluster_ids = ids[mask].tolist()

        demand = int(cluster_weights.sum())
        center = np.average(cluster_points, axis=0, weights=cluster_weights)
        sq_dists = np.sum((cluster_points - center[None, :]) ** 2, axis=1)
        radius = float(np.sqrt(np.max(sq_dists))) if len(sq_dists) else 0.0
        compactness = float(np.average(np.sqrt(sq_dists), weights=cluster_weights)) if len(sq_dists) else 0.0
        temporal_span = float(cluster_times.max() - cluster_times.min()) if len(cluster_times) else 0.0

        min_x = float(cluster_points[:, 0].min())
        min_y = float(cluster_points[:, 1].min())
        max_x = float(cluster_points[:, 0].max())
        max_y = float(cluster_points[:, 1].max())
        bbox = (min_x, min_y, max_x, max_y)

        score = demand / (1.0 + compactness + 0.5 * temporal_span)

        clusters.append(
            RequestCluster(
                cluster_id=cluster_id,
                request_ids=cluster_ids,
                center=(float(center[0]), float(center[1])),
                demand=demand,
                radius=radius,
                temporal_span=temporal_span,
                bbox=bbox,
                compactness=compactness,
                score=float(score),
            )
        )

    clusters.sort(key=lambda c: (-c.score, c.cluster_id))
    return clusters


def build_hub_proposals(
    requests: Sequence[PassengerRequest],
    num_hubs: Optional[int] = None,
    seed: int = 1234,
) -> List[HubProposal]:
    clusters = cluster_requests(requests, num_clusters=num_hubs, seed=seed)
    proposals: List[HubProposal] = []

    for cluster in clusters:
        access_cost = cluster.compactness + 0.5 * cluster.temporal_span
        proposals.append(
            HubProposal(
                hub_id=f"hub_{cluster.cluster_id}",
                cluster_id=cluster.cluster_id,
                location=cluster.center,
                request_ids=list(cluster.request_ids),
                demand=cluster.demand,
                radius=cluster.radius,
                score=cluster.score,
                access_cost=float(access_cost),
            )
        )

    proposals.sort(key=lambda h: (-h.score, h.hub_id))
    return proposals

