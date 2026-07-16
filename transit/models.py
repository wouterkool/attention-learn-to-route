from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np


Coordinate = Tuple[float, float]


@dataclass(frozen=True)
class PassengerRequest:
    request_id: str
    origin: Coordinate
    destination: Coordinate
    request_time: float = 0.0
    passenger_count: int = 1
    time_window: Optional[Tuple[float, float]] = None
    priority: float = 1.0


@dataclass(frozen=True)
class RequestCluster:
    cluster_id: int
    request_ids: List[str]
    center: Coordinate
    demand: int
    radius: float
    temporal_span: float
    bbox: Tuple[float, float, float, float]
    compactness: float
    score: float


@dataclass(frozen=True)
class HubProposal:
    hub_id: str
    cluster_id: int
    location: Coordinate
    request_ids: List[str] = field(default_factory=list)
    demand: int = 0
    radius: float = 0.0
    score: float = 0.0
    access_cost: float = 0.0

