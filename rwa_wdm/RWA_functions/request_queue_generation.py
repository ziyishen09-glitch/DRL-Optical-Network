from __future__ import annotations

import random
from collections import deque
from typing import Deque, Dict, Iterable, List, Optional, Sequence


Request = Dict[str, float | int]


def generate_request_queue(
    num_requests: int,
    load: float,
    nodes: Sequence[int],
    *,
    seed: Optional[int] = None,
    holding_mean: float = 10.0,
) -> Deque[Request]:
    """Create a queue of requests following a Poisson-like arrival process."""

    if num_requests <= 0 or not nodes:
        return deque()

    rng = random.Random(seed)
    current_time = 0.0
    queue: Deque[Request] = deque()

    for idx in range(num_requests):
        src, dst = rng.sample(nodes, 2)
        inter_arrival = sample_interarrival(load, rng)
        current_time += inter_arrival
        holding_time = 10
        queue.append({
            'id': idx,
            'source': src,
            'destination': dst,
            'arrival_time': current_time,
            'holding_time': holding_time,
        })

    return queue


def sample_interarrival(load: float, rng: Optional[random.Random] = None) -> float:
    """Sample Erlang-style interarrival times (λ = load / 10)."""
    lam = float(load) / 10.0
    generator = rng or random
    interarrival = generator.expovariate(lam)
    return float(max(0.0, interarrival))