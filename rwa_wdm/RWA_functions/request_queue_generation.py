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
        # holding_time = sample_holding_time(holding_mean, rng)
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
    lam = float(load) / 1.5
    generator = rng or random
    interarrival = generator.expovariate(lam)
    return float(max(0.0, interarrival))


def sample_holding_time(holding_mean: float, rng: Optional[random.Random] = None) -> float:
    """Sample holding times using an exponential distribution around the provided mean."""
    generator = rng or random
    mean_value = max(0.0, float(holding_mean))
    if mean_value <= 0.0:
        return 1.0
    rate = 1.0 / mean_value
    holding_time = generator.expovariate(rate)
    return float(max(1.0, holding_time))