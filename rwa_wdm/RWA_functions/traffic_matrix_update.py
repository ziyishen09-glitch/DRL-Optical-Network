from __future__ import annotations

import logging
from typing import Iterable

import numpy as np

from ..net import Network


logger = logging.getLogger(__name__)


def _get_edge_indices(net: Network) -> tuple[np.ndarray, np.ndarray]:
    """Return cached source/destination index arrays for physical edges."""
    cached = getattr(net, '_edge_index_cache', None)
    if cached is not None:
        return cached

    src: list[int] = []
    dst: list[int] = []
    for edge in net.get_edges():
        if len(edge) < 2:
            continue
        try:
            i = int(edge[0])
            j = int(edge[1])
        except Exception:
            continue
        src.append(i)
        dst.append(j)

    src_idx = np.asarray(src, dtype=np.int64)
    dst_idx = np.asarray(dst, dtype=np.int64)
    cache_value = (src_idx, dst_idx)
    setattr(net, '_edge_index_cache', cache_value)
    return cache_value


def advance_traffic_matrix(net: Network, elapsed_time: float) -> None:
    """Move time forward on ``net`` by ``elapsed_time`` seconds.

    This updates every wavelength timer and frees channels whose remaining
    holding time reaches zero, keeping the availability matrix synchronized.
    """

    if elapsed_time <= 0:
        return

    for lightpath in net.t.lightpaths[:]:
        remaining = lightpath.holding_time
        if remaining > elapsed_time:
            lightpath.holding_time = remaining - elapsed_time
        else:
            net.t.remove_lightpath_by_id(lightpath.id)

    src_idx, dst_idx = _get_edge_indices(net)
    if src_idx.size == 0:
        return

    nch = int(net.nchannels)
    # Vectorized update for all physical edges and wavelengths.
    edge_timers = net.t[src_idx, dst_idx, :nch]
    remaining = edge_timers - float(elapsed_time)
    occupied_mask = remaining > 0.0
    updated_timers = np.where(occupied_mask, remaining, 0.0).astype(net.t.dtype, copy=False)
    availability_mask = ~occupied_mask

    net.t[src_idx, dst_idx, :nch] = updated_timers
    net.n[src_idx, dst_idx, :nch] = availability_mask
    # Keep symmetry constraints in sync.
    net.t[dst_idx, src_idx, :nch] = updated_timers
    net.n[dst_idx, src_idx, :nch] = availability_mask

