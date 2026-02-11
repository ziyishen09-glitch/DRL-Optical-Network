from __future__ import annotations

import logging
from typing import Iterable

from ..net import Network


logger = logging.getLogger(__name__)


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
            logger.info('lightpath %s expired after %.3f seconds', lightpath.id, elapsed_time)
            net.t.remove_lightpath_by_id(lightpath.id)

    for edge in net.get_edges():
        if len(edge) < 2:
            continue
        i, j = edge[0], edge[1]
        for w in range(net.nchannels):
            timer = net.t[i][j][w]
            if timer > elapsed_time:
                net.t[i][j][w] = timer - elapsed_time
            else:
                net.t[i][j][w] = 0
                net.n[i][j][w] = 1
            net.t[j][i][w] = net.t[i][j][w]
            net.n[j][i][w] = net.n[i][j][w]

