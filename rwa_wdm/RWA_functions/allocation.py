from __future__ import annotations

from typing import Optional, Sequence

from ..net import Lightpath, Network
from ..rwa.wlassignment.ff import first_fit


def allocate_lightpath(
    net: Network,
    route: Sequence[int],
    *,
    holding_time: int = 10,
    enable_new_ff: bool = True,
) -> Optional[dict]:
    """Run wavelength assignment on ``route`` and update ``net`` if successful."""

    if net is None or len(route) < 2:
        return None

    w_list = first_fit(net, route, False, enable_new_ff=enable_new_ff)
    if not w_list:
        return None

    base_w = next((w for w in w_list if isinstance(w, int) and w >= 0), 0)
    candidate = Lightpath(list(route), base_w)
    try:
        candidate.w_list = list(w_list)
    except Exception:
        pass

    candidate.holding_time = holding_time
    net.t.add_lightpath(candidate)

    links_list = list(candidate.links)
    if not links_list:
        links_list = []

    w_list = getattr(candidate, 'w_list', None) or []
    holding_time_value = candidate.holding_time
    for idx, w in enumerate(w_list):
        if idx >= len(links_list):
            break
        if w is None or not isinstance(w, int) or w < 0 or w >= getattr(net, 'nchannels', 0):
            continue
        i, j = links_list[idx]
        net.n[i][j][w] = 0
        net.t[i][j][w] = holding_time_value
        net.n[j][i][w] = net.n[i][j][w]
        net.t[j][i][w] = net.t[i][j][w]

    return {
        'lightpath': candidate,
        'links': links_list,
        'holding_time': candidate.holding_time,
        'w_list': getattr(candidate, 'w_list', None),
    }
