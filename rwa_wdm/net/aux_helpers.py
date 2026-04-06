"""Helper utilities for auxiliary-graph cross-mapping checks.

Provides a small helper `map_to_other_aux(net, lightpath)` that, given a
`net` instance (expected to be an auxgraph d2 instance) and a `Lightpath`
already expanded on the d2 -> physical mapping (`lightpath.mapped_virtual_route`),
returns True if the corresponding d1 paths contain any d1-level virtual hops.

This is intentionally defensive: it tolerates missing mappings and falls
back to conservative False when data isn't available.
"""

from typing import Any


def map_to_other_aux(net: Any, lightpath: Any) -> bool:
    """Return True if d1 (the other aux graph) contains virtual hops for
    any of the virtual hops used by `lightpath`.

    Args:
        net: network instance (typically auxgraph_aux_d2)
        lightpath: Lightpath instance whose `mapped_virtual_route` lists the
                   physical subpaths replacing d2 virtual hops.

    Returns:
        bool: True if any corresponding d1 path contains d1 virtual edges.
    """
    # defensive early exits
    if lightpath is None:
        return False

    mapped_virtual_route = getattr(lightpath, 'mapped_virtual_route', None)
    if not mapped_virtual_route:
        return False

    # get mapping d2 -> d1 paths if the net provides it
    try:
        d1_map = net.virtual_adjacency2d1_path()
    except Exception:
        d1_map = {}

    # instantiate an auxgraph d1 topology to inspect d1's own aux mapping
    try:
        from .auxgraph_aux_d1 import auxgraph_aux_d1 as auxgraph_aux_d1_class

        aux1 = auxgraph_aux_d1_class(net.nchannels)
        aux1_map = aux1.virtual_adjacency2physical_path()
    except Exception:
        aux1_map = {}

    # For every physical subpath used for a d2 virtual hop, recover its
    # endpoints (first and last nodes) which correspond to the original
    # virtual pair key, and check the d1 mapping for virtual hops.
    for phys in mapped_virtual_route:
        if not phys or len(phys) < 2:
            continue
        key = (phys[0], phys[-1])
        d1_path = d1_map.get(key) if d1_map else None
        if not d1_path:
            continue
        # check each hop in the d1 path for presence in aux1_map
        for i in range(len(d1_path) - 1):
            a, b = d1_path[i], d1_path[i + 1]
            if (a, b) in aux1_map or (b, a) in aux1_map:
                return True

    return False
