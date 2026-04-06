"""Dijkstra shortest path algorithm as routing strategy

"""

from typing import List

import math
import numpy as np
import networkx as nx
import logging

# logger used for debug tracing. Simulator configures this logger to write to
# a file when args.debug_dijkstra is enabled so debug output doesn't get
# overwritten by simulator's dynamic progress prints.
_dij_logger = logging.getLogger('rwa_dijkstra_debug')


def dijkstra(mat: np.ndarray, s: int, d: int, debug: bool = False) -> List[int]:
    """Dijkstra routing algorithm

    Args:
        mat: Network's adjacency matrix graph
        s: source node index
        d: destination node index
        debug: when True, print step-by-step internal state for tracing

    Returns:
        :obj:`list` of :obj:`int`: sequence of router indices encoding a path

    """
    if s < 0 or d < 0:
        raise ValueError('Source nor destination nodes cannot be negative')
    elif s >= mat.shape[0] or d >= mat.shape[0]:
        raise ValueError('Source nor destination nodes should exceed '
                         'adjacency matrix dimensions')

    # Fast path: delegate to networkx when no debug tracing requested
    if not debug:
        G = nx.from_numpy_array(mat, create_using=nx.Graph())
        hops, path = nx.bidirectional_dijkstra(G, s, d, weight='weight')
        return path
    else:
        # Debug path: build the same NetworkX graph as the fast path and
        # implement Dijkstra over that graph so debug uses identical input
        # semantics (edge presence and 'weight' attribute).
        G = nx.from_numpy_array(mat, create_using=nx.Graph())
        n = G.number_of_nodes()
        inf = math.inf
        dist = [inf] * n
        prev = [None] * n
        visited = [False] * n
        dist[s] = 0.0

        _dij_logger.debug('Dijkstra debug: from %s to %s, n=%d', s, d, n)
        _dij_logger.debug('Initial dist: %s', dist)

        for _ in range(n):
            # select the unvisited node with the smallest tentative distance
            u = None
            best = inf
            for i in range(n):
                if not visited[i] and dist[i] < best:
                    best = dist[i]
                    u = i
            if u is None or dist[u] == inf:
                _dij_logger.debug('No more reachable nodes, stopping.')
                break
            visited[u] = True
            _dij_logger.debug('Select node %s with dist %s', u, dist[u])
            if u == d:
                _dij_logger.debug('Reached destination node.')
                break

            # relax edges from u using NetworkX adjacency (so attributes like
            # 'weight' are respected exactly as in the fast path)
            for v, data in G[u].items():
                if visited[v]:
                    continue
                # read weight attribute; default to 1 if missing
                w = data.get('weight', 1)
                try:
                    w = float(w)
                except Exception:
                    continue
                # treat non-positive weight as no edge
                if w <= 0:
                    continue
                alt = dist[u] + w
                _dij_logger.debug('  examine edge %s->%s weight=%s, alt=%s, current dist[%s]=%s', u, v, w, alt, v, dist[v])
                if alt < dist[v]:
                    dist[v] = alt
                    prev[v] = u
                    _dij_logger.debug('    relax: dist[%s] -> %s, prev[%s] -> %s', v, alt, v, u)

        # reconstruct path
        if dist[d] == inf:
            _dij_logger.debug('No path from %s to %s (unreachable).', s, d)
            return []
        path = []
        cur = d
        while cur is not None:
            path.append(cur)
            cur = prev[cur]
        path.reverse()
        _dij_logger.debug('Finished: shortest distance %s, path %s', dist[d], path)
        return path
