"""Helper utilities to enumerate Yen-style k-shortest paths."""
from __future__ import annotations

from typing import List

import numpy as np
import networkx as nx


def find_k_shortest_paths(adjacency: np.ndarray,
                          source: int,
                          destination: int,
                          k: int = 2) -> List[List[int]]:
    """Return up to ``k`` simple paths between ``source`` and ``destination``.

    This wraps :func:`networkx.shortest_simple_paths`, which is an implementation
    of Yen's k-shortest paths algorithm when the graph is undirected and
    unweighted.

    Args:
        adjacency: square adjacency matrix for the topology.
        source: index of the source node.
        destination: index of the destination node.
        k: number of candidate paths to return (minimum 1).

    Returns:
        A list of paths where each path is a list of node indices.

    Raises:
        ValueError: if ``k`` is not positive or the source/destination pair is
            invalid.
    """
    if k < 1:
        raise ValueError('k must be at least 1')

    adjacency = np.asarray(adjacency)
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError('Adjacency matrix must be square')

    num_nodes = adjacency.shape[0]
    if not (0 <= source < num_nodes) or not (0 <= destination < num_nodes):
        raise ValueError('Source and destination must lie within node range')

    graph = nx.from_numpy_array(adjacency, create_using=nx.Graph())
    path_generator = nx.shortest_simple_paths(graph, source, destination, weight=None)

    paths: List[List[int]] = []
    try:
        for _ in range(k):
            paths.append(list(next(path_generator)))
    except nx.NetworkXNoPath as exc:
        raise ValueError('No path exists between the requested nodes') from exc
    except StopIteration:
        # Fewer than k paths exist; return the ones we have
        pass

    return paths
