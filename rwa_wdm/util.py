import logging
import json
import os
import re
from argparse import ArgumentTypeError, Namespace
from collections import deque
from typing import Iterable

import numpy as np

from .shortest_path.k_shortest import find_k_shortest_paths

logger = logging.getLogger(__name__)


def validate_args(args: Namespace) -> None:
    """Validates arguments passed via command line through argparse module

    Args:
        args: `Namespace` object from argparse module

    Raises:
        ValueError: if a combination of both routing and wavelength assignment
            algorithms is not specified nor a single RWA algorithm as one.


    """
    runner = getattr(args, 'runner', None)
    if runner in {'train', 'eval'}:
        return

    if args.rwa is None:
        if args.r is None and args.w is None:
            raise ValueError('The use of either --rwa flag or both -r and -w '
                             'combined is required.')
        elif not (args.r and args.w):
            raise ValueError('Flags -r and -w should be set in combination.')
    elif args.rwa is not None:
        if args.r is not None or args.w is not None:
            raise ValueError('Set either --rwa flag alone or both -r and -w '
                             'flags combined.')
    if args.num_sim < 1:
        raise ValueError('Expect a positive integer as number of simulations.')


def normalize_link_pair(u: int | str, v: int | str) -> tuple[int, int]:
    """Return an ordered node-pair tuple (min, max) as ints."""
    try:
        ui = int(u)
        vi = int(v)
    except Exception as exc:  # pragma: no cover - defensive
        raise ValueError('Invalid node identifiers %r and %r' % (u, v)) from exc
    return (min(ui, vi), max(ui, vi))


def parse_link_argument(value: str) -> tuple[int, int]:
    """Argparse type: parse an edge descriptor such as "2,4" or "2-4"."""
    if not isinstance(value, str):
        raise ArgumentTypeError('Failure link must be provided as a string in the form "src,dst" or "src-dst"')
    stripped = value.strip()
    if not stripped:
        raise ArgumentTypeError('Failure link cannot be empty')
    parts = re.split(r'[\s,;:-]+', stripped)
    if len(parts) != 2:
        raise ArgumentTypeError('Failure link must contain exactly two node identifiers separated by , or -')
    try:
        return normalize_link_pair(parts[0], parts[1])
    except ValueError as exc:
        raise ArgumentTypeError(str(exc)) from exc


def coerce_link_argument(value: str | Iterable[int] | tuple[int, int] | None) -> tuple[int, int] | None:
    """Ensure a normalized link tuple regardless of input type."""
    if value is None:
        return None
    if isinstance(value, tuple) or isinstance(value, list):
        if len(value) < 2:
            raise ValueError('Failure link tuple must have at least two elements')
        return normalize_link_pair(value[0], value[1])
    if isinstance(value, str):
        return parse_link_argument(value)
    raise ValueError('Unsupported failure link value %r' % (value,))


def route_contains_link(route: Iterable[int | str] | None, link: tuple[int, int] | None) -> bool:
    """Return True if the ordered route contains the given undirected link."""
    if not route or link is None:
        return False
    normalized_link = normalize_link_pair(link[0], link[1])
    it = iter(route)
    try:
        prev = int(next(it))
    except StopIteration:
        return False
    except Exception:
        prev = None
    for node in it:
        try:
            current = int(node)
        except Exception:
            prev = None
            continue
        if prev is not None and normalize_link_pair(prev, current) == normalized_link:
            return True
        prev = current
    return False


def build_adjacency_list(net) -> list[list[int]]:
    """Return adjacency lists derived from a network's adjacency matrix."""
    mat = getattr(net, 'a', None)
    if mat is None:
        return []
    n = int(mat.shape[0])
    adjacency: list[list[int]] = [[] for _ in range(n)]
    for i in range(n):
        try:
            row = mat[i]
        except Exception:
            continue
        for j, val in enumerate(row):
            try:
                weight = float(val)
            except Exception:
                weight = 0.0
            if weight != 0.0:
                adjacency[i].append(j)
    return adjacency


def failure_link_impact(src: int, dst: int, adjacency: list[list[int]], failure_link: tuple[int, int]) -> bool:
    """Return True if the shortest path (with the link re-added) uses that link."""
    if failure_link is None or not adjacency:
        return False
    n = len(adjacency)
    if not (0 <= src < n and 0 <= dst < n):
        return False
    if src == dst:
        return False
    u, v = failure_link
    if not (0 <= u < n and 0 <= v < n):
        return False
    visited = [False] * n
    parent = [-1] * n
    queue = deque([src])
    visited[src] = True
    reached = False
    while queue:
        node = queue.popleft()
        if node == dst:
            reached = True
            break
        for neighbor in adjacency[node]:
            if not visited[neighbor]:
                visited[neighbor] = True
                parent[neighbor] = node
                queue.append(neighbor)
        if node == u and not visited[v]:
            visited[v] = True
            parent[v] = node
            queue.append(v)
        elif node == v and not visited[u]:
            visited[u] = True
            parent[u] = node
            queue.append(u)
    if not reached:
        return False
    path: list[int] = []
    cursor = dst
    while cursor != -1:
        path.append(cursor)
        cursor = parent[cursor]
    path.reverse()
    return route_contains_link(path, failure_link)


def build_failure_link_lookup_ksp(
    adjacency: list[list[int]] | np.ndarray,
    failure_link: tuple[int, int],
    k_paths: int,
) -> dict[tuple[int, int], bool]:
    """Precompute whether each (src, dst) has a KSP route touching failure_link."""
    mat = np.asarray(adjacency, dtype=np.float32)
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError('Adjacency matrix must be square to build failure lookup.')
    n = int(mat.shape[0])
    k_paths = max(1, int(k_paths))

    lookup: dict[tuple[int, int], bool] = {}
    for src in range(n):
        for dst in range(n):
            if src == dst:
                continue
            impacted = False
            try:
                candidates = find_k_shortest_paths(mat, src, dst, k_paths)
            except Exception:
                candidates = []
            for route in candidates:
                if route_contains_link(route, failure_link):
                    impacted = True
                    break
            lookup[(src, dst)] = impacted
    return lookup


def save_failure_link_lookup(
    file_path: str,
    lookup: dict[tuple[int, int], bool],
    *,
    topology: str | None = None,
    channels: int | None = None,
    failure_link: tuple[int, int] | None = None,
    k_paths: int | None = None,
) -> None:
    """Persist failure-link lookup table to JSON."""
    payload = {
        'metadata': {
            'topology': topology,
            'channels': channels,
            'failure_link': list(failure_link) if failure_link is not None else None,
            'k_paths': k_paths,
        },
        'lookup': {
            f'{int(src)}-{int(dst)}': bool(flag)
            for (src, dst), flag in lookup.items()
        },
    }
    out_dir = os.path.dirname(file_path)
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    if os.path.isdir(file_path):
        raise IsADirectoryError(
            'Output path points to an existing directory, expected a file: %s' % file_path
        )
    with open(file_path, 'w', encoding='utf-8') as fh:
        json.dump(payload, fh, ensure_ascii=True)


def load_failure_link_lookup(file_path: str) -> tuple[dict[tuple[int, int], bool], dict]:
    """Load failure-link lookup table from JSON."""
    with open(file_path, 'r', encoding='utf-8') as fh:
        payload = json.load(fh)

    if isinstance(payload, dict) and 'lookup' in payload:
        raw_lookup = payload.get('lookup', {})
        metadata = payload.get('metadata', {}) if isinstance(payload.get('metadata'), dict) else {}
    elif isinstance(payload, dict):
        # Backward-compatible plain-map format.
        raw_lookup = payload
        metadata = {}
    else:
        raise ValueError('Invalid failure lookup format in %s' % file_path)

    lookup: dict[tuple[int, int], bool] = {}
    for key, value in raw_lookup.items():
        if not isinstance(key, str):
            continue
        parts = key.split('-')
        if len(parts) != 2:
            continue
        try:
            src = int(parts[0])
            dst = int(parts[1])
        except Exception:
            continue
        lookup[(src, dst)] = bool(value)
    return lookup, metadata
