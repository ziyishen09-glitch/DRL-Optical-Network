"""Implements network topologies

"""

__author__ = 'Cassio Batista'

import logging
from itertools import count
from operator import itemgetter
from random import randint
from typing import Iterable, List, Tuple, Dict

import numpy as np
import matplotlib.pyplot as plt

__all__ = (
    'Lightpath',
    'AdjacencyMatrix',
    'WavelengthAvailabilityMatrix',
    'TrafficMatrix',
    'Network',
)

logger = logging.getLogger(__name__)


class Lightpath(object):
    """Emulates a lightpath composed by a route and a wavelength channel

    Lightpath is pretty much a regular path, but must also specify a wavelength
    index, since WDM optical networks span multiple wavelength channels over a
    single fiber link on the topology.

    A Lightpath object also store a holding time parameter, which is set along
    the simulation to specify how long the connection may be alive and running
    on network links, and therefore taking up space in the traffic matrix,
    before it finally terminates and resources are deallocated.

    Args:
        route: a liste of nodes encoded as integer indices
        wavelength: a single number representing the wavelength channel index

    """

    # https://stackoverflow.com/questions/8628123/counting-instances-of-a-class
    _ids = count(0)

    def __init__(self, route: List[int], wavelength: int):
        # New optional flag `contains_virtual` is supported by RWA layer
        # to indicate the returned route used one or more auxiliary hops.
        self._id: int = next(self._ids)
        self._route: List[int] = route
        self._wavelength: int = wavelength
        self._holding_time: float = 0.0
        self._contains_virtual: bool = False
        # optional container mapping virtual (aux) hops to the physical
        # paths they represent. This mirrors the `contains_virtual` flag and
        # can hold a list of physical path lists, e.g. [[p0,p1,...], ...]
        # Set by RWA routines when an expanded/aux route was produced.
        self._mapped_virtual_route: List[List[int]] | None = None
        # optional per-link wavelength assignment: list with one entry per
        # link in the route. When present, this takes precedence over the
        # single-channel value stored in `_wavelength` for allocation and
        # release operations.
        self._w_list: List[int] | None = None

    @property
    def id(self) -> int:
        """A unique identifier to the Lightpath object"""
        return self._id

    @property
    def r(self) -> List[int]:
        """The path as a sequence of router indices"""
        return self._route

    # https://stackoverflow.com/questions/5389507/iterating-over-every-two-elements-in-a-list
    # pairwise: https://docs.python.org/3/library/itertools.html
    @property
    def links(self) -> Iterable[Tuple[int, int]]:
        """Network links as a sequence of pairs of nodes"""
        iterable = iter(self._route)
        while True:
            try:
                yield next(iterable), next(iterable)
            except StopIteration:
                return

    @property
    def w(self) -> int:
        """The wavelength channel index"""
        return self._wavelength

    @property
    def w_list(self) -> List[int] | None:
        """Optional per-link wavelength assignment.

        When present, this is a list of integers where each element is the
        wavelength index used on the corresponding link of the route. The
        list length should match len(self.r) - 1 (number of links).
        """
        return getattr(self, '_w_list', None)

    @w_list.setter
    def w_list(self, val: List[int] | None) -> None:
        if val is None:
            self._w_list = None
        else:
            try:
                self._w_list = [int(x) for x in val]
            except Exception:
                # fallback: wrap single int into list
                try:
                    self._w_list = [int(val)]
                except Exception:
                    self._w_list = None

    @property
    def holding_time(self) -> float:
        """Time that the lightpath remains occupying net resources"""
        return self._holding_time

    @holding_time.setter
    def holding_time(self, time: float) -> None:
        self._holding_time = time

    def __len__(self):
        return len(self.r)

    def __str__(self):
        return '%s %d' % (self._route, self._wavelength)

    @property
    def contains_virtual(self) -> bool:
        """Whether the lightpath route included auxiliary (virtual) hops."""
        return getattr(self, '_contains_virtual', False)

    @contains_virtual.setter
    def contains_virtual(self, val: bool) -> None:
        self._contains_virtual = bool(val)

    @property
    def mapped_virtual_route(self) -> List[List[int]] | None:
        """If present, a container mapping the virtual hops used in the
        route to their corresponding physical subpaths.

        Structure: a list where each element is a list[int] describing a
        physical path (sequence of node indices) that replaces one virtual
        hop in the expanded route. May be None when no virtual hops were
        present or when not set by the RWA layer.
        """
        return getattr(self, '_mapped_virtual_route', None)

    @mapped_virtual_route.setter
    def mapped_virtual_route(self, val: List[List[int]] | None) -> None:
        if val is None:
            self._mapped_virtual_route = None
        else:
            # coerce to list-of-lists for safety
            try:
                self._mapped_virtual_route = [list(p) for p in val]
            except Exception:
                # fallback: wrap single route in a list
                self._mapped_virtual_route = [list(val)]


class AdjacencyMatrix(np.ndarray):
    """Boolean 2D matrix that stores network neighbourhood info

    The adjacency matrix is basically a binary, bidimensional matrix that
    informs whether two nodes in a network physical topology are neighbours,
    i.e,. share a link connection. This class is a subclass of a NumPy array.

    Args:
        num_nodes: number of nodes in the network, which define a square
            matrix's dimensions

    """

    def __new__(cls, num_nodes: int):
        # use a numeric dtype so adjacency can store edge weights (not only bool)
        arr = np.zeros((num_nodes, num_nodes), dtype=np.float32)
        obj = np.asarray(arr, dtype=np.float32).view(cls)
        #the same as traffic matrix
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return


class WavelengthAvailabilityMatrix(np.ndarray):
    """Boolean 3D matrix that stores network wavelength availability info

    The wavelength availability matrix is a tridimensional, binary matrix that
    stores information on whether a particular wavelength λ is available on an
    optical link (i, j). This class is a subclass of a NumPy array.

    Args:
        num_nodes: number of nodes in the network, which defines two of the
            matrix's dimensions
        num_ch: number of wavelength channels on each link, defining the shape
            of the third dimension of the matrix

    """

    def __new__(cls, num_nodes: int, num_ch: int):
        arr = np.zeros((num_nodes, num_nodes, num_ch))
        obj = np.asarray(arr, dtype=np.bool_).view(cls)

        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return


class TrafficMatrix(np.ndarray):
    """Boolean 3D matrix that stores traffic info

    Args:
        num_nodes: number of nodes in the network, which defines two of the
            matrix's dimensions
        num_ch: number of wavelength channels on each link, defining the shape
            of the third dimension of the matrix

    """

    def __new__(cls, num_nodes: int, num_ch: int):
        arr = np.zeros((num_nodes, num_nodes, num_ch))
        obj = np.asarray(arr, dtype=np.float32).view(cls)

        # set extra parameters
        obj._usage = np.zeros(num_ch, dtype=np.uint16)
        obj._lightpaths = []

        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self._usage = getattr(obj, "_usage", None)
        self._lightpaths = getattr(obj, "_lightpaths", None)

    @property
    def lightpaths(self) -> List[Lightpath]:
        """The list of connections (lightpaths) currently running"""
        return self._lightpaths

    @property
    def nconns(self):
        """The number of connections (lightpaths) currently running"""
        return len(self._lightpaths)

    def add_lightpath(self, lightpath: Lightpath) -> None:
        """Add a lightpath to the list of lightpath

        Args:
            lightpath: a Lightpath instance

        """
        self._lightpaths.append(lightpath)

    # FIXME this seems silly, but...
    # https://stackoverflow.com/questions/9140857/oop-python-removing-class-instance-from-a-list/9140906
    
    def remove_lightpath_by_id(self, _id: int) -> None:
        """Remove a lightpath from the list of currently running connections

        Args:
            _id: the unique identifier of a lightpath

        """
        for i, lightpath in enumerate(self.lightpaths):
            if lightpath.id == _id:
                del self._lightpaths[i]
                break


class Network(object):
    """Network base class

    Hols network properties such as adjacency, wavelength-availability and
    traffic graph matrices, fixed source and destination nodes for all
    connections, number of λ channels per link

    Args:
        num_channels: number of wavelength channels per link
        num_nodes: number of routes along the path
        num_links: number of links along the path, typically `num_nodes` - 1

    """

    def __init__(self,
                 num_channels: int, num_nodes: int, num_links: int) -> None:
        self._num_channels = num_channels
        self._num_nodes = num_nodes
        self._num_links = num_links

        self._n = WavelengthAvailabilityMatrix(self._num_nodes,
                                               self._num_channels)
        self._a = AdjacencyMatrix(self._num_nodes)
        self._t = TrafficMatrix(self._num_nodes, self._num_channels)

        # fill in wavelength availability matrix (original behaviour)
        for edge in self.get_edges():
            # support edge formats: (i, j) or (i, j, weight)
            if len(edge) == 2:
                i, j = edge
            else:
                i, j = edge[0], edge[1]
            for w in range(self._num_channels):
                availability = np.random.choice((0, 1))
                self._n[i][j][w] = availability
                self._n[j][i][w] = self._n[i][j][w] #for symmetry

        # fill in adjacency matrix using only physical edges (get_edges()).
        # Auxiliary edges must not populate the base adjacency/availability
        # structures during initialization; they are used only for routing
        # augmentation at runtime.
        for edge in self.get_edges():
            if len(edge) == 2:
                i, j = edge
                neigh = 1
            else:
                i, j, neigh = edge
            # cast/convert to numeric if needed
            try:
                self._a[i][j] = neigh
            except Exception:
                # fallback: if weight is not directly assignable, coerce to 1
                self._a[i][j] = 1
            self._a[j][i] = self._a[i][j]
                
        # fill in traffic matrix
        # FIXME when updating the traffic matrix via holding time parameter,
        # these random time attributions may seem not the very smart ones,
        # since decreasing values by until_next leads T to be uneven and
        # unbalanced
        for edge in self.get_edges():
            if len(edge) == 2:
                i, j = edge
            else:
                i, j = edge[0], edge[1]
            for w in range(self._num_channels):
                # initialize traffic matrix times: when a wavelength is
                # currently occupied (self._n[i][j][w] == 1), assign a
                # random integer remaining time in [0, 10] so initial
                # allocations are spread over the first 10 slots. When the
                # wavelength is free, time is 0.
                if self._n[i][j][w]:
                    random_time = np.random.randint(1, 11)
                else:
                    random_time = 0
                self._t[i][j][w] = random_time
                self._t[j][i][w] = self._t[i][j][w]

        # initialize Quantum Key Pools (QKP) for every unordered node pair
        # Keys are stored as integer counters per undirected edge (i, j) with
        # i < j. This supports recording keys saved by bypass operations and
        # querying/consuming keys during routing decisions.
        self._qkp_pools: Dict[Tuple[int, int], int] = {}
        for i in range(self._num_nodes):
            for j in range(i + 1, self._num_nodes):
                self._qkp_pools[(i, j)] = 0

        # history log of recorded bypass-saved keys (edge, amount)
        self._qkp_log: List[Tuple[Tuple[int, int], int]] = []

        # history log of QKP consumption events (edge, amount, info)
        # info is an optional dict describing the request that consumed keys
        self._qkp_usage_log: List[Tuple[Tuple[int, int], int, dict]] = []

        # precompute a fast lookup map from any ordered node pair (i,j)
        # to the normalized QKP key (undirected tuple with i <= j). This
        # avoids repeated normalization computations during runtime.
        self._qkp_key_map: Dict[Tuple[int, int], Tuple[int, int]] = {}
        for i in range(self._num_nodes):
            for j in range(self._num_nodes):
                if i == j:
                    continue
                self._qkp_key_map[(i, j)] = (i, j) if i <= j else (j, i)

    # Children are responsible for overriding this method
    def get_edges(self):
        raise NotImplementedError

    # Children are responsible for overriding this method
    def get_nodes_2D_pos(self):
        raise NotImplementedError

    @property
    def n(self) -> np.ndarray:
        """The wavelength availability matrix graph"""
        return self._n

    @property
    def a(self) -> np.ndarray:
        """The adjacency matrix graph"""
        return self._a

    @property
    def t(self) -> np.ndarray:
        """The traffic matrix"""
        return self._t

    @property
    def s(self) -> int:
        """The source node"""
        return self._s

    @property
    def d(self) -> int:
        """The destination node"""
        return self._d

    @property
    def name(self) -> str:
        """The short name tag idenfier of the network topology"""
        return self._name

    @property
    def nchannels(self) -> int:
        """The number of wavelength channels per fiber link"""
        return self._num_channels

    @property
    def nnodes(self) -> int:
        """The number of router nodes (vertices) in the network"""
        return self._num_nodes

    @property
    def nlinks(self) -> int:
        """The number of links (edges) in the network"""
        return self._num_links

    # --- Quantum Key Pool (QKP) API ---------------------------------
    def _normalize_edge(self, edge: Tuple[int, int]) -> Tuple[int, int]:
        """Return the unordered (i, j) tuple used as key for QKP pools.

        Accepts either a 2-tuple (i, j) or a sequence where first two
        elements are node indices. Normalizes so that i < j.
        """
        # Fast-path: if the edge is an ordered pair present in precomputed
        # map, return it directly. Otherwise coerce and fallback to compute.
        try:
            # support sequences (list/tuple) and numeric-like inputs
            i = int(edge[0])
            j = int(edge[1])
        except Exception:
            raise ValueError('edge must be a pair of node indices')

        key = self._qkp_key_map.get((i, j))
        if key is not None:
            return key
        # fallback (shouldn't happen if map was built for all pairs)
        return (i, j) if i <= j else (j, i)

    def add_qkp(self, edge: Tuple[int, int], amount: int = 1) -> None:
        """Add `amount` keys to the QKP pool for `edge` (undirected).

        edge: pair of node indices or sequence with first two elements.
        """
        # inline fast lookup using precomputed map to avoid repeated
        # normalization calls and reduce overhead.
        try:
            i = int(edge[0]); j = int(edge[1])
            key = self._qkp_key_map.get((i, j))
        except Exception:
            key = None
        if key is None:
            # final fallback to normalized pair
            key = self._normalize_edge(edge)
        self._qkp_pools[key] = int(self._qkp_pools.get(key, 0)) + int(amount)

    def use_qkp(self, edge: Tuple[int, int], amount: int = 1) -> bool:
        """Consume `amount` keys from the pool for `edge` if available.

        Returns True if keys were available and consumed, False otherwise.
        """
        try:
            i = int(edge[0]); j = int(edge[1])
            key = self._qkp_key_map.get((i, j))
        except Exception:
            key = None
        if key is None:
            key = self._normalize_edge(edge)
        avail = int(self._qkp_pools.get(key, 0))
        if avail >= amount:
            self._qkp_pools[key] = avail - int(amount)
            return True
        return False

    def get_qkp(self, edge: Tuple[int, int]) -> int:
        """Return the number of keys in the pool for `edge` (undirected)."""
        try:
            i = int(edge[0]); j = int(edge[1])
            key = self._qkp_key_map.get((i, j))
        except Exception:
            key = None
        if key is None:
            key = self._normalize_edge(edge)
        return int(self._qkp_pools.get(key, 0))

    def record_bypass_saved_keys(self, edge: Tuple[int, int], amount: int = 1) -> None:
        """Record keys saved by performing a bypass: increment pool and log it."""
        try:
            i = int(edge[0]); j = int(edge[1])
            key = self._qkp_key_map.get((i, j))
        except Exception:
            key = None
        if key is None:
            key = self._normalize_edge(edge)
        self._qkp_pools[key] = int(self._qkp_pools.get(key, 0)) + int(amount)
        self._qkp_log.append((key, int(amount)))

    @property
    def qkp_pools(self) -> Dict[Tuple[int, int], int]:
        """A copy of the all-pairs QKP pools mapping."""
        return dict(self._qkp_pools)

    def get_qkp_log(self) -> List[Tuple[Tuple[int, int], int]]:
        """Return a copy of the QKP history log (edge, amount)."""
        return list(self._qkp_log)

    def record_qkp_consumption(self, edge: Tuple[int, int], amount: int = 1, info: dict | None = None) -> None:
        """Record an event where keys were consumed from a QKP pool.

        edge: pair of node indices
        amount: number of keys consumed
        info: optional dictionary with extra metadata (e.g., route or call id)
        """
        try:
            i = int(edge[0]); j = int(edge[1])
            key = self._qkp_key_map.get((i, j))
        except Exception:
            key = None
        if key is None:
            key = self._normalize_edge(edge)
        self._qkp_usage_log.append((key, int(amount), dict(info or {})))

    def get_qkp_usage_log(self) -> List[Tuple[Tuple[int, int], int, dict]]:
        """Return a copy of the QKP consumption history log."""
        return list(self._qkp_usage_log)


    def plot_topology(self, bestroute: List[int] = None) -> None:
        """Plots the physical topology in a 2D Cartesian plan

        Args:
            bestroute: a route encoded as a list of router indices to be
                highlighted in red over some network edges

        """
        fig, ax = plt.subplots()
        ax.grid()

        # define vertices or nodes as points in 2D cartesian plan
        # define links or edges as node index ordered pairs in cartesian plan
        try:
            links = self.get_edges()
            aux_links = self.get_aux_edges()
        except Exception:
            links = self.get_edges()
            aux_links = []
            
        nodes = self.get_nodes_2D_pos()
        # build mapping from node index -> coordinates (keys may be strings)
        node_coords = {}
        for key, coord in nodes.items():
            try:
                idx = int(key)
            except Exception:
                continue
            node_coords[idx] = coord
        node_coords = {idx: coord for idx, coord in sorted(node_coords.items())}

        # draw edges before vertices
        for edge in links:
            # support edge formats (i,j) or (i,j,weight)
            if len(edge) == 2:
                i, j = edge
                weight = 1
            else:
                i, j = edge[0], edge[1]
                # third element is treated as weight (numeric)
                try:
                    weight = float(edge[2])
                except Exception:
                    weight = 1
            x = (node_coords[i][0], node_coords[j][0])
            y = (node_coords[i][1], node_coords[j][1])
            ax.plot(x, y, lw=2, color='black')
            # annotate weight at edge midpoint(with a small offset)
            try:
                mx = (node_coords[i][0] + node_coords[j][0]) / 2.0 +0.08
                my = (node_coords[i][1] + node_coords[j][1]) / 2.0 +0.1
                ax.annotate(str(int(weight)) if weight == int(weight) else str(weight), xy=(mx, my), xytext=(0, 0), textcoords='offset points', ha='center', va='center', fontsize=12, color='green')
            except Exception:
                pass
            
        # draw auxiliary links as curved segments so they don't overlap
        # physical straight edges and are visually distinct.
        from matplotlib.path import Path
        from matplotlib.patches import PathPatch

        # Group auxiliary links by unordered node pair so that multiple
        # distinct auxiliary paths between the same two nodes are drawn as
        # parallel curves instead of overlapping each other.
        from collections import defaultdict
        aux_list = list(aux_links)
        groups = defaultdict(list)
        for (entry) in aux_list:
            if len(entry) == 2:
                u, v = entry
                w = 1
            else:
                u, v, w = entry[0], entry[1], entry[2]
            key = (u, v) if u <= v else (v, u)
            groups[key].append((u, v, w))

        # draw each group's entries with slightly offset curvature
        for key, entries in groups.items():
            u, v = key
            x0, y0 = node_coords[u][0], node_coords[u][1]
            x1, y1 = node_coords[v][0], node_coords[v][1]

            dx = x1 - x0
            dy = y1 - y0
            perp_x, perp_y = -dy, dx
            norm = (perp_x ** 2 + perp_y ** 2) ** 0.5
            if norm != 0:
                perp_x /= norm
                perp_y /= norm

            dist = ((dx ** 2) + (dy ** 2)) ** 0.5
            base_amp = max(0.15 * dist, 0.2)

            # multiple entries: space them symmetrically around center
            m = len(entries)
            for k, (orig_u, orig_v, weight) in enumerate(entries):
                # center the offsets: indices -> -floor((m-1)/2) ... +ceil((m-1)/2)
                offset_index = k - (m - 1) / 2.0
                # scale amplitude so larger offsets are more curved
                amp = base_amp * (1.0 + 0.3 * abs(offset_index))
                # direction sign from offset_index
                direction = 1 if offset_index >= 0 else -1

                cx = (x0 + x1) / 2.0 + direction * perp_x * amp
                cy = (y0 + y1) / 2.0 + direction * perp_y * amp

                verts = [(x0, y0), (cx, cy), (x1, y1)]
                codes = [Path.MOVETO, Path.CURVE3, Path.CURVE3]
                path = Path(verts, codes)
                patch = PathPatch(path, facecolor='none', edgecolor='blue', lw=2)
                ax.add_patch(patch)

                # annotate weight at curve midpoint (t=0.5)
                try:
                    t = 0.5
                    mx = (1 - t) ** 2 * x0 + 2 * (1 - t) * t * cx + t ** 2 * x1
                    my = (1 - t) ** 2 * y0 + 2 * (1 - t) * t * cy + t ** 2 * y1
                    # weight label rounding as earlier
                    try:
                        wlabel = str(int(round(weight)))
                    except Exception:
                        wlabel = str(weight)
                    ax.annotate(wlabel, xy=(mx, my), xytext=(0, 0), textcoords='offset points', ha='center', va='center', fontsize=12, color='green')
                except Exception:
                    pass
        # highlight in red the shortest path with wavelength(s) available
        # a.k.a. 'best route'
        if bestroute is not None:
            for i in range(len(bestroute) - 1):
                rcurr, rnext = bestroute[i], bestroute[i + 1]
                x = (node_coords[rcurr][0], node_coords[rnext][0])
                y = (node_coords[rcurr][1], node_coords[rnext][1])
                ax.plot(x, y, 'r', lw=3)

        # draw vertices
        for label, (i, j) in nodes.items():
            ax.plot(i, j, 'wo', ms=25, mec='k')
            ax.annotate(label, xy=(i, j), ha='center', va='center')

        coord_list = list(node_coords.values())
        # https://stackoverflow.com/questions/13145368/find-the-maximum-value-in-a-list-of-tuples-in-python
        xlim = np.ceil(max(coord_list, key=itemgetter(0))[0]) + 2
        ylim = np.ceil(max(coord_list, key=itemgetter(1))[1]) + 2
        if self.name == 'nsf':
            xlim -= 1  # FIXME gambiarra, hehe. NSF needs redrawing

        # adjust values over both x and y axis
        ax.set_xticks(np.arange(xlim))
        ax.set_yticks(np.arange(ylim))

        # finally, show the plotted graph
        plt.show(block=True)