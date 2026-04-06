from typing import Dict, List, Tuple
from collections import OrderedDict
from . import Network
from ..rwa.routing.dijkstra import dijkstra


class auxgraph_aux_d1(Network):
    """A network subclass that can build an auxiliary graph.
    The auxiliary graph connects pairs of nodes (u,v) when the shortest
    physical-path distance between them is below a provided threshold.
    It also stores the physical path corresponding to each auxiliary edge.
    """

    def __init__(self, ch_n: int):
        self._name = 'auxgraph_aux_d1'
        self._fullname = 'auxgraph_aux_d1'
        self._s = 0
        self._d = 1
        super().__init__(ch_n,
                         len(self.get_nodes_2D_pos()),
                         len(self.get_edges()))

        # containers populated by build_auxiliary_graph
        self._aux_edges: List[Tuple[int, int, float]] = []
        # map (s,d) -> physical path (list of node indices)
        self._aux_paths: Dict[Tuple[int, int], List[int]] = {}
        self._aux_edges, self._aux_paths = self.build_auxiliary_graph(threshold=33.0)
        # self._aux_path shall be constructed over a lower grade
        # aux_graph like auxgraph_aux_d1 if the experiment needs 
        # to be expanded, and aux_path may look like this:
        # list[virtual_d2: tuple, virtual_d1: tuple, physical: tuple ]
        # so if using aux_d2 exceeds resource thresholds, turn to d1 and 
        # use d1 to conduct routing.
        # but here d1 is just the same with physical, so no additional configuration
        # is done.
        
        # augment adjacency matrix with auxiliary edges so routing that
        # consults `self._a` sees both physical and virtual adjacencies.
        # have to do this here as Network.__init__ builds self._a
        # and aux paths are added into a after network init.
        for (u, v, dist) in list(self._aux_edges):
            try:
                # only set adjacency when there isn't already a direct
                # physical edge (preserve physical weights)
                if float(self._a[u][v]) == 0.0:
                    self._a[u][v] = float(dist)
                    self._a[v][u] = float(dist)
            except Exception:
                # be tolerant to any index/assignment issues
                try:
                    self._a[u][v] = float(dist)
                    self._a[v][u] = float(dist)
                except Exception:
                    pass

    def get_edges(self) -> List[Tuple[int, int, int]]:
        """Return the physical edges with optional weights."""
        edges = [
            (0, 1, 20), (1, 2, 12), (2, 3, 25),
            (4, 5, 25), (5, 6, 12), (6, 7, 20),
            (3, 7, 20), (0, 4, 20), (4, 1, 28), (3, 6, 28)
        ]
        return edges
        
    def get_all_edges(self) -> List[Tuple[int, int, int]]:
        """Return all(phys + aux) edges with optional weights."""
        edges =  [
            (0, 1, 20), (1, 2, 12), (2, 3, 25),
            (4, 5, 25), (5, 6, 12), (6, 7, 20),
            (3, 7, 20), (0, 4, 20), (4, 1, 28), (3, 6, 28)
        ]
        aux = self.get_aux_edges()
        # aux entries are (u, v, dist) already
        for (u, v, dist) in list(aux):
            edges.append((u, v, dist))
        return edges

    def get_aux_edges(self) -> List[Tuple[int, int, float]]:
        """Return the auxiliary edges with weights."""
        return self._aux_edges

    def get_nodes_2D_pos(self) -> Dict[str, Tuple[float, float]]:
        return OrderedDict([
            ('0', (-1.0,  0)),
            ('1', (-0.33, 0.5)),
            ('2', (0.2,  0.5)),
            ('3', (1.36,   0.5)),
            ('4', (-0.33, -0.5)),
            ('5', (0.83,-0.5)),
            ('6', (1.36, -0.5)),
            ('7', (2.0,  0)),
        ])

    def _path_length(self, path: List[int]) -> float:
        """Return the total length/weight of a path using adjacency weights."""
        if not path or len(path) < 2:
            return 0.0
        total = 0.0
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            try:
                w = float(self._a[u][v])
            except Exception:
                w = 0.0
            total += w
        return total

    def build_auxiliary_graph(self, threshold: float) -> Tuple[List[Tuple[int, int, float]], Dict[Tuple[int, int], List[int]]]:
        """Construct auxiliary edges for all node pairs whose shortest-path
        distance is <= threshold.

        Returns:
            aux_edges: list of tuples (s, d, distance)
            aux_paths: dict mapping (s, d) -> physical path (list of node indices)
        """
        self._aux_edges = []
        self._aux_paths = {}
        n = self._num_nodes

        # Try to import the project dijkstra at runtime; if that fails,
        # use the lightweight fallback implementation above.

        for s in range(n):
            for d in range(n):
                if d == s:
                    continue
                path = list(dijkstra(self._a, s, d, debug=False))
                if not path:
                    continue
                
                # skip if there's already a direct physical link between s and d
                try:
                    direct = float(self._a[s][d]) 
                #check original _a matrix for connection
                except Exception:
                    direct = 0.0
                if direct > 0:
                    # already a physical neighbour, so not an auxiliary link
                    continue

                dist = self._path_length(path)
                if dist <= threshold:
                    self._aux_edges.append((s, d, dist))
                    self._aux_paths[(s, d)] = path

        return self._aux_edges, self._aux_paths
    
    def virtual_adjacency2physical_path(self):
        return self._aux_paths