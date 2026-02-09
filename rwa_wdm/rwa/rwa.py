from typing import Callable, List, Union

from ..net import Lightpath, Network
from ..shortest_path.k_shortest import find_k_shortest_paths
from .routing import dijkstra
from .wlassignment import first_fit
from .ga import GeneticAlgorithm

__all__ = (
    'dijkstra_first_fit',
    'ksp_first_fit',
    'genetic_algorithm',
)


# genetic algorithm object (global)
# FIXME this looks bad. perhaps this whole script should be a class
ga: Union[GeneticAlgorithm, None] = None


def _get_candidate_routes(net: Network, s: int, d: int, k: int | None, debug: bool) -> List[List[int]]:
    num_paths = int(k) if k is not None else 1
    if num_paths < 1:
        num_paths = 1
    if num_paths == 1:
        route = dijkstra(net.a, s, d, debug=debug)
        return [route] if route else []
    try:
        return find_k_shortest_paths(net.a, s, d, num_paths)
    except ValueError:
        return []


def _expand_aux_route(net: Network, route: List[int]) -> tuple[List[int], bool, list[list[int]]]:
    contains_virtual_path = False
    if not route or len(route) < 2:
        return route, contains_virtual_path, []

    try:
        mapping = net.virtual_adjacency2physical_path()
    except Exception:
        mapping = {}

    expanded: list[int] = []
    mapped_virtual_route: list[list[int]] = []
    for i in range(len(route) - 1):
        u, v = route[i], route[i + 1]
        key = (u, v)
        if key in mapping:
            contains_virtual_path = True
            phys = mapping[key]
            mapped_virtual_route.append(phys)
            if expanded and expanded[-1] == phys[0]:
                expanded.extend(phys[1:])
            else:
                expanded.extend(phys)
        else:
            if not expanded:
                expanded.append(u)
            expanded.append(v)

    if not expanded:
        return route, contains_virtual_path, mapped_virtual_route
    return expanded, contains_virtual_path, mapped_virtual_route


def _try_first_fit_routes(net: Network, routes: List[List[int]], aux_graph_mode: bool, enable_new_ff: bool) -> Union[Lightpath, None]:
    for route in routes:
        expanded_route = route
        contains_virtual_path = False
        mapped_virtual_route: list[list[int]] = []
        if aux_graph_mode:
            expanded_route, contains_virtual_path, mapped_virtual_route = _expand_aux_route(net, route)

        if not expanded_route:
            continue

        w_list = first_fit(net, expanded_route, contains_virtual_path, enable_new_ff=enable_new_ff)
        if not w_list:
            continue
        if not all(((w >= 0 and w < net.nchannels) or (isinstance(w, int) and w < 0)) for w in w_list):
            continue
        base_w = next((w for w in w_list if isinstance(w, int) and w >= 0), 0)
        lp = Lightpath(expanded_route, base_w)
        try:
            lp.w_list = list(w_list)
        except Exception:
            pass
        try:
            lp.contains_virtual = contains_virtual_path
            lp.mapped_virtual_route = mapped_virtual_route
        except Exception:
            pass
        return lp
    return None


def dijkstra_first_fit(net: Network, s: int, d: int, k: int, debug: bool = False,
                       aux_graph_mode: bool = False, enable_new_ff: bool = False,
                       ) -> Union[Lightpath, None]:
    routes = _get_candidate_routes(net, s, d, 1, debug)
    return _try_first_fit_routes(net, routes, aux_graph_mode, enable_new_ff)


def ksp_first_fit(net: Network, s: int, d: int, k: int, debug: bool = False,
                  aux_graph_mode: bool = False, enable_new_ff: bool = False,
                  ) -> Union[Lightpath, None]:
    routes = _get_candidate_routes(net, s, d, k, debug)
    return _try_first_fit_routes(net, routes, aux_graph_mode, enable_new_ff)


def genetic_algorithm_callback(net: Network, k: int) -> Union[Lightpath, None]:
    """Callback function to perform RWA via genetic algorithm

    Args:
        net: Network topology instance
        k: number of alternate paths (ignored)

    Returns:
        Lightpath: if successful, returns both route and wavelength index as a
            lightpath

    """
    route, wavelength = ga.run(net, k)
    if wavelength is not None and wavelength < net.nchannels:
        return Lightpath(route, wavelength)
    return None


def genetic_algorithm(pop_size: int, num_gen: int,
                      cross_rate: float, mut_rate: float) -> Callable:
    """Genetic algorithm as both routing and wavelength assignment algorithm

    This function just sets the parameters to the GA, so it acts as if it were
    a class constructor, setting a global variable as instance to the
    `GeneticAlgorithm` object in order to be further used by a callback
    function, which in turn returns the lightpath itself upon RWA success. This
    split into two classes is due to the fact that the class instance needs to
    be executed only once, while the callback may be called multiple times
    during simulation, namely one time per number of arriving call times number
    of load in Erlags (calls * loads)

    Note:
        Maybe this entire script should be a class and `ga` instance could be
        an attribute. Not sure I'm a good programmer.

    Args:
        pop_size: number of chromosomes in the population
        num_gen: number of generations towards evolve
        cross_rate: percentage of individuals to perform crossover
        mut_rate: percentage of individuals to undergo mutation

    Returns:
        callable: a callback function that calls the `GeneticAlgorithm` runner
            class, which finally and properly performs the RWA procedure

    """
    global ga
    ga = GeneticAlgorithm(pop_size, num_gen, cross_rate, mut_rate)
    return genetic_algorithm_callback
