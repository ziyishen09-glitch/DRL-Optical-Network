"""RWA simulator main function

"""

# [1] https://la.mathworks.com/matlabcentral/fileexchange/4797-wdm-network-blocking-computation-toolbox

import logging
import math
from timeit import default_timer  # https://stackoverflow.com/a/25823885/3798300
from typing import Callable
from argparse import Namespace
import shutil
import os

# normal package-relative import (works when running as a module)
from .io import write_bp_to_disk, write_it_to_disk, write_sbp_to_disk, plot_bp
from .net.factory import get_net_instance_from_args
from .rwa import dijkstra_first_fit, ksp_first_fit
from .RWA_functions.allocation import allocate_lightpath as rwa_allocate_lightpath
from .RWA_functions.request_queue_generation import generate_request_queue
from .RWA_functions.traffic_matrix_update import advance_traffic_matrix
from .util import (
    build_failure_link_lookup_ksp,
    build_adjacency_list,
    coerce_link_argument,
    failure_link_impact,
    load_failure_link_lookup,
    route_contains_link,
)


__all__ = (
    'get_net_instance_from_args',
    'get_rwa_algorithm_from_args',
    'simulator'
)


logger = logging.getLogger(__name__)


def get_rwa_algorithm_from_args(r_alg: str, wa_alg: str, rwa_alg: str,
                                ga_popsize: int, ga_ngen: int,
                                ga_xrate: float, ga_mrate: float,
                                ) -> Callable:
    """Defines the main function to perform RWA from CLI string args

    Args:
        r_alg: identifier for a sole routing algorithm
        wa_alg: identifier for a sole wavelength assignment algorithm
        rwa_alg: identifier for a routine that performs RWA as one
        ga_popsize: population size for the GA-RWA procedure
        ga_ngen: number of generations for the GA-RWA procedure
        ga_xrate: crossover rate for the GA-RWA procedure
        ga_mrate: mutation rate for the GA-RWA procedure

    Returns:
        callable: a function that combines a routing algorithm and a
            wavelength assignment algorithm if those are provided
            separately, or an all-in-one RWA procedure

    Raises:
        ValueError: if neither `rwa_alg` nor both `r_alg` and `wa_alg`
            are provided

    """
    if r_alg is not None and wa_alg is not None:
        if r_alg == 'dijkstra' and wa_alg == 'first-fit':
            return dijkstra_first_fit
        if r_alg == 'ksp' and wa_alg == 'first-fit':
            return ksp_first_fit
        raise ValueError('Unsupported routing/wavelength combination "%s + %s"' % (r_alg, wa_alg))
    elif rwa_alg is not None:
        if rwa_alg == 'genetic-algorithm':
            from .rwa import genetic_algorithm
            return genetic_algorithm(ga_popsize, ga_ngen, ga_xrate, ga_mrate)
        else:
            raise ValueError('Unknown RWA algorithm "%s"' % rwa_alg)
    else:
        raise ValueError('RWA algorithm not specified')


def simulator(args: Namespace) -> None:
    """Main RWA simulation routine over WDM networks

    The loop levels of the simulator iterate over the number of repetitions,
    (simulations), the number of Erlangs (load), and the number of connection
    requests (calls) to be either allocated on the network or blocked if no
    resources happen to be available.

    Args:
        args: set of arguments provided via CLI to argparse module

    """
    load_min = getattr(args, 'load_min', 1)
    load_step = getattr(args, 'load_step', 1)
    aux_graph_mode = False
    enable_new_ff = False
    failure_link = coerce_link_argument(getattr(args, 'failure_link', None))
    print('Load:   ', end='')
    for i in range(load_min, args.load + 1, load_step):
        print('%4d' % i, end=' ')
    print()

    time_per_simulation = []
    for simulation in range(args.num_sim):
        sim_time = default_timer()
        net = get_net_instance_from_args(args.topology, args.channels)
        adjacency = build_adjacency_list(net)
        impacted_pair_cache: dict[tuple[int, int], bool] = {}
        if failure_link:
            lookup_path = getattr(args, 'failure_lookup_path', None)
            lookup_k = max(1, int(getattr(args, 'failure_lookup_k', getattr(args, 'y', 2))))
            if lookup_path and os.path.exists(lookup_path):
                try:
                    loaded_lookup, _ = load_failure_link_lookup(lookup_path)
                    impacted_pair_cache.update(loaded_lookup)
                    logger.info(
                        'Loaded failure-link lookup (%d pairs) from %s',
                        len(loaded_lookup),
                        lookup_path,
                    )
                except Exception:
                    logger.exception('Failed to load failure-link lookup from %s', lookup_path)
            elif getattr(args, 'precompute_failure_lookup', True):
                try:
                    adj_mat = net.a
                    precomputed = build_failure_link_lookup_ksp(adj_mat, failure_link, lookup_k)
                    impacted_pair_cache.update(precomputed)
                    logger.info(
                        'Precomputed failure-link lookup (%d pairs, k=%d)',
                        len(precomputed),
                        lookup_k,
                    )
                except Exception:
                    logger.exception('Failed to precompute failure-link lookup; falling back to per-pair checks')

        dij_logger_handler = None
        if getattr(args, 'debug_dijkstra', False):
            import logging as _logging
            from pathlib import Path
            dij_log = _logging.getLogger('rwa_dijkstra_debug')
            dij_log.setLevel(_logging.DEBUG)
            Path(getattr(args, 'result_dir', '.')).mkdir(parents=True, exist_ok=True)
            logfile = Path(getattr(args, 'result_dir', '.')) / f'dijkstra_debug.log'
            dij_logger_handler = _logging.FileHandler(logfile, mode='w')
            dij_logger_handler.setLevel(_logging.DEBUG)
            dij_logger_handler.setFormatter(_logging.Formatter('%(asctime)s %(message)s'))
            dij_log.addHandler(dij_logger_handler)

        if getattr(args, 'debug_adjacency', False):
            try:
                import numpy as _np
                print('\n[debug] adjacency dtype:', net.a.dtype)
                if _np.all(_np.mod(net.a, 1) == 0):
                    print('[debug] adjacency matrix (int):\n', net.a.astype(int))
                else:
                    print('[debug] adjacency matrix (float):\n', net.a)
            except Exception as _e:
                print('[debug] failed to print adjacency matrix:', _e)

        if getattr(args, 'plot_topo', False):
            net.plot_topology()

        rwa = get_rwa_algorithm_from_args(
            args.r, args.w, args.rwa,
            getattr(args, 'pop_size', None),
            getattr(args, 'num_gen', None),
            getattr(args, 'cross_rate', None),
            getattr(args, 'mut_rate', None)
        )

        blocklist: list[int] = []
        blocks_per_erlang: list[float] = []
        resource_used_per_erlang: list[float] = []
        resource_util_per_erlang: list[float] = []
        failure_link_block_rates: list[float] = []

        for load in range(load_min, args.load + 1, load_step):
            blocks = 0
            failure_link_total_requests = 0
            failure_link_blocked_requests = 0
            current_slot = 0
            nodes = list(range(net.a.shape[0]))
            event_queue = generate_request_queue(
                args.calls,
                load,
                nodes,
                holding_mean=getattr(args, 'holding_time', 10),
            )
            resource_used_time = 0.0
            pending_requests: list[dict] = []
            while event_queue or pending_requests:
                # 1. 推进到下一个有意义的时隙
                if not pending_requests and event_queue:
                    # 池子空了，直接跳到下一个请求到达的时隙
                    next_time = float(event_queue[0]['arrival_time'])
                    next_slot = int(math.ceil(next_time))
                else:
                    # 池子还有请求，按步长推进 1 个时隙
                    next_slot = current_slot + 1
                
                delta_slots = next_slot - current_slot
                if delta_slots > 0:
                    advance_traffic_matrix(net, float(delta_slots))
                current_slot = next_slot

                # 2. 把当前时隙到达的任务加入待分配池
                while event_queue:
                    next_event = event_queue[0]
                    arrival_time = float(next_event['arrival_time'])
                    arrival_slot = int(math.ceil(arrival_time))
                    if arrival_slot <= current_slot:
                        event = event_queue.popleft()
                        event['arrival_slot'] = arrival_slot
                        latest_departure = event.get('latest_departure_time')
                        if latest_departure is None:
                            deadline_slot = arrival_slot
                        else:
                            holding_time = float(event.get('holding_time', 0.0))
                            deadline_time = float(latest_departure) - holding_time
                            deadline_slot = int(math.ceil(deadline_time))
                        event['deadline_slot'] = deadline_slot
                        pending_requests.append(event)
                    else:
                        break

                # 3. 对池子里所有请求在这个时隙尝试分配
                updated_pending: list[dict] = []
                for event in pending_requests:
                    prefix = 'Blocks: '
                    call_part = ' %04d' % event['id']
                    entries = ['%04d' % b for b in blocklist]
                    term_w = shutil.get_terminal_size(fallback=(80, 20)).columns
                    avail = term_w - len(prefix) - len(call_part) - 1
                    if avail <= 0:
                        out = f"{prefix}{call_part}"
                    else:
                        per_len = 5
                        max_entries = avail // per_len
                        if max_entries >= len(entries):
                            entries_str = ' '.join(entries) + ' '
                            out = f"{prefix}{entries_str}{call_part}"
                        else:
                            shown = entries[-max_entries:] if max_entries > 0 else []
                            entries_str = ' '.join(shown) + ' '
                            out = f"{prefix}... {entries_str}{call_part}"
                    print('\r' + out, end='', flush=True)

                    s, d = int(event['source']), int(event['destination'])
                    if failure_link and 'source' in event and 'destination' in event and not event.get('_failure_link_identified'):
                        pair = (s, d)
                        impacted = impacted_pair_cache.get(pair)
                        if impacted is None:
                            impacted = failure_link_impact(s, d, adjacency, failure_link)
                            impacted_pair_cache[pair] = impacted
                        event['_failure_link_impact'] = impacted
                        event['_failure_link_identified'] = True
                        if impacted and not event.get('_route_contains_failure_link'):
                            event['_route_contains_failure_link'] = True
                            failure_link_total_requests += 1
                    debug_dij = getattr(args, 'debug_dijkstra', False)
                    lightpath = rwa(net, s, d, args.y, debug=debug_dij, aux_graph_mode=aux_graph_mode, enable_new_ff=enable_new_ff)
                    if failure_link and lightpath is not None and not event.get('_failure_link_route_evaluated'):
                        route_nodes = list(lightpath.r)
                        contains_failure_link = route_contains_link(route_nodes, failure_link)
                        already_flagged = bool(event.get('_route_contains_failure_link'))
                        if contains_failure_link and not already_flagged:
                            failure_link_total_requests += 1
                        event['_route_contains_failure_link'] = contains_failure_link or already_flagged
                        event['_failure_link_route_evaluated'] = True

                    if lightpath is None:
                        updated_pending.append(event)
                        continue

                    # 分配成功逻辑
                    holding_time_value = float(event.get('holding_time', 10))
                    holding_time_int = max(1, int(round(holding_time_value)))
                    lightpath.holding_time = holding_time_int
                    route = list(lightpath.r)
                    allocation = rwa_allocate_lightpath(net, route, holding_time=holding_time_int, enable_new_ff=enable_new_ff)
                    
                    if allocation is None:
                        updated_pending.append(event)
                        continue
                    
                    # 真正分配成功，增加资源统计
                    links = allocation.get('links') or []
                    resource_used_time += holding_time_value * max(1, len(links))

                # 4. 检查池子里的请求是否过期
                final_pending: list[dict] = []
                for event in updated_pending:
                    if current_slot >= event['deadline_slot']:
                        blocks += 1 # 正式阻塞
                        if event.get('_route_contains_failure_link'):
                            failure_link_blocked_requests += 1
                    else:
                        final_pending.append(event)
                pending_requests = final_pending

            blocklist.append(blocks)
            blocks_per_erlang.append(100.0 * blocks / args.calls if args.calls > 0 else 0.0)
            resource_used_per_erlang.append(resource_used_time)

            try:
                # dynamically count unique undirected links from topology
                edges = getattr(net, 'get_edges', lambda: [])()
                unique_edges = set()
                for edge in edges:
                    if len(edge) >= 2:
                        src, dst = int(edge[0]), int(edge[1])
                        unique_edges.add((min(src, dst), max(src, dst)))
                n_links_total = len(unique_edges)
                n_channels = getattr(net, 'nchannels', getattr(net, 'num_ch', None))
                total_time = float(current_slot) if current_slot > 0 else 0.0
                if n_links_total > 0 and n_channels and total_time > 0:
                    denom = float(n_links_total) * float(n_channels) * total_time
                    rutil = float(resource_used_time) / denom
                else:
                    rutil = 0.0
            except Exception:
                rutil = 0.0
            resource_util_per_erlang.append(rutil)
            if failure_link:
                if failure_link_total_requests > 0:
                    failure_rate = 100.0 * failure_link_blocked_requests / failure_link_total_requests
                else:
                    failure_rate = math.nan
                failure_link_block_rates.append(failure_rate)

        sim_time = default_timer() - sim_time
        time_per_simulation.append(sim_time)

        term_w = shutil.get_terminal_size(fallback=(80, 20)).columns
        prefix = 'Blocks: '
        entries = ['%04d' % b for b in blocklist]
        per_len = 5
        max_entries = max((term_w - len(prefix)) // per_len, 0)
        if max_entries >= len(entries):
            entries_str = ' '.join(entries) + ' '
            print('\r' + prefix + entries_str, end='', flush=True)
        else:
            shown = entries[-max_entries:] if max_entries > 0 else []
            entries_str = ' '.join(shown) + ' '
            print('\r' + prefix + '... ' + entries_str, end='', flush=True)

        print('\n%-7s ' % 'BP (%):', end='')
        print(' '.join(['%4.1f' % b for b in blocks_per_erlang]), end=' ')
        print('[sim %d: %.2f secs]' % (simulation + 1, sim_time))
        if failure_link:
            rates = []
            for rate in failure_link_block_rates:
                if math.isnan(rate):
                    rates.append('   -')
                else:
                    rates.append('%4.1f' % rate)
            link_label = f'Link {failure_link[0]}-{failure_link[1]} BP (%):'
            print(f"{link_label:30s} {' '.join(rates)}")
        fbase = 'BASE_%s_%dch' % (
            args.rwa if args.rwa is not None else '%s_%s' % (args.r, args.w),
            int(args.channels))

        write_bp_to_disk(args.result_dir, fbase + '.bp', blocks_per_erlang)
        try:
            from rwa_wdm.io import write_rutil_to_disk
            write_rutil_to_disk(args.result_dir, fbase + '.rutil', resource_util_per_erlang)
        except Exception:
            logger.exception('Failed to write resource utilization (.rutil)')
        write_it_to_disk(args.result_dir, fbase + '.it', [sim_time])
        if failure_link:
            try:
                write_sbp_to_disk(args.result_dir, fbase + '.sbp', failure_link_block_rates)
            except Exception:
                logger.exception('Failed to write single-link blocking (.sbp)')

        if dij_logger_handler is not None:
            dij_log = __import__('logging').getLogger('rwa_dijkstra_debug')
            dij_log.removeHandler(dij_logger_handler)
            dij_logger_handler.close()

        if args.plot:
            load_min = getattr(args, 'load_min', 1)
            load_step = getattr(args, 'load_step', 1)
            plot_bp(args.result_dir, load_min=load_min, load_max=args.load, load_step=load_step)
            try:
                from rwa_wdm.io import plot_rutil
                plot_rutil(args.result_dir, load_min=load_min, load_max=args.load, load_step=load_step)
            except Exception:
                logger.exception('Failed to plot resource utilization')
            try:
                os.makedirs(args.result_dir, exist_ok=True)
                res_file = os.path.join(args.result_dir, fbase + '.res')
                with open(res_file, 'w', encoding='utf-8') as rf:
                    rf.write('load,resource_used_time\n')
                    load_val = load_min
                    for value in resource_used_per_erlang:
                        rf.write(f"{load_val},{int(value)}\n")
                        load_val += load_step
                logger.info('Wrote resource usage to %s', res_file)
            except Exception:
                logger.exception('Failed to write resource usage (.res)')
