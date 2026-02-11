"""RWA simulator main function

"""

# [1] https://la.mathworks.com/matlabcentral/fileexchange/4797-wdm-network-blocking-computation-toolbox

import logging
from timeit import default_timer  # https://stackoverflow.com/a/25823885/3798300
from typing import Callable
from argparse import Namespace
import shutil
import os

# normal package-relative import (works when running as a module)
from .io import write_bp_to_disk, write_it_to_disk, plot_bp
from .net.factory import get_net_instance_from_args
from .rwa import dijkstra_first_fit, ksp_first_fit
from .RWA_functions.allocation import allocate_lightpath as rwa_allocate_lightpath
from .RWA_functions.request_queue_generation import generate_request_queue
from .RWA_functions.traffic_matrix_update import advance_traffic_matrix


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
    debug_counter = 5  # dijkstra debug counter
    aux_graph_mode = False
    enable_new_ff = True
    print('Load:   ', end='')
    for i in range(load_min, args.load + 1, load_step):
        print('%4d' % i, end=' ')
    print()

    time_per_simulation = []
    for simulation in range(args.num_sim):
        sim_time = default_timer()
        net = get_net_instance_from_args(args.topology, args.channels)

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

        for load in range(load_min, args.load + 1, load_step):
            blocks = 0
            current_time = 0.0
            nodes = list(range(net.a.shape[0]))
            event_queue = generate_request_queue(
                args.calls,
                load,
                nodes,
                holding_mean=getattr(args, 'holding_time', 10),
            )
            resource_used_time = 0.0

            while event_queue:
                event = event_queue.popleft()
                event_time = float(event['arrival_time'])
                if getattr(args, 'debug_queue', False):
                    print(f"[event pop] t={event_time:.4f} call={event['id']}")
                until_next = event_time - current_time
                #advance_traffic_matrix(net, until_next)

                current_time = event_time

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

                debug_dij = getattr(args, 'debug_dijkstra', False)
                if debug_dij:
                    debug_counter -= 1
                    if debug_counter < 0:
                        lightpath = rwa(net, s, d, args.y, debug=False, aux_graph_mode=aux_graph_mode, enable_new_ff=enable_new_ff)
                    else:
                        lightpath = rwa(net, s, d, args.y, debug=debug_dij, aux_graph_mode=aux_graph_mode, enable_new_ff=enable_new_ff)
                else:
                    lightpath = rwa(net, s, d, args.y, debug=debug_dij, aux_graph_mode=aux_graph_mode, enable_new_ff=enable_new_ff)

                if lightpath is None:
                    blocks += 1
                else:
                    holding_time_value = float(event.get('holding_time', getattr(args, 'holding_time', 10)))
                    holding_time_value = max(1.0, holding_time_value)
                    holding_time_int = max(1, int(round(holding_time_value)))
                    lightpath.holding_time = holding_time_int
                    # use the full node sequence so wavelength indexing stays valid
                    route = list(lightpath.r)
                    allocation = rwa_allocate_lightpath(
                        net,
                        route,
                        holding_time=holding_time_int,
                        enable_new_ff=enable_new_ff,
                    )
                    if allocation is None:
                        blocks += 1
                        continue
                    links = allocation.get('links') or []
                    resource_used_time += holding_time_value * max(1, len(links))

            blocklist.append(blocks)
            blocks_per_erlang.append(100.0 * blocks / args.calls if args.calls > 0 else 0.0)
            resource_used_per_erlang.append(resource_used_time)

            try:
                n_links_total = 25
                n_channels = getattr(net, 'nchannels', getattr(net, 'num_ch', None))
                total_time = float(current_time) if current_time > 0 else 0.0
                if n_links_total > 0 and n_channels and total_time > 0:
                    denom = float(n_links_total) * float(n_channels) * total_time
                    rutil = float(resource_used_time) / denom
                else:
                    rutil = 0.0
            except Exception:
                rutil = 0.0
            resource_util_per_erlang.append(rutil)

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
