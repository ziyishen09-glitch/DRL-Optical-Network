"""RWA simulator main function

"""

# [1] https://la.mathworks.com/matlabcentral/fileexchange/4797-wdm-network-blocking-computation-toolbox

import logging
from timeit import default_timer  # https://stackoverflow.com/a/25823885/3798300
from typing import Callable
from argparse import Namespace


import numpy as np
import shutil
import os
import heapq

# normal package-relative import (works when running as a module)
from .io import write_bp_to_disk, write_it_to_disk, write_SP_A_to_disk, write_SP_R_to_disk, plot_bp, plot_sp_a, plot_sp_r
from .net import Network
from .net.factory import get_net_instance_from_args


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
        if r_alg == 'dijkstra':
            if wa_alg == 'vertex-coloring':
                from .rwa import dijkstra_vertex_coloring
                return dijkstra_vertex_coloring
            elif wa_alg == 'first-fit':
                from .rwa import dijkstra_first_fit
                return dijkstra_first_fit
            elif wa_alg == 'random-fit':
                from .rwa import dijkstra_random_fit
                return dijkstra_random_fit
            else:
                raise ValueError('Unknown wavelength assignment '
                                 'algorithm "%s"' % wa_alg)
        elif r_alg == 'yen':
            if wa_alg == 'vertex-coloring':
                from .rwa import yen_vertex_coloring
                return yen_vertex_coloring
            elif wa_alg == 'first-fit':
                from .rwa import yen_first_fit
                return yen_first_fit
            elif wa_alg == 'random-fit':
                from .rwa import yen_random_fit
                return yen_random_fit
            else:
                raise ValueError('Unknown wavelength assignment '
                                 'algorithm "%s"' % wa_alg)
        else:
            raise ValueError('Unknown routing algorithm "%s"' % r_alg)
    elif rwa_alg is not None:
        if rwa_alg == 'genetic-algorithm':
            from .rwa import genetic_algorithm
            return genetic_algorithm(ga_popsize, ga_ngen, ga_xrate, ga_mrate)
        else:
            raise ValueError('Unknown RWA algorithm "%s"' % rwa_alg)
    else:
        raise ValueError('RWA algorithm not specified')


def RWA_for_update(net: "Network", s: int, d: int,
                   debug: bool = False,
                   aux_graph_mode: bool = False):
    """Route via Dijkstra and apply single-try first-fit for updates.

    Policy:
    - Run Dijkstra to get a route (optionally expand virtual edges).
    - On the first link choose the first available wavelength (lowest index).
    - Verify the same wavelength is available on all subsequent links.
    - If any link fails, do NOT try other wavelengths; return None.

    Returns a Lightpath or None on failure.
    """
    try:
        from .rwa.routing.dijkstra import dijkstra as _dijkstra
        from .net import Lightpath as _Lightpath
    except Exception:
        _dijkstra = None
        _Lightpath = None

    route = []
    if _dijkstra is not None:
        route = _dijkstra(net.a, s, d, debug=debug)

    def _expand_aux_route(rt):
        contains_virtual_path = False
        if not rt or len(rt) < 2:
            return rt, contains_virtual_path
        mapping = {}
        try:
            mapping = net.virtual_adjacency2physical_path()
        except Exception:
            mapping = {}
        expanded = []
        for i_idx in range(len(rt) - 1):
            u, v = rt[i_idx], rt[i_idx + 1]
            key = (u, v)
            if key in mapping:
                contains_virtual_path = True
                phys = mapping[key]
                if expanded and expanded[-1] == phys[0]:
                    expanded.extend(phys[1:])
                else:
                    expanded.extend(phys)
            else:
                if not expanded:
                    expanded.append(u)
                expanded.append(v)
        return (expanded if expanded else rt), contains_virtual_path

    contains_virtual_path = False
    if aux_graph_mode:
        route, contains_virtual_path = _expand_aux_route(route)

    if not route or len(route) < 2 or _Lightpath is None:
        return None

    # single-try first-fit: pick first available on first link only
    i0, j0 = route[0], route[1]
    chosen_w = None
    for w in range(net.nchannels):
        if net.n[i0][j0][w]:
            chosen_w = w
            break
    if chosen_w is None:
        return None

    # verify this wavelength along the entire path
    for idx in range(len(route) - 1):
        i, j = route[idx], route[idx + 1]
        if not net.n[i][j][chosen_w]:
            return None

    lp = _Lightpath(route, chosen_w)
    try:
        lp.contains_virtual = contains_virtual_path
    except Exception:
        pass
    return lp


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
    debug_counter = 5 #dijkstra debug counter
    aux_graph_mode = False
    enable_new_ff = True
    # print header for pretty stdout console logging
    print('Load:   ', end='')
    for i in range(load_min, args.load + 1, load_step):
        print('%4d' % i, end=' ')
    print()

    time_per_simulation = []
    for simulation in range(args.num_sim):
        sim_time = default_timer()
        net = get_net_instance_from_args(args.topology, args.channels)

        # Configure dijkstra debug logger to a per-simulation file if
        # requested. We do this here so the logger doesn't intermingle with
        # simulator's interactive stdout prints.
        dij_logger_handler = None
        if getattr(args, 'debug_dijkstra', False):
            import logging
            from pathlib import Path
            dij_log = logging.getLogger('rwa_dijkstra_debug')
            dij_log.setLevel(logging.DEBUG)
            # ensure results dir exists
            Path(getattr(args, 'result_dir', '.')).mkdir(parents=True, exist_ok=True)
            logfile = Path(getattr(args, 'result_dir', '.')) / f'dijkstra_debug.log'
            dij_logger_handler = logging.FileHandler(logfile, mode='w')
            dij_logger_handler.setLevel(logging.DEBUG)
            dij_logger_handler.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
            dij_log.addHandler(dij_logger_handler)

        # Debug: if args has attribute `debug` truthy, print adjacency matrix
        if getattr(args, 'debug_adjacency', False):
            try:
                import numpy as _np
                print('\n[debug] adjacency dtype:', net.a.dtype)
                # print matrix with integer formatting when values are integral
                if _np.all(_np.mod(net.a, 1) == 0):
                    print('[debug] adjacency matrix (int):\n', net.a.astype(int))
                else:
                    print('[debug] adjacency matrix (float):\n', net.a)
            except Exception as _e:
                print('[debug] failed to print adjacency matrix:', _e)

        # Optional topology plot only
        if getattr(args, 'plot_topo', False): # get attr默认false
            net.plot_topology()

        # RWA algorithm must be resolved regardless of plotting
        rwa = get_rwa_algorithm_from_args(
            args.r, args.w, args.rwa,
            getattr(args, 'pop_size', None),
            getattr(args, 'num_gen', None),
            getattr(args, 'cross_rate', None),
            getattr(args, 'mut_rate', None)
        )

        blocklist = []
        blocks_per_erlang = []
        # lists to store SPA and SPR per load (as proportions in [0,1])
        sp_a_per_erlang = []
        sp_r_per_erlang = []
        # resource usage numerator per load (link*wavelength*time)
        resource_used_per_erlang = []
        # resource utilization proportion per load
        resource_util_per_erlang = []

            # iterate through Erlangs (loads)
        for load in range(load_min, args.load + 1, load_step):
                blocks = 0
                SPANUM = args.calls
                SPRNUM = 0
                # track per-request counting metadata
                updates_meta: dict[int, dict] = {}
                call_sd: dict[int, tuple[int, int]] = {}
                # planned updates per call (computed at request time,
                # scheduled only if the original request is allocated)
                call_updates: dict[int, int] = {}

                # unified event queue contains both arrivals and updates
                # explicit priority to break ties at same time:
                #   priority = 0 for 'request' (arrivals)
                #   priority = 1 for 'update'
                # entries: (event_time, priority, event_type, call_id, is_last_update)
                event_queue: list[tuple[int, int, str, int, bool]] = []
                current_time = 0
                next_call_id = 0
                n_nodes = net.a.shape[0]

                # helper to push events and validate that 'update' events
                # are only scheduled from 'request' handling
                def push_event(ev_time: int, ev_type: str, ev_call: int, ev_last: bool, source: str = ''):
                    if ev_type == 'update' and source != 'request':
                        msg = f"Scheduling update event from non-request source='{source}' for call={ev_call} at t={ev_time}"
                        if getattr(args, 'debug_updates', False):
                            raise RuntimeError(msg)
                        else:
                            logger.warning(msg)
                    # priority: ensure requests are popped before updates at the same ev_time
                    prio = 0 if ev_type == 'request' else 1
                    heapq.heappush(event_queue, (ev_time, prio, ev_type, ev_call, ev_last))

                # per-load counters for update diagnostics
                upd_scheduled = 0
                upd_popped = 0
                upd_success = 0
                upd_failure = 0
                upd_missing_callsd = 0
                # upd_stats (transient per-load list) removed to reduce I/O overhead

                # resource usage numerator: total link*wavelength-time occupied
                # by all allocated lightpaths during this load (accumulate on allocation)
                resource_used_time = 0

                # helper to sample interarrival given load
                def sample_interarrival(load_val):
                    # New (preferred): discrete-time Poisson arrivals → geometric interarrival (support starts at 1)
                    # per-slot arrival rate: lambda_arr = load / E[S] where E[S]=250 slots (geometric service mean)
                    
                    # lambda_arr = float(load_val) / 250.0
                    # # # convert to geometric success probability p = 1 - exp(-lambda_arr)
                    # # # guard for extreme values to avoid p<=0 or p>=1 due to float issues
                    # p = 1.0 - np.exp(-lambda_arr)
                    # if p <= 0.0:
                    #     p = 1e-12
                    # elif p >= 1.0:
                    #     p = 1.0 - 1e-12
                    # return int(np.random.geometric(p))  # returns 1,2,3,... (slots until next arrival)

                    # Old (kept for comparison): continuous exponential + ceil to slots
                    lam = float(load_val) / 250.0
                    if lam > 1.0:
                        lam = 1.0
                    return int(np.round(np.random.exponential(scale=1.0 / lam)+0.2))
                    #Alternative rounding variant considered previously:
                    # return int(np.round(np.random.exponential(scale=1.0 / lam) + 0.15))

                # schedule first original arrival if any
                if args.calls > 0:
                    #first_inter = sample_interarrival(load)
                    push_event(current_time, 'request', next_call_id, False, source='initial')
                    next_call_id += 1

                # process events in strict time order
                while event_queue:
                    event_time, _prio, event_type, call, is_last_update = heapq.heappop(event_queue)
                    until_next = event_time - current_time

                    # Advance time to this event BEFORE processing it:
                    # release channels whose timers expire and decrement holding times.
                    # This ensures resources that become free at t=event_time are
                    # available to the event occurring at t=event_time, and avoids
                    # subtracting the prior interval from newly-created lightpaths.
                    for lightpath in net.t.lightpaths[:]:
                        if lightpath.holding_time > until_next:
                            lightpath.holding_time -= until_next
                        else:
                            net.t.remove_lightpath_by_id(lightpath.id)

                    for edge in net.get_edges():
                        if len(edge) == 2:
                            i, j = edge
                        else:
                            i, j = edge[0], edge[1]
                        for w in range(net.nchannels):
                            if net.t[i][j][w] > until_next:
                                net.t[i][j][w] -= until_next
                            else:
                                net.t[i][j][w] = 0
                                if not net.n[i][j][w]:
                                    net.n[i][j][w] = 1  # free channel

                            net.t[j][i][w] = net.t[i][j][w]
                            net.n[j][i][w] = net.n[i][j][w]

                    current_time = event_time
                    # optional debug trace for events
                    if getattr(args, 'debug_updates', False):
                        print(f"[event] t={current_time} type={event_type} call={call} last={is_last_update}")

                    # status print (same formatting as before)
                    prefix = 'Blocks: '
                    call_part = ' %04d' % call
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

                    is_update = (event_type == 'update')
                    if is_update:
                        upd_popped += 1

                    # when we process a request event, schedule the next original
                    # arrival (so we preserve the original sequential arrival process)
                    if event_type == 'request':
                        # choose endpoints for this original request and store
                        a, b = np.random.choice(n_nodes, size=2, replace=False)
                        s_init, d_init = int(a), int(b)
                        call_sd[call] = (s_init, d_init)

                        # compute planned number of updates but DO NOT push
                        # them onto the event heap yet. If the original
                        # request gets blocked we must discard these plans
                        # so blocked calls do not produce update events.
                        p = 0.004 #leave rate
                        #according to leave rate, the actual holding time of a request
                        data_layer_holding_time = np.random.geometric(p)
                        update_nums = int(np.floor(data_layer_holding_time / 100))
                        call_updates[call] = update_nums

                        # schedule the next original arrival if we still have calls
                        if next_call_id < args.calls:
                            inter = sample_interarrival(load)
                            push_event(current_time + inter, 'request', next_call_id, False, source='request')
                            next_call_id += 1

                    # determine endpoints for this event
                    skip_rwa = False  # if True, skip RWA (e.g., missing (s,d) on update)
                    if is_update:
                        try:
                            s, d = call_sd[call]
                        except Exception:
                            # Missing (s,d): discard this update (do NOT inject random endpoints)
                            upd_missing_callsd += 1
                            skip_rwa = True
                            s, d = None, None
                    else:
                        s, d = call_sd.get(call, (None, None))
                        if s is None:
                            a, b = np.random.choice(n_nodes, size=2, replace=False)
                            s, d = int(a), int(b)

                    debug_dij = getattr(args, 'debug_dijkstra', False)
                    if not skip_rwa:
                        # if is_update:
                        #     # Updates: use fixed RWA for updates
                        #     lightpath = RWA_for_update(net, s, d, debug=debug_dij, aux_graph_mode=aux_graph_mode)
                        # else:
                        # Original requests: keep using configured RWA (e.g., dijkstra_first_fit)
                        if debug_dij:
                            debug_counter -= 1
                            if debug_counter < 0:
                                lightpath = rwa(net, s, d, args.y, debug=False, aux_graph_mode=aux_graph_mode, enable_new_ff=enable_new_ff)
                            else:
                                lightpath = rwa(net, s, d, args.y, debug=debug_dij, aux_graph_mode=aux_graph_mode, enable_new_ff=enable_new_ff)
                        else:
                            lightpath = rwa(net, s, d, args.y, debug=debug_dij, aux_graph_mode=aux_graph_mode, enable_new_ff=enable_new_ff)
                    else:
                        lightpath = None

                    # Determine satisfaction & allocate resources if lightpath exists
                    is_satisfied = False
                    if lightpath is None:
                        if not is_update:
                            # original request blocked
                            blocks += 1
                            SPANUM -= 1
                            # discard any planned updates for this blocked call
                            try:
                                call_updates.pop(call, None)
                            except Exception:
                                pass
                            # also free stored endpoints for cleanliness
                            try:
                                if isinstance(call_sd, dict):
                                    call_sd.pop(call, None)
                            except Exception:
                                pass
                        else:
                            # update failed
                            is_satisfied = False
                    else:
                        # allocate resources and set holding time
                        holding_time = 10
                        lightpath.holding_time = holding_time
                        net.t.add_lightpath(lightpath)
                        # accumulate resource usage: holding_time * number_of_links
                        # lightpath.links may be a generator — materialize once for reuse
                        links_list = list(lightpath.links)
                        n_links = len(links_list) if links_list is not None else 0
                        if n_links <= 0:
                            n_links = 1  # defensive fallback
                        resource_used_time += holding_time * n_links
                        # support per-link wavelength assignments (w_list) or
                        # fall back to single-wavelength stored in lightpath.w
                        if hasattr(lightpath, 'w_list') and lightpath.w_list:
                            for idx, (i, j) in enumerate(links_list):
                                try:
                                    w = lightpath.w_list[idx]
                                except Exception:
                                    # fallback to first wavelength if indexing fails
                                    w = getattr(lightpath, 'w', None)
                                    if w is None:
                                        continue
                                net.n[i][j][w] = 0  # lock channel
                                net.t[i][j][w] = holding_time
                                net.n[j][i][w] = net.n[i][j][w]
                                net.t[j][i][w] = net.t[i][j][w]
                        else:
                            for (i, j) in links_list:
                                w = getattr(lightpath, 'w', None)
                                if w is None:
                                    continue
                                net.n[i][j][w] = 0  # lock channel
                                net.t[i][j][w] = holding_time
                                net.n[j][i][w] = net.n[i][j][w]
                                net.t[j][i][w] = net.t[i][j][w]
                        # if this was an original request allocation, schedule
                        # any planned data-layer updates computed earlier
                        if not is_update:
                            planned = call_updates.get(call, 0)
                            if planned > 0:
                                for k in range(planned):
                                    is_last = (k == planned - 1)
                                    update_time = current_time + 100 * (k + 1)
                                    push_event(update_time, 'update', call, is_last, source='request')
                                    upd_scheduled += 1
                            else:
                                # no updates will reference this call -> count as SPR success
                                SPRNUM += 1
                                # free endpoints storage immediately
                                if isinstance(call_sd, dict):
                                    call_sd.pop(call, None)
                                else:
                                    if 0 <= call < len(call_sd):
                                        call_sd[call] = None
                        if is_update:
                            is_satisfied = True

                    # SP counters and diagnostics for update events
                    if is_update:
                        # check if call_sd is present (works for dict or list)
                        s_d = None
                        if isinstance(call_sd, dict):
                            s_d = call_sd.get(call)
                        else:
                            if 0 <= call < len(call_sd):
                                s_d = call_sd[call]
                        if s_d is None:
                            upd_missing_callsd += 1

                        meta = updates_meta.setdefault(call, {'failed_counted': False, 'success_counted': False})
                        if not is_satisfied:
                            if not meta['failed_counted']:
                                SPANUM -= 1
                                meta['failed_counted'] = True
                            upd_failure += 1
                        else:
                            if not meta['success_counted']:
                                SPRNUM += 1
                                meta['success_counted'] = True
                            upd_success += 1

                        if is_last_update:
                            updates_meta.pop(call, None)
                            # free stored endpoints for this call — no further updates will reference it
                            if isinstance(call_sd, dict):
                                call_sd.pop(call, None)
                            else:
                                if 0 <= call < len(call_sd):
                                    call_sd[call] = None

                    # (time already advanced before processing the event)

                # end of event_queue while
                blocklist.append(blocks)
                blocks_per_erlang.append(100.0 * blocks / args.calls)
                # also collect resource usage numerator per load
                resource_used_per_erlang.append(resource_used_time)

                # Compute SPA and SPR as proportions. Use only non-blocked
                # requests in the denominator (blocked requests cannot generate updates).
                # Exclude blocked originals from denominators
                effective_calls_SPA = args.calls
                effective_calls_SPR = args.calls - blocks
                try:
                    SPA = float(SPANUM) / float(effective_calls_SPA) if effective_calls_SPA > 0 else 0.0
                except Exception:
                    SPA = 0.0
                try:
                    SPR = float(SPRNUM) / float(effective_calls_SPR) if effective_calls_SPR > 0 else 0.0
                except Exception:
                    SPR = 0.0
                sp_a_per_erlang.append(SPA)
                sp_r_per_erlang.append(SPR)

                # compute resource utilization (numerator / denominator)
                # denominator = number_of_links * channels * total_simulation_time
                try:
                    # compute unique undirected link count from net.get_edges()
                    
                    n_links_total = 10
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

        # end of per-load loop; finalize this simulation
        sim_time = default_timer() - sim_time
        time_per_simulation.append(sim_time)

        # Print final blocklist but truncate if it's too wide
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

        # Write one line per simulation (append mode)
        write_bp_to_disk(args.result_dir, fbase + '.bp', blocks_per_erlang)
        write_SP_A_to_disk(args.result_dir, fbase + '.spa', sp_a_per_erlang)
        write_SP_R_to_disk(args.result_dir, fbase + '.spr', sp_r_per_erlang)
        try:
            from rwa_wdm.io import write_rutil_to_disk
            write_rutil_to_disk(args.result_dir, fbase + '.rutil', resource_util_per_erlang)
        except Exception:
            logger.exception('Failed to write resource utilization (.rutil)')
        # Write only the current simulation time on its own line
        write_it_to_disk(args.result_dir, fbase + '.it', [sim_time])

        # per-simulation writes already performed above

        # cleanup dij logger handler if it was configured
        if dij_logger_handler is not None:
            dij_log = __import__('logging').getLogger('rwa_dijkstra_debug')
            dij_log.removeHandler(dij_logger_handler)
            dij_logger_handler.close()

        if args.plot:
            load_min = getattr(args, 'load_min', 1)
            load_step = getattr(args, 'load_step', 1)
            plot_bp(args.result_dir, load_min=load_min, load_max=args.load, load_step=load_step)
            # also plot SPA and SPR when plotting is requested
            try:
                plot_sp_a(args.result_dir, load_min=load_min, load_max=args.load, load_step=load_step)
            except Exception:
                logger.exception('Failed to plot SP_A')
            try:
                plot_sp_r(args.result_dir, load_min=load_min, load_max=args.load, load_step=load_step)
            except Exception:
                logger.exception('Failed to plot SP_R')
            try:
                from rwa_wdm.io import plot_rutil
                plot_rutil(args.result_dir, load_min=load_min, load_max=args.load, load_step=load_step)
            except Exception:
                logger.exception('Failed to plot resource utilization')
        # write update stats to a simple CSV-style file for debugging SPR
            # write resource usage numerators per load to a separate file
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

