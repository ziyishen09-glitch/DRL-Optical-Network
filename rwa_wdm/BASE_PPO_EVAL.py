"""PPO evaluation harness that reuses `BASEEnv` arrivals and allocations."""

from __future__ import annotations

import logging
import os
import shutil
from argparse import Namespace
from collections import deque
from glob import glob
from timeit import default_timer
from typing import Optional

import numpy as np
from stable_baselines3 import PPO

from .BASE_env_offline import BASEEnv as BASEEnvOffline
from .BASE_env_online import BASEEnv as BASEEnvOnline
from .io import (
    plot_bp,
    write_bp_to_disk,
    write_it_to_disk,
    write_rutil_to_disk,
)
from .net.factory import get_net_instance_from_args
from .RWA_functions.allocation import allocate_lightpath
from .RWA_functions.request_queue_generation import Request, generate_request_queue
from .RWA_functions.traffic_matrix_update import advance_traffic_matrix


__all__ = (
    'get_net_instance_from_args',
    'simulator',
)

logger = logging.getLogger(__name__)

_ENV_CLASSES = {
    'offline': BASEEnvOffline,
    'online': BASEEnvOnline,
}


def _find_latest_checkpoint(log_dir: Optional[str]) -> Optional[str]:
    if not log_dir:
        return None
    pattern = os.path.join(log_dir, 'rwa_model_*steps.zip')
    checkpoints = glob(pattern)
    if not checkpoints:
        return None
    return max(checkpoints, key=os.path.getmtime)


def _resolve_model_checkpoint(model_path: Optional[str], log_dir: Optional[str]) -> str:
    if model_path:
        if os.path.isdir(model_path):
            raise FileNotFoundError('Model path %s is a directory' % model_path)
        if os.path.exists(model_path):
            return model_path
        raise FileNotFoundError('PPO model "%s" does not exist' % model_path)
    candidate = _find_latest_checkpoint(log_dir)
    if candidate:
        return candidate
    raise FileNotFoundError('No PPO checkpoint found in %s' % (log_dir or '<undefined>'))


def _load_trained_model(model_path: Optional[str], log_dir: Optional[str]) -> PPO:
    checkpoint = _resolve_model_checkpoint(model_path, log_dir)
    logger.info('Loading PPO checkpoint from %s', checkpoint)
    return PPO.load(checkpoint, device='cpu')


def _is_blocked(info: Optional[dict]) -> bool:
    if not info:
        return False
    if info.get('null_action'):
        return True
    if info.get('blocked_request'):
        return True
    if info.get('selected_candidate') is None:
        return True
    if info.get('allocation_success') is False:
        return True
    return False


def _count_unique_links(net) -> int:
    edges = getattr(net, 'get_edges', lambda: [])()
    unique_edges = set()
    for edge in edges:
        if len(edge) < 2:
            continue
        src, dst = int(edge[0]), int(edge[1])
        unique_edges.add(tuple(sorted((src, dst))))
    return len(unique_edges)


def simulator(args: Namespace) -> None:
    load_min = getattr(args, 'load_min', 1)
    load_step = max(1, getattr(args, 'load_step', 1))
    load_max = getattr(args, 'load', load_min)
    if load_max < load_min:
        logger.warning('load (%d) < load_min (%d); swapping bounds', load_max, load_min)
        load_min, load_max = load_max, load_min
    load_values = list(range(load_min, load_max + 1, load_step)) or [load_min]

    calls = max(0, getattr(args, 'calls', 0))
    deterministic = bool(getattr(args, 'deterministic', True))
    seed = getattr(args, 'seed', None)
    env_mode = getattr(args, 'env_mode', 'online')
    external_control = bool(getattr(args, 'external_control', False))
    env_cls = _ENV_CLASSES.get(env_mode, BASEEnvOnline)
    log_dir = getattr(args, 'log_dir', None)
    model_path = getattr(args, 'model_path', None)
    agent_model = _load_trained_model(model_path, log_dir)

    logger.info('Running PPO evaluation (deterministic=%s, seed=%s)', deterministic, seed)
    print('Load:   ', end='')
    for load in load_values:
        print('%4d' % load, end=' ')
    print()

    result_dir = getattr(args, 'result_dir', '.')
    algo_tag = 'ppo_env'
    channels = getattr(args, 'channels', 0)
    fbase = f'PPO_{algo_tag}_{int(channels)}ch'

    def print_block_progress(completed_blocks: list[int], current_blocks: int, call_idx: int) -> None:
        prefix = 'Blocks: '
        call_part = f' {call_idx:04d}'
        entries = ['%04d' % b for b in completed_blocks]
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

    time_per_simulation: list[float] = []
    for simulation in range(max(1, getattr(args, 'num_sim', 1))):
        sim_start = default_timer()
        net = get_net_instance_from_args(getattr(args, 'topology', None), getattr(args, 'channels', None))
        if env_mode == 'offline' or external_control:
            env = env_cls(
                net,
                network_instance=net,
                max_candidates=getattr(args, 'k', 2),
                holding_time_mean=getattr(args, 'holding_time', 10),
                auto_manage_resources=False,
            )
        else:
            env = env_cls(
                net,
                network_instance=net,
                max_candidates=getattr(args, 'k', 2),
                holding_time_mean=getattr(args, 'holding_time', 10),
            )
        env.attach_network(net)
        max_steps_setter = getattr(env, 'set_max_steps_per_episode', None)
        if callable(max_steps_setter):
            max_steps_setter(max(1, calls))
        load_setter = getattr(env, 'set_episode_load', None)
        traffic_setter = getattr(env, 'set_traffic_load', None)

        blocklist: list[int] = []
        blocks_per_erlang: list[float] = []
        resource_util_per_erlang: list[float] = []

        for idx, load in enumerate(load_values):
            if callable(load_setter):
                load_setter(load)
            if callable(traffic_setter):
                traffic_setter(load)
            reset_seed = None if seed is None else seed + simulation * 1000 + idx
            manual_mode = not getattr(env, '_auto_manage_resources', True)
            request_queue: deque[Request] = deque()
            if manual_mode:
                queue_length = max(1, calls)
                nodes = list(range(getattr(env, '_num_nodes', 0)))
                if len(nodes) >= 2:
                    queue_seed = None
                    rng = getattr(env, '_rng', None)
                    if rng is not None:
                        try:
                            queue_seed = int(rng.integers(0, 2**31))
                        except Exception:
                            queue_seed = None
                    holding_mean = getattr(env, '_holding_time_mean', getattr(env, '_holding_time', 10))
                    request_queue.extend(
                        generate_request_queue(
                            num_requests=queue_length,
                            load=load,
                            nodes=nodes,
                            seed=queue_seed,
                            holding_mean=holding_mean,
                        )
                    )
            reset_observation, _ = env.reset(seed=reset_seed)
            scheduling_clock = 0.0
            current_request = None
            if manual_mode:
                if request_queue:
                    current_request = request_queue.popleft()
                elif hasattr(env, 'sample_request'):
                    current_request = env.sample_request()
                else:
                    current_request = {'id': 0, 'source': 0, 'destination': 0, 'arrival_time': 0.0, 'holding_time': 0.0}
                arrival_time = float(current_request.get('arrival_time', scheduling_clock))
                elapsed_since_last_arrival = max(0.0, arrival_time - scheduling_clock)
                if elapsed_since_last_arrival > 0.0:
                    advance_traffic_matrix(net, elapsed_since_last_arrival)
                    env.advance_time(elapsed_since_last_arrival)
                scheduling_clock = arrival_time
                env.set_external_request(current_request)
                observation = env.current_state
            else:
                observation = reset_observation

            blocks = 0
            steps = 0
            resource_used_time = 0.0
            last_arrival_time = 0.0
            while steps < calls:
                call_idx = steps
                action, _ = agent_model.predict(observation, deterministic=deterministic)
                observation, _, terminated, truncated, info = env.step(action)
                if info.get('needs_external_management'):
                    holding_time = float(info.get('holding_time', 0.0))
                    blocked_request = bool(info.get('blocked_request'))
                    allocation_success = False
                    allocation_result = None
                    if not blocked_request:
                        selected_route = info.get('selected_candidate')
                        if selected_route and len(selected_route) >= 2:
                            allocation_result = allocate_lightpath(
                                net,
                                selected_route,
                                holding_time=max(1, int(round(holding_time))) if holding_time > 0.0 else 1,
                                enable_new_ff=getattr(env, '_enable_new_ff', True),
                            )
                            allocation_success = allocation_result is not None
                    info['allocation_success'] = allocation_success
                    info['blocked_request'] = blocked_request or not allocation_success
                    if allocation_result:
                        info.update(allocation_result)
                    info['arrival_time'] = float(getattr(env, '_current_time', 0.0))
                if _is_blocked(info):
                    blocks += 1
                if info.get('allocation_success') and info.get('links'):
                    holding_time = float(info.get('holding_time', 0.0))
                    resource_used_time += holding_time * len(info['links'])
                last_arrival_time = max(last_arrival_time, float(info.get('arrival_time', last_arrival_time)))
                print_block_progress(blocklist, blocks, call_idx)
                steps += 1
                if manual_mode:
                    if request_queue:
                        manual_request = request_queue.popleft()
                    elif hasattr(env, 'sample_request'):
                        manual_request = env.sample_request()
                    else:
                        manual_request = {
                            'id': steps,
                            'source': 0,
                            'destination': 0,
                            'arrival_time': float(info.get('arrival_time', scheduling_clock)),
                            'holding_time': float(info.get('holding_time', 1.0)),
                        }
                    arrival_time = float(manual_request.get('arrival_time', scheduling_clock))
                    elapsed_since_last_arrival = max(0.0, arrival_time - scheduling_clock)
                    if elapsed_since_last_arrival > 0.0:
                        advance_traffic_matrix(net, elapsed_since_last_arrival)
                        env.advance_time(elapsed_since_last_arrival)
                    scheduling_clock = arrival_time
                    env.set_external_request(manual_request)
                    observation = env.current_state
                if terminated or truncated:
                    break

            blocklist.append(blocks)
            denom = max(calls, 1)
            blocks_per_erlang.append(100.0 * blocks / denom)
            unique_links = _count_unique_links(net)
            n_channels = getattr(net, 'nchannels', getattr(net, 'num_ch', 0))
            total_time = float(last_arrival_time) if last_arrival_time > 0 else 0.0
            if unique_links > 0 and n_channels and total_time > 0.0:
                denom_slots = float(unique_links) * float(n_channels) * total_time
                rutil = float(resource_used_time) / denom_slots
            else:
                rutil = 0.0
            resource_util_per_erlang.append(rutil)

        sim_time = default_timer() - sim_start
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

        print('\nBP (%): ' + ' '.join('%4.1f' % v for v in blocks_per_erlang) + f' [sim {simulation + 1}: {sim_time:.2f}s]')

        os.makedirs(result_dir, exist_ok=True)
        write_bp_to_disk(result_dir, fbase + '.bp', blocks_per_erlang)
        write_rutil_to_disk(result_dir, fbase + '.rutil', resource_util_per_erlang)
        write_it_to_disk(result_dir, fbase + '.it', [sim_time])

    if getattr(args, 'plot', False):
        try:
            plot_bp(
                result_dir,
                load_min,
                load_max,
                load_step,
            )
        except Exception:
            logger.exception('Failed to plot blocking probabilities')