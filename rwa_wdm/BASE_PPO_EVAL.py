"""PPO evaluation harness that reuses `BASEEnv` arrivals and allocations."""

from __future__ import annotations

import logging
import os
import shutil
from argparse import Namespace
from collections import deque
from glob import glob
from timeit import default_timer
from typing import Any, Optional
import math
import numpy as np
from sb3_contrib import MaskablePPO

from .BASE_env_offline import BASEEnv as BASEEnvOffline
from .BASE_env_online import BASEEnv as BASEEnvOnline
from .io import (
    plot_bp,
    write_bp_to_disk,
    write_it_to_disk,
    write_rutil_to_disk,
    write_sbp_to_disk,
)
from .net.factory import get_net_instance_from_args
from .RWA_functions.allocation import allocate_lightpath
from .RWA_functions.request_queue_generation import Request, generate_request_queue
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


def _load_trained_model(model_path: Optional[str], log_dir: Optional[str]) -> MaskablePPO:
    checkpoint = _resolve_model_checkpoint(model_path, log_dir)
    logger.info('Loading PPO checkpoint from %s', checkpoint)
    return MaskablePPO.load(checkpoint, device='cpu')


def _resolve_onnx_checkpoint(
    onnx_model_path: Optional[str],
    model_path: Optional[str],
    log_dir: Optional[str],
) -> str:
    if onnx_model_path:
        if os.path.isdir(onnx_model_path):
            raise FileNotFoundError('ONNX model path %s is a directory' % onnx_model_path)
        if os.path.exists(onnx_model_path):
            return onnx_model_path
        raise FileNotFoundError('ONNX model "%s" does not exist' % onnx_model_path)

    if model_path and model_path.lower().endswith('.onnx'):
        if os.path.exists(model_path):
            return model_path
        raise FileNotFoundError('ONNX model "%s" does not exist' % model_path)

    if model_path and model_path.lower().endswith('.zip'):
        candidate = model_path[:-4] + '.onnx'
        if os.path.exists(candidate):
            return candidate

    latest = _find_latest_checkpoint(log_dir)
    if latest:
        candidate = latest[:-4] + '.onnx'
        if os.path.exists(candidate):
            return candidate

    raise FileNotFoundError(
        'No ONNX checkpoint found. Set --onnx-model-path or provide a matching .onnx file.'
    )


class _SB3InferenceBackend:
    def __init__(self, model: MaskablePPO):
        self._model = model

    def predict(self, observation: Any, masks: Optional[np.ndarray], deterministic: bool) -> int:
        action, _ = self._model.predict(observation, deterministic=deterministic, action_masks=masks)
        return int(action)


class _ONNXInferenceBackend:
    _DTYPE_MAP = {
        'tensor(float)': np.float32,
        'tensor(double)': np.float64,
        'tensor(int64)': np.int64,
        'tensor(int32)': np.int32,
        'tensor(bool)': np.bool_,
    }

    def __init__(self, onnx_path: str):
        try:
            import onnxruntime as ort
        except Exception as exc:
            raise ImportError('ONNX backend requires onnxruntime to be installed.') from exc

        self._session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        self._input_metas = {meta.name: meta for meta in self._session.get_inputs()}
        self._input_names = [meta.name for meta in self._session.get_inputs()]
        self._output_name = self._session.get_outputs()[0].name
        self._rng = np.random.default_rng()

    def _meta_rank(self, name: str) -> Optional[int]:
        shape = self._input_metas[name].shape
        if shape is None:
            return None
        return len(shape)

    def _cast_input(self, name: str, arr: np.ndarray) -> np.ndarray:
        target_dtype = self._DTYPE_MAP.get(self._input_metas[name].type, np.float32)
        if arr.dtype != target_dtype:
            arr = arr.astype(target_dtype, copy=False)
        return arr

    def _prepare_input_array(self, name: str, value: Any) -> np.ndarray:
        arr = np.asarray(value)
        rank = self._meta_rank(name)
        if rank is not None and arr.ndim == rank - 1:
            arr = np.expand_dims(arr, axis=0)
        elif rank is not None and arr.ndim == 0 and rank == 1:
            arr = arr.reshape(1)
        return self._cast_input(name, arr)

    def _build_feed(self, observation: Any) -> dict[str, np.ndarray]:
        if isinstance(observation, dict):
            feed: dict[str, np.ndarray] = {}
            for name in self._input_names:
                if name.startswith('obs_'):
                    key = name[4:]
                    if key not in observation:
                        raise KeyError('Observation key "%s" required by ONNX input "%s" was not found.' % (key, name))
                    feed[name] = self._prepare_input_array(name, observation[key])
                elif name == 'obs' and len(observation) == 1:
                    only_key = next(iter(observation))
                    feed[name] = self._prepare_input_array(name, observation[only_key])
                else:
                    raise KeyError('Cannot map ONNX input "%s" from dict observation.' % name)
            return feed

        if len(self._input_names) != 1:
            raise ValueError('ONNX model expects multiple inputs but observation is not a dict.')
        input_name = self._input_names[0]
        return {input_name: self._prepare_input_array(input_name, observation)}

    def predict(self, observation: Any, masks: Optional[np.ndarray], deterministic: bool) -> int:
        feed = self._build_feed(observation)
        logits = self._session.run([self._output_name], feed)[0]
        logits = np.asarray(logits)
        if logits.ndim > 1:
            logits = logits[0]

        if masks is not None:
            mask = np.asarray(masks, dtype=np.bool_).reshape(-1)
            if mask.size == logits.size:
                logits = np.where(mask, logits, -1e9)
                if not mask.any():
                    return int(np.argmax(logits))

        if deterministic:
            return int(np.argmax(logits))

        # Stable softmax sampling for stochastic ONNX inference.
        shifted = logits - np.max(logits)
        exp_logits = np.exp(shifted)
        total = np.sum(exp_logits)
        if not np.isfinite(total) or total <= 0.0:
            return int(np.argmax(logits))
        probs = exp_logits / total
        return int(self._rng.choice(np.arange(logits.size), p=probs))


def _is_blocked(info: Optional[dict]) -> bool:
    if not info:
        return False
    allocation_events = info.get('allocation_events')
    if allocation_events:
        return any(event.get('success') is False for event in allocation_events)
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

    # Build discrete load levels for one-hot encoding (must match training).
    _ll_min = getattr(args, 'load_levels_min', 50)
    _ll_max = getattr(args, 'load_levels_max', 150)
    _ll_step = max(1, getattr(args, 'load_levels_step', 10))
    _load_levels = [float(v) for v in range(int(_ll_min), int(_ll_max) + 1, int(_ll_step))] or [float(_ll_min)]

    _max_load = max(float(load_max), getattr(args, 'episode_load', 1.0), 100.0)

    calls = max(0, getattr(args, 'calls', 0))
    deterministic = bool(getattr(args, 'deterministic', True))
    seed = getattr(args, 'seed', None)
    env_mode = getattr(args, 'env_mode', 'online')
    external_control = bool(getattr(args, 'external_control', False))
    env_cls = _ENV_CLASSES.get(env_mode, BASEEnvOnline)
    log_dir = getattr(args, 'log_dir', None)
    model_path = getattr(args, 'model_path', None)
    inference_backend = str(getattr(args, 'inference_backend', 'sb3')).strip().lower()
    onnx_model_path = getattr(args, 'onnx_model_path', None)
    if inference_backend == 'onnx':
        onnx_path = _resolve_onnx_checkpoint(onnx_model_path, model_path, log_dir)
        logger.info('Running ONNX inference backend using %s', onnx_path)
        agent_predictor = _ONNXInferenceBackend(onnx_path)
    else:
        agent_model = _load_trained_model(model_path, log_dir)
        logger.info('Running SB3 inference backend')
        agent_predictor = _SB3InferenceBackend(agent_model)
    failure_link = coerce_link_argument(getattr(args, 'failure_link', None))

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
        if getattr(args, 'plot_topo', False):
            try:
                net.plot_topology()
            except Exception:
                logger.exception('Failed to plot topology')
        adjacency = build_adjacency_list(net)
        impacted_pair_cache: dict[tuple[int, int], bool] = {}
        if failure_link:
            lookup_path = getattr(args, 'failure_lookup_path', None)
            lookup_k = max(1, int(getattr(args, 'failure_lookup_k', getattr(args, 'k', 2))))
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
                    adj_mat = np.asarray(getattr(net, 'a', adjacency), dtype=np.float32)
                    precomputed = build_failure_link_lookup_ksp(adj_mat, failure_link, lookup_k)
                    impacted_pair_cache.update(precomputed)
                    logger.info(
                        'Precomputed failure-link lookup (%d pairs, k=%d)',
                        len(precomputed),
                        lookup_k,
                    )
                except Exception:
                    logger.exception('Failed to precompute failure-link lookup; falling back to per-pair checks')
        if env_mode == 'offline' or external_control:
            env = env_cls(
                net,
                network_instance=net,
                max_candidates=getattr(args, 'k', 2),
                max_time_slots=getattr(args, 'max_time_slots', 1),
                holding_time_mean=getattr(args, 'holding_time', 10),
                max_load=_max_load,
                auto_manage_resources=False,
                load_levels=_load_levels,
            )
        else:
            env = env_cls(
                net,
                network_instance=net,
                max_candidates=getattr(args, 'k', 2),
                max_time_slots=getattr(args, 'max_time_slots', 1),
                holding_time_mean=getattr(args, 'holding_time', 10),
                max_load=_max_load,
                load_levels=_load_levels,
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
        failure_link_block_rates: list[float] = []

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
            pending_allocations: list[dict] = []
            manual_next_request: Optional[Request] = None

            def _pop_manual_request() -> Optional[Request]:
                nonlocal manual_next_request
                if manual_next_request is not None:
                    return manual_next_request
                if request_queue:
                    manual_next_request = request_queue.popleft()
                elif hasattr(env, 'sample_request'):
                    manual_next_request = env.sample_request()
                else:
                    manual_next_request = None
                return manual_next_request

            def _schedule_manual_allocation(entry_info: dict) -> None:
                scheduled = entry_info.get('scheduled_slot')
                if scheduled is not None:
                    scheduled_time = float(scheduled)
                else:
                    scheduled_time = scheduling_clock
                pending_allocations.append({
                    'info': entry_info,
                    'scheduled_time': max(scheduling_clock, scheduled_time),
                })
                pending_allocations.sort(key=lambda e: e['scheduled_time'])

            def _process_pending_allocations(limit_time: float, clock: float) -> float:
                nonlocal blocks, resource_used_time
                while pending_allocations and pending_allocations[0]['scheduled_time'] <= limit_time:
                    entry = pending_allocations.pop(0)
                    target_time = entry['scheduled_time']
                    if target_time > clock:
                        delta = target_time - clock
                        advance_traffic_matrix(net, delta)
                        env.advance_time(delta)
                        clock = target_time
                    info_entry = entry['info']
                    success = False
                    allocation_result = None
                    selected_route = info_entry.get('selected_candidate')
                    if selected_route and len(selected_route) >= 2:
                        holding_time = float(info_entry.get('holding_time', 0.0))
                        allocation_result = allocate_lightpath(
                            net,
                            selected_route,
                            holding_time=max(1, int(round(holding_time))) if holding_time > 0.0 else 1,
                            enable_new_ff=getattr(env, '_enable_new_ff', True),
                        )
                        success = allocation_result is not None
                    info_entry['allocation_success'] = success
                    info_entry['blocked_request'] = bool(info_entry.get('blocked_request')) or not success
                    if allocation_result:
                        info_entry.update(allocation_result)
                    if _is_blocked(info_entry):
                        blocks += 1
                    if info_entry.get('allocation_success') and info_entry.get('links'):
                        holding_time = float(info_entry.get('holding_time', 0.0))
                        resource_used_time += holding_time * len(info_entry['links'])
                return clock

            def _prepare_manual_observation(next_request: Request, current_clock: float) -> float:
                arrival = float(next_request.get('arrival_time', current_clock))
                delta = max(0.0, arrival - current_clock)
                if delta > 0.0:
                    advance_traffic_matrix(net, delta)
                    env.advance_time(delta)
                env.set_external_request(next_request)
                return max(current_clock, arrival)

            if manual_mode:
                first_request = _pop_manual_request()
                if first_request is not None:
                    scheduling_clock = _prepare_manual_observation(first_request, scheduling_clock)
                    observation = env.current_state
                    last_arrival_time = scheduling_clock
                    manual_next_request = None
                else:
                    observation = reset_observation
                    last_arrival_time = 0.0
            else:
                observation = reset_observation
                last_arrival_time = 0.0

            blocks = 0
            steps = 0
            resource_used_time = 0.0
            failure_link_requests = 0
            failure_link_blocked = 0
            while steps < calls:
                call_idx = steps
                masks = env.action_masks() if hasattr(env, 'action_masks') else None
                action = agent_predictor.predict(observation, masks, deterministic)
                observation, _, terminated, truncated, info = env.step(action)
                blocked_flag = _is_blocked(info)
                if failure_link:
                    impacted = False
                    contains_failure_link = route_contains_link(info.get('selected_candidate'), failure_link)
                    if contains_failure_link:
                        impacted = True
                    else:
                        src = info.get('request_source')
                        dst = info.get('request_destination')
                        if src is not None and dst is not None:
                            try:
                                pair = (int(src), int(dst))
                            except Exception:
                                pair = None
                            if pair is not None:
                                impacted = impacted_pair_cache.get(pair)
                                if impacted is None:
                                    impacted = failure_link_impact(pair[0], pair[1], adjacency, failure_link)
                                    impacted_pair_cache[pair] = impacted
                    if impacted:
                        failure_link_requests += 1
                        if blocked_flag:
                            failure_link_blocked += 1
                if manual_mode:
                    _schedule_manual_allocation(info)
                    next_arrival_time = math.inf
                    next_request = _pop_manual_request()
                    if next_request is not None:
                        next_arrival_time = float(next_request.get('arrival_time', scheduling_clock))
                    scheduling_clock = _process_pending_allocations(next_arrival_time, scheduling_clock)
                    if next_request is not None:
                        scheduling_clock = _prepare_manual_observation(next_request, scheduling_clock)
                        observation = env.current_state
                        last_arrival_time = max(last_arrival_time, scheduling_clock)
                        manual_next_request = None
                    else:
                        break
                else:
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
                    if blocked_flag:
                        blocks += 1
                    # Accumulate resource usage from ALL resolved allocation
                    # events (includes delayed allocations resolved during
                    # the clock advance at the start of this step).
                    _alloc_events = info.get('allocation_events') or []
                    _counted_from_events = False
                    for _ae in _alloc_events:
                        if _ae.get('success') and _ae.get('allocation_info'):
                            _ai = _ae['allocation_info']
                            _al = _ai.get('links') or []
                            _ah = float(_ai.get('holding_time', 0.0))
                            if _al:
                                resource_used_time += _ah * len(_al)
                                _counted_from_events = True
                    # Fallback for external-management path (no allocation_events)
                    if not _counted_from_events:
                        if info.get('allocation_success') and info.get('links'):
                            holding_time = float(info.get('holding_time', 0.0))
                            resource_used_time += holding_time * len(info['links'])
                    last_arrival_time = max(last_arrival_time, float(info.get('arrival_time', last_arrival_time)))
                print_block_progress(blocklist, blocks, call_idx)
                steps += 1
                if terminated or truncated:
                    break
            if manual_mode:
                scheduling_clock = _process_pending_allocations(math.inf, scheduling_clock)

            blocklist.append(blocks)
            denom = max(calls, 1)
            blocks_per_erlang.append(100.0 * blocks / denom)
            unique_links = _count_unique_links(net)
            n_channels = getattr(net, 'nchannels', getattr(net, 'num_ch', 0))
            # Use the latest clock: env internal time, scheduling_clock (manual), or last_arrival_time
            env_clock = float(getattr(env, '_current_time', 0.0))
            total_time = max(float(last_arrival_time), env_clock)
            if manual_mode:
                total_time = max(total_time, float(scheduling_clock))
            if unique_links > 0 and n_channels and total_time > 0.0:
                denom_slots = float(unique_links) * float(n_channels) * total_time
                rutil = float(resource_used_time) / denom_slots
            else:
                rutil = 0.0
            resource_util_per_erlang.append(rutil)
            if failure_link:
                if failure_link_requests > 0:
                    failure_rate = 100.0 * failure_link_blocked / failure_link_requests
                else:
                    failure_rate = math.nan
                failure_link_block_rates.append(failure_rate)

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
        if failure_link:
            fl_rates = []
            for rate in failure_link_block_rates:
                if math.isnan(rate):
                    fl_rates.append('   -')
                else:
                    fl_rates.append('%4.1f' % rate)
            fl_label = f'Link {failure_link[0]}-{failure_link[1]} BP (%):'
            print(f"{fl_label:30s} {' '.join(fl_rates)}")

        os.makedirs(result_dir, exist_ok=True)
        write_bp_to_disk(result_dir, fbase + '.bp', blocks_per_erlang)
        write_rutil_to_disk(result_dir, fbase + '.rutil', resource_util_per_erlang)
        write_it_to_disk(result_dir, fbase + '.it', [sim_time])
        if failure_link:
            try:
                write_sbp_to_disk(result_dir, fbase + '.sbp', failure_link_block_rates)
            except Exception:
                logger.exception('Failed to write single-link blocking (.sbp)')

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
