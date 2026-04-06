# inherit from gym.Env to create a custom environment
from __future__ import annotations
from collections import deque
from typing import Callable, Deque, List, Optional, Sequence, Tuple

import copy
import math

import numpy as np
from gymnasium import Env, spaces

from .RWA_functions.allocation import allocate_lightpath
from .RWA_functions.request_queue_generation import Request, generate_request_queue
from .RWA_functions.traffic_matrix_update import advance_traffic_matrix
from .net import Lightpath, Network
from .shortest_path.k_shortest import find_k_shortest_paths

class BASEEnv(Env):
    """Base environment for RWA in WDM networks."""

    # Default discrete Erlang load levels used for one-hot encoding.
    # Must match the load range used during training and evaluation.
    DEFAULT_LOAD_LEVELS: Tuple[float, ...] = tuple(range(50, 160, 10))  # (50,60,...,150)

    def __init__(
        self,
        network_topology,
        network_instance: Optional[Network] = None,
        *,
        max_candidates: int = 4,
        max_time_slots: int = 3,
        network_factory: Optional[Callable[[], Network]] = None,
        max_steps_per_episode: Optional[int] = None,
        k_shortest_paths: Optional[int] = None,
        episode_load: float = 1.0,
        holding_time_mean: float = 10.0,
        max_load: float = 100.0,
        auto_manage_resources: bool = True,
        auto_generate_requests: Optional[bool] = None,
        load_levels: Optional[Sequence[float]] = None,
    ):
        super().__init__()
        self.network_topology = network_topology
        num_nodes = getattr(network_topology, 'nnodes', getattr(network_topology, 'num_nodes', 0))
        num_links = getattr(network_topology, 'nlinks', getattr(network_topology, 'num_links', len(getattr(network_topology, 'get_edges', lambda: [])())))
        num_wavelengths = getattr(network_topology, 'nchannels', getattr(network_topology, 'num_wavelengths', 0))
        self._num_nodes = max(1, num_nodes)
        self._num_links = max(1, num_links)
        self._num_wavelengths = max(1, num_wavelengths)
        self._max_candidates = max(1, int(max_candidates))
        self._max_time_slots = max(1, int(max_time_slots))
        # Discrete load levels for one-hot encoding of Erlang traffic load.
        self._load_levels: Tuple[float, ...] = tuple(
            float(v) for v in (load_levels if load_levels is not None else self.DEFAULT_LOAD_LEVELS)
        )
        self._num_load_levels = max(1, len(self._load_levels))
        self.observation_space = spaces.Dict({
            # 鍏ㄥ眬浜ら€氱煩闃典笌鍏ㄥ眬娉㈤暱鍙敤鐭╅樀鏆傛椂涓嶅惎鐢?
            # mask椤规槸鍖哄垎鏈夋晥璺緞涓庡崰浣嶇殑锛堟湁鏃跺彲鑳芥壘涓嶅埌K鏉¤矾寰勶級
            # traffic and availability matrices commented out to keep observations lightweight
            # 'traffic_matrix': spaces.Box(low=0, high=np.inf, shape=(self._num_nodes, self._num_nodes), dtype=np.float32),
            # 'wavelength_availability': spaces.Box(low=0, high=1, shape=(self._num_links, self._num_wavelengths), dtype=np.int8),
            'requested_source': spaces.Box(low=0, high=self._num_nodes, shape=(1,), dtype=np.int32),
            'requested_destination': spaces.Box(low=0, high=self._num_nodes, shape=(1,), dtype=np.int32),
            'candidate_lengths': spaces.Box(low=0, high=self._num_nodes, shape=(self._max_candidates,), dtype=np.int32),
            'candidate_occupancy': spaces.Box(low=0.0, high=1.0, shape=(self._max_candidates,), dtype=np.float32),
            'candidate_traffic': spaces.Box(low=0.0, high=1.0, shape=(self._max_candidates,), dtype=np.float32),
            'candidate_mask': spaces.MultiBinary(self._max_candidates * self._max_time_slots),
            'candidate_availability': spaces.MultiBinary((self._max_candidates, self._num_wavelengths)),
            'request_allowed_slots': spaces.Box(low=1, high=self._max_time_slots, shape=(1,), dtype=np.int32),
            'traffic_load': spaces.MultiBinary(self._num_load_levels),
        })
        self._null_action_index = self._max_candidates * self._max_time_slots
        self.action_space = spaces.Discrete(self._null_action_index + 1) # +1 灏辨槸鍟撶敤null action
        self.network_instance: Optional[Network] = network_instance
        self._candidate_features = self._empty_candidate_features()
        self._candidate_paths: List[Sequence[int]] = []
        self._last_selected_route: Optional[List[int]] = None
        self._last_selected_index: Optional[int] = None
        self._last_selected_delay: Optional[int] = None
        self._pending_reward: float = 0.0
        self._last_reward: Optional[float] = None
        self._reward_success = 1.0
        self._reward_failure = -1.0
        self._reward_null = -1
        self._request_source: int = 0 # 璁板綍褰撳墠璇锋眰鐨勬簮鑺傜偣锛堝唴閮ㄥ彉閲忥級
        self._request_destination: int = 0 # 璁板綍褰撳墠璇锋眰鐨勭洰鐨勮妭鐐癸紙鍐呴儴鍙橀噺锛?
        self._current_allowed_slots: int = 1
        self._episode_load = max(0.0, float(episode_load))
        self._max_load = max(1.0, float(max_load))
        self._holding_time_mean = max(0.0, float(holding_time_mean))
        self._holding_time: int = max(1, int(round(self._holding_time_mean or 1.0)))
        self._enable_new_ff: bool = False # wavelength continuity flag
        self._network_factory = network_factory
        self._max_steps_per_episode = max_steps_per_episode
        self._steps_since_reset = 0
        self._k_shortest_paths_request = max(1, k_shortest_paths or max_candidates)
        self._rng = np.random.default_rng()
        if self._network_factory and network_instance is None:
            self.network_instance = self._network_factory()
        else:
            self.network_instance = network_instance
        self._request_queue: Deque[Request] = deque()
        self._pending_decisions: Deque[Request] = deque()
        self._scheduled_allocations: List[dict] = []
        self._last_allocation_events: List[dict] = []
        self._current_time = 0.0
        self._current_slot = 0
        self._request_id_counter = 0
        self._current_request_meta: Optional[Request] = None
        self._auto_manage_resources = bool(auto_manage_resources)
        if auto_generate_requests is None:
            self._auto_generate_requests = self._auto_manage_resources
        else:
            self._auto_generate_requests = bool(auto_generate_requests)
        if not self._auto_manage_resources:
            self._auto_generate_requests = False

    def reset(self, *, seed: Optional[int] = None, options=None):
        """Reset the environment to an initial state."""
        if seed is not None:
            self.action_space.seed(seed)
            self.observation_space.seed(seed)
            seed_value = None
            if isinstance(seed, (int, np.integer)):
                seed_value = int(seed)
            elif isinstance(seed, (tuple, list)) and seed:
                try:
                    seed_value = int(seed[0])
                except Exception:
                    seed_value = None
            self._rng = np.random.default_rng(seed_value)
        self._reset_network()
        self._candidate_features = self._empty_candidate_features()
        self._candidate_paths = []
        self._last_selected_route = None
        self._last_selected_index = None
        self._last_selected_delay = None
        self._pending_reward = 0.0
        self._steps_since_reset = 0
        self._current_allowed_slots = 1
        self._current_time = 0.0
        self._current_slot = 0
        self._pending_decisions.clear()
        self._scheduled_allocations.clear()
        self._last_allocation_events.clear()
        self._request_id_counter = 0
        if self._auto_generate_requests:
            self._prepare_request_queue()
            self._advance_clock_to_next_decision()
        else:
            self._candidate_features = self._empty_candidate_features()
            self.current_state = self._get_observation()
            self._current_request_meta = None
        return self.current_state, {}

    def _reset_network(self) -> None:
        if self._network_factory:
            self.network_instance = self._network_factory()
        elif self.network_instance is not None:
            net = self.network_instance
            net.n.fill(1)
            net.t.fill(0)
            if hasattr(net.t, 'lightpaths'):
                net.t.lightpaths.clear()
        Lightpath.reset_id_counter()

    def step(self, action):
        """Take an action in the environment using Gymnasium semantics."""
        observation, reward, terminated, truncated, info = self._take_action(action)
        self.current_state = observation
        return observation, reward, terminated, truncated, info

    def _get_initial_state(self):
        """Legacy getter retained for compatibility, but reset now drives observation prep directly."""
        return self._get_observation()

    def _take_action(self, action):
        self._steps_since_reset += 1
        mask = self._candidate_features['candidate_mask']
        candidate_paths = self._candidate_paths
        selected_route: Optional[List[int]] = None
        selected_index: Optional[int] = None
        selected_delay: Optional[int] = None
        scheduled_slot: Optional[int] = None
        self._pending_reward = 0.0
        if self._current_request_meta is None and self._auto_generate_requests:
            self._advance_clock_to_next_decision()
        if self._current_request_meta is None:
            observation = self._get_observation()
            truncated = False
            if self._max_steps_per_episode is not None:
                truncated = self._steps_since_reset >= self._max_steps_per_episode
            info = {
                'candidate_index': None,
                'selected_candidate': None,
                'selected_delay_slots': None,
                'selected_action_index': None,
                'valid_action': False,
                'allocation_success': False,
                'null_action': True,
                'blocked_request': True,
                'needs_external_management': not self._auto_manage_resources,
            }
            return observation, 0.0, False, truncated, info
        try:
            action_idx = int(action)
        except Exception:
            action_idx = None

        allocation_info = None
        null_action = False
        blocked_request = False
        valid_action = False
        no_candidates = not np.any(mask)
        if no_candidates:
            null_action = True
            valid_action = True
            blocked_request = True
            # sparse reward: null choices penalized
            # previous reward: self._pending_reward = 0.0
            self._pending_reward = self._reward_null
        elif action_idx == self._null_action_index:
            null_action = True
            valid_action = True
            # self._pending_reward = -1.0 #null action鎳茬桨
            # previous reward: self._pending_reward = 0.0
            self._pending_reward = self._reward_null
        else:
            valid_action = (
                action_idx is not None
                and 0 <= action_idx < len(mask)
                and mask[action_idx]
            )
            if valid_action:
                path_idx, delay_slots = self._decode_action_index(action_idx)
                if path_idx is None or delay_slots is None:
                    valid_action = False
                elif path_idx >= len(candidate_paths):
                    valid_action = False
                if valid_action:
                    route = candidate_paths[path_idx]
                    if len(route) >= 2:
                        selected_route = list(route)
                        selected_index = path_idx
                        selected_delay = delay_slots
                        scheduled_slot = self._schedule_request_allocation(
                            self._current_request_meta,
                            selected_route,
                            path_idx,
                            delay_slots,
                        )
                        if scheduled_slot is None:
                            blocked_request = True
                            self._pending_reward = self._reward_failure
                        elif scheduled_slot <= self._current_slot:
                            # Resolve delay-0 allocations immediately so the
                            # reward is attributed to THIS step, not a future one.
                            imm_item = self._scheduled_allocations.pop()
                            imm_resolved = self._resolve_scheduled_allocation(imm_item)
                            if imm_resolved is not None:
                                imm_event, imm_reward = imm_resolved
                                self._last_allocation_events.append(imm_event)
                                self._pending_reward = imm_reward
                            else:
                                blocked_request = True
                                self._pending_reward = self._reward_failure
                    else:
                        blocked_request = True
                        # invalid candidate, penalize immediately
                        # previous reward: self._pending_reward = 0.0
                        self._pending_reward = self._reward_failure
            else:
                blocked_request = True
                # previous reward: self._pending_reward = 0.0
                self._pending_reward = self._reward_failure

        self._last_selected_route = selected_route
        self._last_selected_index = selected_index
        self._last_selected_delay = selected_delay
        request_meta = self._current_request_meta
        allocation_events = list(self._last_allocation_events)
        allocation_success: Optional[bool] = None
        if request_meta is not None:
            request_id = request_meta.get('id')
            if request_id is not None:
                matched = next((event for event in allocation_events if event.get('request_id') == request_id), None)
                if matched is not None:
                    allocation_success = bool(matched.get('success'))
                    if allocation_success is False:
                        blocked_request = True
                    allocation_info = matched.get('allocation_info')
        info = {
            'candidate_index': selected_index,
            'selected_candidate': selected_route,
            'selected_delay_slots': selected_delay,
            'selected_action_index': action_idx,
            'scheduled_slot': scheduled_slot,
            'valid_action': valid_action,
            'allocation_success': allocation_success,
            'allocation_events': allocation_events,
            'null_action': null_action,
            'blocked_request': blocked_request,
            'needs_external_management': not self._auto_manage_resources,
        }
        if request_meta is not None:
            info['request_source'] = int(request_meta.get('source', self._request_source))
            info['request_destination'] = int(request_meta.get('destination', self._request_destination))
        else:
            info['request_source'] = int(self._request_source)
            info['request_destination'] = int(self._request_destination)
        info['holding_time'] = self._holding_time
        if request_meta is not None:
            info['arrival_time'] = float(request_meta.get('arrival_time', self._current_time))
            request_id = request_meta.get('id')
            if request_id is not None:
                info['request_id'] = request_id
        else:
            info['arrival_time'] = float(self._current_time)
        if allocation_info is not None:
            info.update(allocation_info)
        terminated = False
        truncated = False
        if self._max_steps_per_episode is not None:
            truncated = self._steps_since_reset >= self._max_steps_per_episode
        self._current_request_meta = None
        if self._auto_generate_requests:
            self._advance_clock_to_next_decision()
        observation = self._get_observation()
        reward = float(self._pending_reward)
        self._pending_reward = 0.0
        return observation, reward, terminated, truncated, info

    def _attempt_allocation(self, route: Sequence[int]) -> Optional[dict]:
        net = self.network_instance
        return allocate_lightpath(
            net,
            route,
            holding_time=self._holding_time,
            enable_new_ff=self._enable_new_ff,
        )

    def _reserve_allocation_on_mask(self, mask_net: Network, allocation: dict) -> None:
        """Reserve resources of a pending allocation on a cloned network."""
        route = allocation.get('route')
        if not route or len(route) < 2:
            return
        holding_time_value = int(max(1, allocation.get('holding_time', self._holding_time)))
        allocate_lightpath(
            mask_net,
            route,
            holding_time=holding_time_value,
            enable_new_ff=self._enable_new_ff,
        )

    def _apply_scheduled_allocations_for_slot(self, mask_net: Network, slot: int) -> None:
        for schedule in self._scheduled_allocations:
            scheduled_slot = int(schedule.get('scheduled_slot', self._current_slot))
            if scheduled_slot != slot:
                continue
            self._reserve_allocation_on_mask(mask_net, schedule)

    def _build_slot_availability_cache(
        self,
        base_slot: int,
        allowed_slots: int,
    ) -> list[np.ndarray]:
        """Build per-slot availability snapshots with one cloned network.

        This keeps the original masking semantics (scheduled allocations +
        traffic clock advances) while avoiding one deepcopy per time slot.
        """
        net = self.network_instance
        if net is None:
            raise RuntimeError('Network instance must be attached before building masking state.')
        allowed_slots = max(1, int(allowed_slots))
        mask_net = copy.deepcopy(net)
        snapshots: list[np.ndarray] = []
        current_slot = int(base_slot)
        for slot_offset in range(allowed_slots):
            self._apply_scheduled_allocations_for_slot(mask_net, current_slot)
            snapshots.append(mask_net.n[:, :, :self._num_wavelengths].copy())
            if slot_offset + 1 < allowed_slots:
                advance_traffic_matrix(mask_net, 1.0)
                current_slot += 1
        return snapshots

    def _get_observation(self):
        net = self.network_instance
        if net is None:
            raise RuntimeError('Network instance must be attached before observing.')

        # Global matrices temporarily disabled to lighten observation footprint
        observation = {}
        observation.update(self._candidate_features)
        observation['traffic_load'] = self._one_hot_load(self._episode_load)
        return observation

    def _one_hot_load(self, load_value: float) -> np.ndarray:
        """Return a one-hot vector encoding the nearest discrete load level."""
        vec = np.zeros(self._num_load_levels, dtype=np.int8)
        if self._num_load_levels == 0:
            return vec
        # Find the index of the closest load level
        best_idx = 0
        best_dist = abs(load_value - self._load_levels[0])
        for i in range(1, self._num_load_levels):
            dist = abs(load_value - self._load_levels[i])
            if dist < best_dist:
                best_dist = dist
                best_idx = i
        vec[best_idx] = 1
        return vec

    def set_traffic_load(self, value: float) -> None:
        load_value = max(0.0, float(value))
        self._episode_load = load_value
        self._traffic_load = load_value

    def _prepare_request_queue(self) -> None:
        if not self._auto_generate_requests:
            return
        # Rebuild the Erlang-driven arrival stream before each episode.
        queue_length = self._max_steps_per_episode
        queue_size = int(max(1, queue_length)) if queue_length is not None else 1
        self._current_time = 0.0
        self._current_slot = 0
        nodes = list(range(self._num_nodes))
        if len(nodes) < 2:
            self._request_queue = deque()
            return
        queue_seed = None
        if self._rng is not None:
            try:
                queue_seed = int(self._rng.integers(0, 2**31))
            except Exception:
                queue_seed = None
        self._request_queue = generate_request_queue(
            num_requests=queue_size,
            load=self._episode_load,
            nodes=nodes,
            seed=queue_seed,
            holding_mean=self._holding_time_mean,
        )

    def _prepare_request(self) -> None:
        if not self._auto_generate_requests:
            return
        self._advance_clock_to_next_decision()

    def sample_request(self) -> Request:
        nodes = list(range(self._num_nodes))
        if len(nodes) < 2:
            return {
                'id': self._request_id_counter,
                'source': 0,
                'destination': 0,
                'arrival_time': float(self._current_time),
                'holding_time': float(self._holding_time_mean or 1.0),
            }
        queue = generate_request_queue(
            num_requests=1,
            load=self._episode_load,
            nodes=nodes,
            seed=None,
            holding_mean=self._holding_time_mean,
        )
        request = queue[0] if queue else {
            'id': self._request_id_counter,
            'source': 0,
            'destination': 0,
            'arrival_time': float(self._current_time),
            'holding_time': float(self._holding_time_mean or 1.0),
        }
        self._request_id_counter += 1
        return request
    
    def _advance_time(self, delta: float) -> None:
        if delta <= 0.0:
            return
        net = self.network_instance
        if net is None:
            return
        if self._auto_manage_resources:
            advance_traffic_matrix(net, delta)
        self._current_time += delta

    def advance_time(self, delta: float) -> None:
        self._advance_time(delta)

    def set_external_request(self, request: Request) -> None:
        net = self.network_instance
        if net is None:
            raise RuntimeError('Network instance must be attached before setting external requests.')
        self._current_request_meta = request
        self._current_allowed_slots = self._compute_allowed_slots(request)
        arrival_time = float(request.get('arrival_time', self._current_time))
        self._current_time = arrival_time
        arrival_slot = int(math.ceil(arrival_time))
        self._current_slot = arrival_slot
        requested_source = int(request.get('source', self._request_source))
        requested_destination = int(request.get('destination', self._request_destination))
        holding_time = float(request.get('holding_time', self._holding_time_mean or 1.0))
        self._holding_time = max(1, int(round(holding_time)))
        candidate_paths = find_k_shortest_paths(
            net.a,
            requested_source,
            requested_destination,
            k=self._k_shortest_paths_request,
        )
        self.set_candidate_paths(
            candidate_paths,
            requested_source=requested_source,
            requested_destination=requested_destination,
        )
        self.current_state = self._get_observation()

    def set_candidate_paths(
        self,
        candidate_paths: Sequence[Sequence[int]],
        *,
        requested_source: Optional[int] = None,
        requested_destination: Optional[int] = None,
    ) -> None:
        """Cache the candidate path features so `_get_observation` can expose them."""
        if requested_source is not None:
            self._request_source = requested_source
        if requested_destination is not None:
            self._request_destination = requested_destination
        truncated = list(candidate_paths)[:self._max_candidates]
        self._candidate_paths = truncated
        self._candidate_features = self._extract_candidate_features(truncated)
        self.current_state = self._get_observation()

    def _empty_candidate_features(self) -> dict[str, np.ndarray]:
        return {
            'requested_source': np.zeros((1,), dtype=np.int32),
            'requested_destination': np.zeros((1,), dtype=np.int32),
            'candidate_lengths': np.zeros(self._max_candidates, dtype=np.int32),
            'candidate_occupancy': np.zeros(self._max_candidates, dtype=np.float32),
            'candidate_traffic': np.zeros(self._max_candidates, dtype=np.float32),
            'candidate_mask': np.zeros(self._max_candidates * self._max_time_slots, dtype=np.int8),
            'candidate_availability': np.zeros((self._max_candidates, self._num_wavelengths), dtype=np.int8),
            'request_allowed_slots': np.ones((1,), dtype=np.int32),
        }

    def _extract_candidate_features(self, candidate_paths: Sequence[Sequence[int]]) -> dict[str, np.ndarray]:
        features = self._empty_candidate_features()
        net = self.network_instance
        if net is None:
            raise RuntimeError('Network instance must be attached before processing candidate paths.')
        channel_count = max(1, getattr(net, 'nchannels', getattr(net, 'num_ch', 0)) or 0)
        features['requested_source'][0] = np.int32(self._request_source)
        features['requested_destination'][0] = np.int32(self._request_destination)
        allowed_slots = max(1, min(self._current_allowed_slots, self._max_time_slots))
        features['request_allowed_slots'][0] = np.int32(allowed_slots)
        action_mask = features['candidate_mask']
        base_slot = int(self._current_slot)
        slot_availability_cache = self._build_slot_availability_cache(base_slot, allowed_slots)

        for idx, path in enumerate(candidate_paths[:self._max_candidates]):
            if len(path) < 2:
                continue
            features['candidate_lengths'][idx] = len(path) - 1
            features['candidate_occupancy'][idx] = self._path_occupancy(path, net, channel_count)
            features['candidate_traffic'][idx] = self._path_traffic(path, net)
            availability = self._path_availability(path, net, self._num_wavelengths)
            features['candidate_availability'][idx] = availability
            for slot_offset in range(allowed_slots):
                action_idx = idx * self._max_time_slots + slot_offset
                slot_availability = slot_availability_cache[slot_offset]
                if self._enable_new_ff:
                    if self._path_has_any_wavelength_from_matrix(path, slot_availability, self._num_wavelengths):
                        action_mask[action_idx] = 1
                else:
                    availability = self._path_availability_from_matrix(path, slot_availability, self._num_wavelengths)
                    if np.any(availability):
                        action_mask[action_idx] = 1
        return features

    def _path_availability_from_matrix(
        self,
        path: Sequence[int],
        availability_matrix: np.ndarray,
        num_wavelengths: int,
    ) -> np.ndarray:
        if len(path) < 2 or num_wavelengths == 0:
            return np.zeros(num_wavelengths, dtype=np.int8)
        availability = np.ones(num_wavelengths, dtype=np.int8)
        for src, dst in zip(path[:-1], path[1:]):
            availability = np.minimum(
                availability,
                availability_matrix[src][dst][:num_wavelengths].astype(np.int8),
            )
        return availability

    def _path_has_any_wavelength_from_matrix(
        self,
        path: Sequence[int],
        availability_matrix: np.ndarray,
        num_wavelengths: int,
    ) -> bool:
        if len(path) < 2 or num_wavelengths == 0:
            return False
        for src, dst in zip(path[:-1], path[1:]):
            if not np.any(availability_matrix[src][dst][:num_wavelengths]):
                return False
        return True

    def _path_occupancy(self, path: Sequence[int], net: Network, channel_count: int) -> float:
        if len(path) < 2 or channel_count == 0:
            return 0.0
        occupancy_sum = 0.0
        hop_count = len(path) - 1
        for src, dst in zip(path[:-1], path[1:]):
            availability = net.n[src][dst]
            free = float(np.count_nonzero(availability))
            occupied_ratio = 1.0 - (free / float(channel_count))
            occupancy_sum += occupied_ratio
        return occupancy_sum / hop_count

    def _path_traffic(self, path: Sequence[int], net: Network) -> float:
        if len(path) < 2:
            return 0.0
        return float(sum(np.sum(net.t[src][dst][:self._num_wavelengths]) for src, dst in zip(path[:-1], path[1:])))

    def _path_availability(self, path: Sequence[int], net: Network, num_wavelengths: int) -> np.ndarray:
        if len(path) < 2 or num_wavelengths == 0:
            return np.zeros(num_wavelengths, dtype=np.int8)
        availability = np.ones(num_wavelengths, dtype=np.int8)
        for src, dst in zip(path[:-1], path[1:]):
            availability = np.minimum(availability, net.n[src][dst][:num_wavelengths].astype(np.int8))
        return availability

    def _compute_allowed_slots(self, request: Request) -> int:
        try:
            arrival_time = float(request.get('arrival_time', self._current_time))
            latest_departure = float(request.get('latest_departure_time', arrival_time))
            holding_time = float(request.get('holding_time', 0.0))
        except Exception:
            return 1
        deadline_time = latest_departure - holding_time
        arrival_slot = int(request.get('arrival_slot', math.ceil(arrival_time)))
        deadline_slot = int(request.get('deadline_slot', math.ceil(deadline_time)))
        return max(1, deadline_slot - arrival_slot + 1)

    def _compute_deadline_slot(self, request: Request) -> int:
        try:
            arrival_time = float(request.get('arrival_time', self._current_time))
            latest_departure = float(request.get('latest_departure_time', arrival_time))
            holding_time = float(request.get('holding_time', 0.0))
        except Exception:
            return int(math.ceil(self._current_time))
        deadline_time = latest_departure - holding_time
        return int(math.ceil(deadline_time))

    def _decode_action_index(self, action_idx: int) -> tuple[Optional[int], Optional[int]]:
        if action_idx is None or action_idx < 0:
            return None, None
        path_idx = action_idx // self._max_time_slots
        delay_slots = action_idx % self._max_time_slots
        return path_idx, delay_slots

    def _schedule_request_allocation(
        self,
        request: Request,
        route: Sequence[int],
        path_idx: int,
        delay_slots: int,
    ) -> Optional[int]:
        if request is None:
            return None
        arrival_slot = int(request.get('arrival_slot', math.ceil(float(request.get('arrival_time', self._current_time)))))
        deadline_slot = int(request.get('deadline_slot', self._compute_deadline_slot(request)))
        scheduled_slot = arrival_slot + int(delay_slots)
        if scheduled_slot > deadline_slot:
            return None
        path_load = float(self._candidate_features['candidate_occupancy'][path_idx])
        # Compute available wavelengths for selected path and best candidate
        avail_matrix = self._candidate_features['candidate_availability']
        selected_avail = int(np.sum(avail_matrix[path_idx]))
        max_avail = int(np.max(np.sum(avail_matrix, axis=1))) if avail_matrix.shape[0] > 0 else 0
        try:
            holding_time_value = max(1, int(round(float(request.get('holding_time', self._holding_time)))))
        except Exception:
            holding_time_value = max(1, self._holding_time)
        self._scheduled_allocations.append({
            'request': request,
            'route': list(route),
            'scheduled_slot': scheduled_slot,
            'deadline_slot': deadline_slot,
            'path_load': path_load,
            'selected_avail': selected_avail,
            'max_avail': max_avail,
            'holding_time': holding_time_value,
        })
        return scheduled_slot

    def _resolve_scheduled_allocation(self, item: dict) -> Optional[Tuple[dict, float]]:
        request = item.get('request')
        route = item.get('route')
        scheduled_slot = int(item.get('scheduled_slot', self._current_slot))
        deadline_slot = int(item.get('deadline_slot', scheduled_slot))
        if scheduled_slot > deadline_slot:
            return None
        if not self._auto_manage_resources:
            return None
        allocation_info = self._attempt_allocation(route)
        if allocation_info is not None:
            success = True
            # Reward = selected path available wavelengths / max available among candidates
            # selected_avail = float(item.get('selected_avail', 0))
            # max_avail = float(item.get('max_avail', 0))
            # reward_value = (selected_avail / max_avail) if max_avail > 0 else self._reward_success
            reward_value = self._reward_success
        else:
            reward_value = self._reward_failure
            success = False
        request_id = None
        if request is not None:
            request_id = request.get('id')
        return ({
            'request_id': request_id,
            'scheduled_slot': scheduled_slot,
            'success': success,
            'allocation_info': allocation_info,
        }, reward_value)

    def _set_current_request(self, request: Request) -> None:
        net = self.network_instance
        if net is None:
            raise RuntimeError('Network instance must be attached before generating requests.')
        self._current_request_meta = request
        self._current_allowed_slots = self._compute_allowed_slots(request)
        requested_source = int(request.get('source', self._request_source))
        requested_destination = int(request.get('destination', self._request_destination))
        max_index = max(0, self._num_nodes - 1)
        requested_source = max(0, min(requested_source, max_index))
        requested_destination = max(0, min(requested_destination, max_index))
        holding_time = float(request.get('holding_time', self._holding_time_mean or 1.0))
        self._holding_time = max(1, int(round(holding_time)))
        candidate_paths = find_k_shortest_paths(
            net.a,
            requested_source,
            requested_destination,
            k=self._k_shortest_paths_request,
        )
        self.set_candidate_paths(
            candidate_paths,
            requested_source=requested_source,
            requested_destination=requested_destination,
        )

    def _advance_clock_to_next_decision(self) -> None:
        net = self.network_instance
        if net is None:
            raise RuntimeError('Network instance must be attached before generating requests.')
        self._last_allocation_events = []
        reward_sum = 0.0
        while True:
            if self._pending_decisions:
                request = self._pending_decisions.popleft()
                self._set_current_request(request)
                self._pending_reward += reward_sum
                return

            next_arrival_slot = None
            if self._request_queue:
                next_arrival = self._request_queue[0]
                next_arrival_slot = int(math.ceil(float(next_arrival.get('arrival_time', self._current_time))))
            next_alloc_slot = None
            if self._scheduled_allocations:
                next_alloc_slot = min(int(item.get('scheduled_slot', self._current_slot)) for item in self._scheduled_allocations)

            if next_arrival_slot is None and next_alloc_slot is None:
                self._candidate_features = self._empty_candidate_features()
                self.current_state = self._get_observation()
                self._current_request_meta = None
                self._current_allowed_slots = 1
                self._pending_reward += reward_sum
                return

            if next_alloc_slot is None:
                next_slot = next_arrival_slot
            elif next_arrival_slot is None:
                next_slot = next_alloc_slot
            else:
                next_slot = min(next_arrival_slot, next_alloc_slot)

            delta_slots = next_slot - self._current_slot
            if delta_slots > 0:
                self._advance_time(float(delta_slots))
                self._current_slot = next_slot
            else:
                self._current_slot = next_slot

            while self._request_queue:
                next_event = self._request_queue[0]
                arrival_time = float(next_event.get('arrival_time', self._current_time))
                arrival_slot = int(math.ceil(arrival_time))
                if arrival_slot <= self._current_slot:
                    event = self._request_queue.popleft()
                    event['arrival_slot'] = arrival_slot
                    event['deadline_slot'] = self._compute_deadline_slot(event)
                    self._pending_decisions.append(event)
                else:
                    break

            if self._scheduled_allocations:
                remaining: List[dict] = []
                for item in self._scheduled_allocations:
                    if int(item.get('scheduled_slot', self._current_slot)) <= self._current_slot:
                        resolved = self._resolve_scheduled_allocation(item)
                        if resolved is not None:
                            event, reward_value = resolved
                            self._last_allocation_events.append(event)
                            reward_sum += reward_value
                    else:
                        remaining.append(item)
                self._scheduled_allocations = remaining

    def attach_network(self, network_instance: Network) -> None:
        #PPO_TRAIN涓殑缃戠粶瀹炰緥涓巈nv鐨勬帴鍙ｏ紝offline loop now passes the instance upfront
        self.network_instance = network_instance

    @property
    def max_candidates(self) -> int:
        return self._max_candidates

    @property
    def candidate_mask(self) -> np.ndarray:
        return self._candidate_features['candidate_mask']

    def action_masks(self) -> np.ndarray:
        """Return a boolean mask over the full action space for MaskablePPO."""
        mask = np.zeros(self.action_space.n, dtype=np.bool_)
        candidate_mask = self._candidate_features['candidate_mask']
        mask[:len(candidate_mask)] = candidate_mask.astype(np.bool_)
        # Null action is always available as a fallback
        mask[self._null_action_index] = True
        return mask

    def record_reward(self, reward: float) -> None:
        """Store the final reward returned by the simulator."""
        self._last_reward = float(reward)

    @property
    def last_reward(self) -> Optional[float]:
        return self._last_reward

    @property
    def selected_candidate(self) -> Optional[Sequence[int]]:
        return self._last_selected_route

    @property
    def selected_candidate_index(self) -> Optional[int]:
        return self._last_selected_index

    def set_max_steps_per_episode(self, value: Optional[int]) -> None:
        self._max_steps_per_episode = value

    @property
    def max_steps_per_episode(self) -> Optional[int]:
        return self._max_steps_per_episode

    def set_episode_load(self, value: float) -> None:
        self._episode_load = max(0.0, float(value))
