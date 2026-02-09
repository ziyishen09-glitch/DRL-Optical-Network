# inherit from gym.Env to create a custom environment
from __future__ import annotations
from collections import deque
from typing import Callable, Deque, List, Optional, Sequence

import numpy as np
from gymnasium import Env, spaces

from .RWA_functions.allocation import allocate_lightpath
from .RWA_functions.request_queue_generation import Request, generate_request_queue
from .RWA_functions.traffic_matrix_update import advance_traffic_matrix
from .net import Network
from .shortest_path.k_shortest import find_k_shortest_paths

class BASEEnv(Env):
    """Base environment for RWA in WDM networks."""

    def __init__(
        self,
        network_topology,
        network_instance: Optional[Network] = None,
        *,
        max_candidates: int = 16,
        network_factory: Optional[Callable[[], Network]] = None,
        max_steps_per_episode: Optional[int] = None,
        k_shortest_paths: Optional[int] = None,
        episode_load: float = 1.0,
        holding_time_mean: float = 10.0,
        auto_manage_resources: bool = True,
        auto_generate_requests: Optional[bool] = None,
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
        self.observation_space = spaces.Dict({
            # 全局交通矩阵与全局波长可用矩阵暂时不启用
            # mask项是区分有效路径与占位的（有时可能找不到K条路径）
            # traffic and availability matrices commented out to keep observations lightweight
            # 'traffic_matrix': spaces.Box(low=0, high=np.inf, shape=(self._num_nodes, self._num_nodes), dtype=np.float32),
            # 'wavelength_availability': spaces.Box(low=0, high=1, shape=(self._num_links, self._num_wavelengths), dtype=np.int8),
            'requested_source': spaces.Box(low=0, high=self._num_nodes, shape=(1,), dtype=np.int32),
            'requested_destination': spaces.Box(low=0, high=self._num_nodes, shape=(1,), dtype=np.int32),
            'candidate_lengths': spaces.Box(low=0, high=self._num_nodes, shape=(self._max_candidates,), dtype=np.int32),
            'candidate_occupancy': spaces.Box(low=0.0, high=1.0, shape=(self._max_candidates,), dtype=np.float32),
            'candidate_mask': spaces.MultiBinary(self._max_candidates),
            # 'candidate_traffic': spaces.Box(low=0.0, high=np.inf, shape=(self._max_candidates,), dtype=np.float32), #traffic也可以暂时不用
            'candidate_availability': spaces.MultiBinary((self._max_candidates, self._num_wavelengths)),
            'traffic_load': spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
        })
        self._null_action_index = self._max_candidates
        self.action_space = spaces.Discrete(self._max_candidates + 1) # +1 就是啓用null action
        self.network_instance: Optional[Network] = network_instance
        self._candidate_features = self._empty_candidate_features()
        self._candidate_paths: List[Sequence[int]] = []
        self._last_selected_route: Optional[List[int]] = None
        self._last_selected_index: Optional[int] = None
        self._pending_reward: float = 0.0
        self._last_reward: Optional[float] = None
        self._request_source: int = 0 # 记录当前请求的源节点（内部变量）
        self._request_destination: int = 0 # 记录当前请求的目的节点（内部变量）
        self._episode_load = max(0.0, float(episode_load))
        self._holding_time_mean = max(0.0, float(holding_time_mean))
        self._holding_time: int = max(1, int(round(self._holding_time_mean or 1.0)))
        self._enable_new_ff: bool = True
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
        self._current_time = 0.0
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
        self._pending_reward = 0.0
        self._steps_since_reset = 0
        if self._auto_generate_requests:
            self._prepare_request_queue()
            self._prepare_request()
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
        self._pending_reward = 0.0
        try:
            action_idx = int(action)
        except Exception:
            action_idx = None

        allocation_info = None
        null_action = False
        blocked_request = False
        valid_action = False
        if action_idx == self._null_action_index:
            null_action = True
            valid_action = True
            self._pending_reward = -1.0 #null action懲罰
        else:
            valid_action = (
                action_idx is not None
                and 0 <= action_idx < len(mask)
                and mask[action_idx]
                and action_idx < len(candidate_paths)
            )
            if valid_action:
                route = candidate_paths[action_idx]
                if len(route) >= 2:
                    selected_route = list(route)
                    selected_index = action_idx
                    if self._auto_manage_resources:
                        allocation_info = self._attempt_allocation(selected_route)
                        if allocation_info is not None:
                            self._pending_reward = 1.0
                        else:
                            blocked_request = True
                            self._pending_reward = -1.0
                    else:
                        self._pending_reward = 0.0
                else:
                    blocked_request = True
                    self._pending_reward = -1.0
            else:
                blocked_request = True
                self._pending_reward = -1.0

        self._last_selected_route = selected_route
        self._last_selected_index = selected_index
        request_meta = self._current_request_meta
        info = {
            'candidate_index': selected_index,
            'selected_candidate': selected_route,
            'valid_action': valid_action,
            'allocation_success': allocation_info is not None,
            'null_action': null_action,
            'blocked_request': blocked_request,
            'needs_external_management': not self._auto_manage_resources,
        }
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
        if self._auto_generate_requests:
            self._prepare_request()
        observation = self._get_observation()
        return observation, self._pending_reward, terminated, truncated, info

    def _attempt_allocation(self, route: Sequence[int]) -> Optional[dict]:
        net = self.network_instance
        return allocate_lightpath(
            net,
            route,
            holding_time=self._holding_time,
            enable_new_ff=self._enable_new_ff,
        )


    def _get_observation(self):
        net = self.network_instance
        if net is None:
            raise RuntimeError('Network instance must be attached before observing.')

        # Global matrices temporarily disabled to lighten observation footprint
        observation = {}
        observation.update(self._candidate_features)
        observation['traffic_load'] = np.array([float(self._episode_load)], dtype=np.float32)
        return observation

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
        net = self.network_instance
        if net is None:
            raise RuntimeError('Network instance must be attached before generating requests.')
        if not self._request_queue:
            self._candidate_features = self._empty_candidate_features()
            self.current_state = self._get_observation()
            self._current_request_meta = None
            return
        request = self._request_queue.popleft()
        arrival_time = float(request.get('arrival_time', self._current_time))
        elapsed = max(0.0, arrival_time - self._current_time)
        self._advance_time(elapsed)
        self._current_time = arrival_time
        self._current_request_meta = request
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
        self._current_time = float(request.get('arrival_time', self._current_time))
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
            'candidate_mask': np.zeros(self._max_candidates, dtype=np.int8),
            #'candidate_traffic': np.zeros(self._max_candidates, dtype=np.float32),
            'candidate_availability': np.zeros((self._max_candidates, self._num_wavelengths), dtype=np.int8),
        }

    def _extract_candidate_features(self, candidate_paths: Sequence[Sequence[int]]) -> dict[str, np.ndarray]:
        features = self._empty_candidate_features()
        net = self.network_instance
        if net is None:
            raise RuntimeError('Network instance must be attached before processing candidate paths.')
        channel_count = max(1, getattr(net, 'nchannels', getattr(net, 'num_ch', 0)) or 0)
        features['requested_source'][0] = np.int32(self._request_source)
        features['requested_destination'][0] = np.int32(self._request_destination)
        for idx, path in enumerate(candidate_paths[:self._max_candidates]):
            if len(path) < 2:
                continue
            features['candidate_mask'][idx] = 1
            features['candidate_lengths'][idx] = len(path) - 1
            features['candidate_occupancy'][idx] = self._path_occupancy(path, net, channel_count)
            # features['candidate_traffic'][idx] = self._path_traffic(path, net)
            features['candidate_availability'][idx] = self._path_availability(path, net, self._num_wavelengths)
        return features

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

    def attach_network(self, network_instance: Network) -> None:
        #PPO_TRAIN中的网络实例与env的接口，offline loop now passes the instance upfront
        self.network_instance = network_instance

    @property
    def max_candidates(self) -> int:
        return self._max_candidates

    @property
    def candidate_mask(self) -> np.ndarray:
        return self._candidate_features['candidate_mask']

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
