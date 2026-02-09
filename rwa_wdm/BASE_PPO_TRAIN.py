from __future__ import annotations

from argparse import Namespace
from typing import Callable, Union

from stable_baselines3.common.vec_env import DummyVecEnv

from PPO.episode_length import EpisodeLengthWrapper
from PPO.episode_load import EpisodeLoadWrapper
from .BASE_env_offline import BASEEnv as BASEEnvOffline
from .BASE_env_online import BASEEnv as BASEEnvOnline
from .net import Network
from .net.factory import get_net_instance_from_args

__all__ = (
    'build_env_factory',
)

_ENV_CLASSES = {
    'offline': BASEEnvOffline,
    'online': BASEEnvOnline,
}

BASEEnv = Union[BASEEnvOffline, BASEEnvOnline]


def build_env_factory(args: Namespace) -> DummyVecEnv:
    """Return a vectorized factory that creates PPO-friendly BASEEnv instances."""
    scheduler = getattr(args, 'episode_length_scheduler', None)
    load_scheduler = getattr(args, 'episode_load_scheduler', None)
    env_mode = getattr(args, 'env_mode', 'offline')
    env_cls = _ENV_CLASSES.get(env_mode, BASEEnvOffline)

    def make_env(seed_offset: int) -> Callable[[], BASEEnv]:
        def _init() -> BASEEnv:
            def net_factory() -> Network:
                return get_net_instance_from_args(args.topology, args.channels)
            net = net_factory()
            env = env_cls(
                net,
                network_instance=net,
                max_candidates=getattr(args, 'k', 2),
                network_factory=net_factory,
                max_steps_per_episode=getattr(args, 'episode_length', None),
                k_shortest_paths=max(1, getattr(args, 'k', 2)),
                episode_load=getattr(args, 'episode_load', 1.0),
                auto_manage_resources=getattr(args, 'auto_manage_resources', True),
            )
            env.attach_network(net)
            if load_scheduler is not None:
                env = EpisodeLoadWrapper(env, load_scheduler)
            if scheduler is not None:
                env = EpisodeLengthWrapper(env, scheduler)
            seed_base = getattr(args, 'seed', None)
            seed_value: int | None = None
            if seed_base is not None:
                seed_value = int(seed_base) + seed_offset
            env.reset(seed=seed_value)
            if seed_value is not None:
                if hasattr(env.action_space, 'seed'):
                    env.action_space.seed(seed_value)
                if hasattr(env.observation_space, 'seed'):
                    env.observation_space.seed(seed_value)
            return env

        return _init

    n_envs = max(1, getattr(args, 'n_envs', 1))
    factories = [make_env(idx) for idx in range(n_envs)]
    return DummyVecEnv(factories)
