from __future__ import annotations

from typing import Sequence

from gymnasium import Wrapper


class EpisodeLoadScheduler:
    """Cycle through a predefined sequence of Erlang loads."""

    def __init__(self, loads: Sequence[float], *, cycle: bool = True) -> None:
        values = [float(value) for value in loads if value is not None]
        if not values:
            raise ValueError('Episode load list must contain at least one value')
        self._loads = values
        self._cycle = bool(cycle)
        self._index = 0

    def next_load(self) -> float:
        value = self._loads[self._index]
        self._index += 1
        if self._index >= len(self._loads):
            if self._cycle:
                self._index = 0
            else:
                self._index = len(self._loads) - 1
        return value


class EpisodeLoadWrapper(Wrapper):
    """Wrapper that updates an environment's episode load before each reset."""

    def __init__(self, env, scheduler: EpisodeLoadScheduler) -> None:
        super().__init__(env)
        self._scheduler = scheduler

    def reset(self, **kwargs):
        next_load = self._scheduler.next_load()
        self._set_episode_load(next_load)
        return super().reset(**kwargs)

    def _set_episode_load(self, load: float) -> None:
        target = getattr(self.env.unwrapped, 'set_episode_load', None)
        if callable(target):
            target(load)
            return
        if hasattr(self.env, 'set_episode_load'):
            self.env.set_episode_load(load)
            return
        setattr(self.env, '_episode_load', float(load))
