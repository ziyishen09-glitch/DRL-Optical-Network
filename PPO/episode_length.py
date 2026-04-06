from __future__ import annotations

import random
from typing import Optional

from gymnasium import Wrapper


class EpisodeLengthScheduler:
    """Track and emit episode lengths for a curriculum or randomized schedule."""

    def __init__(
        self,
        min_length: int,
        range_size: Optional[int] = None,
        randomize: bool = False,
        growth_interval: int = 0,
        growth_step: int = 1,
        seed: Optional[int] = None,
    ) -> None:
        self._min_length = max(1, int(min_length))
        self._range_size = max(0, int(range_size) if range_size is not None else 0)
        self.randomize = randomize
        self.growth_interval = growth_interval if growth_interval > 0 else 0
        self.growth_step = max(1, growth_step)
        self._current_lower = self._min_length
        self._current_upper = self._current_lower + self._range_size
        self._episodes_completed = 0
        self._next_growth_threshold = self.growth_interval if self.growth_interval > 0 else None
        self._rng = random.Random(seed)

    @property
    def current_lower(self) -> int:
        return self._current_lower

    @property
    def current_upper(self) -> int:
        return self._current_upper

    def mark_episode_complete(self) -> None:
        self._episodes_completed += 1
        if self.growth_interval <= 0 or self._next_growth_threshold is None:
            return
        while (
            self._next_growth_threshold is not None
            and self._episodes_completed >= self._next_growth_threshold
        ):
            self._current_lower += self.growth_step
            self._current_upper = self._current_lower + self._range_size
            self._next_growth_threshold += self.growth_interval

    def next_length(self) -> int:
        upper = max(self._current_upper, self._current_lower)
        if self.randomize and upper >= self._current_lower:
            return self._rng.randint(self._current_lower, upper)
        return upper


class EpisodeLengthWrapper(Wrapper):
    """Wrapper that updates an environment's episode length before each reset."""

    def __init__(self, env, scheduler: EpisodeLengthScheduler) -> None:
        super().__init__(env)
        self._scheduler = scheduler
        self._episode_started = False

    def reset(self, **kwargs):
        if self._episode_started:
            self._scheduler.mark_episode_complete()
        next_length = self._scheduler.next_length()
        self._set_episode_length(next_length)
        result = super().reset(**kwargs)
        self._episode_started = True
        return result

    def _set_episode_length(self, length: int) -> None:
        target = getattr(self.env.unwrapped, 'set_max_steps_per_episode', None)
        if callable(target):
            target(length)
            return
        if hasattr(self.env, 'set_max_steps_per_episode'):
            self.env.set_max_steps_per_episode(length)
            return
        setattr(self.env, '_max_steps_per_episode', length)

    def close(self) -> None:
        super().close()
