"""Helper that launches the PPO evaluation entry point from outside the package.

This script keeps an editable configuration dictionary for quick runs but
also respects the command-line arguments defined in ``BASE_PPO_EVAL.parse_eval_args``.
"""
from __future__ import annotations

from argparse import ArgumentParser, Namespace
import logging
import sys
import traceback
from typing import Any, Iterable, Sequence

from rwa_wdm.BASE_PPO_EVAL import simulator

__all__ = ["main"]

DEFAULT_CONFIG = {
    "topology": "COST239",
    "channels": 16,
    "load": 150,
    "load_min": 150,
    "load_step": 10,
    "calls": 10000,
    "num_sim": 5,
    "result_dir": "./results_ppo",
    "plot": True,
    "debug_adjacency": False,
    "debug_dijkstra": False,
    "debug_lightpath": False,
    "k": 3,
    "model_path": "tmp/rwa_ppo/rwa_model_1000000_steps.zip",
    "log_dir": "tmp/rwa_ppo",
    "seed": None,
    "deterministic": False,
    "holding_time": 10,
    "env_mode": "offline",
    "external_control": True,
}


def _build_namespace_from_config(config: dict[str, Any]) -> Namespace:
    return Namespace(**config)


def parse_eval_args(argv: Sequence[str] | None = None) -> Namespace:
    parser = ArgumentParser(description='Evaluate a trained PPO agent for the RWA environment')
    parser.add_argument('--topology', default=DEFAULT_CONFIG['topology'], help='Toplogy identifier')
    parser.add_argument('--channels', type=int, default=DEFAULT_CONFIG['channels'], help='Wavelength count per link')
    parser.add_argument('--load', type=int, default=DEFAULT_CONFIG['load'], help='Maximum Erlang load')
    parser.add_argument('--load-min', type=int, default=DEFAULT_CONFIG['load_min'], help='Minimum Erlang load')
    parser.add_argument('--load-step', type=int, default=DEFAULT_CONFIG['load_step'], help='Step between loads')
    parser.add_argument('--calls', type=int, default=DEFAULT_CONFIG['calls'], help='Connection requests per load')
    parser.add_argument('--num-sim', type=int, default=DEFAULT_CONFIG['num_sim'], help='Number of simulations')
    parser.add_argument('--result-dir', default=DEFAULT_CONFIG['result_dir'], help='Directory for results')
    parser.add_argument('--plot', dest='plot', action='store_true', help='Plot blocking probability')
    parser.add_argument('--no-plot', dest='plot', action='store_false', help='Skip plotting')
    parser.add_argument('--debug-adjacency', action='store_true', help='Dump adjacency matrix before running')
    parser.add_argument('--debug-dijkstra', action='store_true', help='Enable Dijkstra debug logging')
    parser.add_argument('--debug-lightpath', action='store_true', help='Enable lightpath debug logging')
    parser.add_argument('--k', type=int, default=DEFAULT_CONFIG['k'], help='Candidate path count requested from k-shortest')
    parser.add_argument('--holding-time', type=int, default=DEFAULT_CONFIG['holding_time'], help='Holding time for each lightpath')
    parser.add_argument('--env-mode', choices=['offline', 'online'], default=DEFAULT_CONFIG['env_mode'], help='Which BASE environment implementation to use')
    parser.add_argument('--model-path', default=DEFAULT_CONFIG['model_path'], help='PPO checkpoint file (.zip)')
    parser.add_argument('--log-dir', default=DEFAULT_CONFIG['log_dir'], help='Directory containing checkpoints')
    parser.add_argument('--seed', type=int, default=DEFAULT_CONFIG['seed'], help='Seed for RNG and env resets')
    parser.add_argument(
        '--external-control',
        action='store_true',
        help='Let caller manage request queue, allocation, and traffic updates externally',
    )
    parser.add_argument('--deterministic', dest='deterministic', action='store_true', help='Force deterministic policy')
    parser.add_argument('--stochastic', dest='deterministic', action='store_false', help='Allow stochastic sampling')
    parser.set_defaults(plot=DEFAULT_CONFIG['plot'], deterministic=DEFAULT_CONFIG['deterministic'])
    return parser.parse_args(argv)


def main(config: dict[str, Any] | None = None, argv: Iterable[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    try:
        if config is not None:
            merged = DEFAULT_CONFIG.copy()
            merged.update(config)
            args = _build_namespace_from_config(merged)
        else:
            args = parse_eval_args(list(argv) if argv is not None else None)
        simulator(args)
        return 0
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
