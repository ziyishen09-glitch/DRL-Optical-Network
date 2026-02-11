import argparse
import logging
import os
import sys
import torch
from argparse import Namespace
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import VecMonitor

from rwa_wdm.BASE_PPO_TRAIN import build_env_factory
from PPO.episode_load import EpisodeLoadScheduler

logger = logging.getLogger(__name__)


def run_training(args: Namespace) -> None:
    os.makedirs(args.log_dir, exist_ok=True)
    vec_env = build_env_factory(args)
    monitor_path = os.path.join(args.log_dir, "monitor.csv")
    vec_env = VecMonitor(vec_env, monitor_path)

    model = PPO(
        "MultiInputPolicy",
        vec_env,
        verbose=1,
        tensorboard_log=os.path.join(args.log_dir, "tensorboard"),
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        batch_size=args.batch_size,
        seed=args.seed,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=args.save_freq,
        save_path=args.log_dir,
        name_prefix="rwa_model",
        save_vecnormalize=True,
    )

    tb_name = "rwa_ppo"
    if args.n_envs > 0:
        tb_name += f"_nenv{args.n_envs}"

    episode_count = args.episodes
    computed_timesteps = episode_count * args.episode_length
    total_timesteps = args.total_timesteps if args.total_timesteps is not None else computed_timesteps
    logger.info(
        "Training with %d episodes (@ %d requests each) -> %d total steps",
        episode_count,
        args.episode_length,
        total_timesteps,
    )

    model.learn(
        total_timesteps=total_timesteps,
        callback=checkpoint_callback,
        tb_log_name=tb_name,
    )

    if args.plot_reward:
        plot_reward_progress(args, args.plot_every)


def plot_reward_progress(args: Namespace, plot_every: int) -> None:
    monitor_file = os.path.join(args.log_dir, "monitor.csv")
    if not os.path.exists(monitor_file):
        logger.warning("No monitor file found at %s, cannot plot rewards", monitor_file)
        return

    rewards: List[float] = []
    try:
        with open(monitor_file, encoding="utf-8") as fh:
            for line in fh:
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                value_str = stripped.split(",", 1)[0]
                if value_str == "r":
                    continue
                try:
                    rewards.append(float(value_str))
                except ValueError:
                    continue
    except Exception as err:
        logger.exception("Failed to read rewards from monitor file: %s", err)
        return

    if not rewards:
        logger.warning("Monitor file %s contains no reward entries", monitor_file)
        return

    block_size = max(1, plot_every)
    blocks: List[float] = []
    for idx in range(0, len(rewards), block_size):
        blocks.append(sum(rewards[idx:idx + block_size]))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, len(blocks) + 1), blocks, marker="o")
    ax.set_xlabel(f"Block ({block_size} episodes)")
    ax.set_ylabel("Total reward")
    ax.set_title("Training reward per block")
    ax.grid(True)
    fig.tight_layout()

    plot_path = os.path.join(args.log_dir, "reward_progress.png")
    try:
        if os.path.exists(plot_path):
            os.remove(plot_path)
    except OSError:
        pass
    fig.savefig(plot_path)
    plt.close(fig)
    logger.info("Saved reward plot to %s", plot_path)


def build_episode_load_scheduler(args: Namespace) -> Optional[EpisodeLoadScheduler]:
    start = args.episode_load_range_min
    stop = args.episode_load_range_max
    step = args.episode_load_range_step
    if start is None or stop is None or step is None:
        return None
    try:
        step_value = float(step)
        if step_value <= 0:
            raise ValueError
    except ValueError:
        logger.warning("Invalid episode-load step %s; skipping load cycling", step)
        return None
    start_value = float(start)
    stop_value = float(stop)
    if start_value > stop_value:
        logger.warning(
            "Episode-load range min (%s) is greater than max (%s); skipping load cycling",
            start_value,
            stop_value,
        )
        return None
    loads: List[float] = []
    current = start_value
    epsilon = step_value * 0.001
    while current <= stop_value + epsilon:
        loads.append(round(current, 6))
        current += step_value
    if not loads:
        return None
    logger.info("Cycling episode loads: %s", loads)
    return EpisodeLoadScheduler(loads)


def parse_args() -> Namespace:
    parser = argparse.ArgumentParser(description="Online PPO training for the RWA environment")
    parser.add_argument("--topology", default="COST239", help="Topology identifier for the network")
    parser.add_argument("--channels", type=int, default=16, help="Number of wavelengths per link")
    parser.add_argument("--episodes", type=int, default=100, help="Total episodes to run; overrides num_sim * loads")
    parser.add_argument("--episode-length", type=int, default=10000, help="Requests per episode before reset")
    parser.add_argument(
        "--episode-load",
        type=float,
        default=150,
        help="Erlang traffic density used when creating each episode's request queue",
    )
    parser.add_argument(
        "--episode-load-range-min",
        type=float,
        default=50,
        help="Lower bound for cycling episode loads (inclusive)",
    )
    parser.add_argument(
        "--episode-load-range-max",
        type=float,
        default=150,
        help="Upper bound for cycling episode loads (inclusive)",
    )
    parser.add_argument(
        "--episode-load-range-step",
        type=float,
        default=10,
        help="Positive step between loads when cycling",
    )
    parser.add_argument("--n-envs", type=int, default=4, help="Number of vectorized environments")
    parser.add_argument("--total-timesteps", type=int, default=None, help="Timesteps for training; inferred from episodes when omitted")
    parser.add_argument("--save-freq", type=int, default=50000, help="Checkpoint frequency")
    parser.add_argument("--log-dir", default="tmp/rwa_ppo", help="Directory for logs and checkpoints")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="Learning rate for PPO")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--plot-reward", dest="plot_reward", action="store_true", help="Save reward plot after training")
    parser.add_argument("--no-plot-reward", dest="plot_reward", action="store_false", help="Skip saving the reward plot")
    parser.add_argument("--env-mode",
                        choices=["offline", "online"],
                        default="online",
                        help="Which BASE environment implementation to use")
    parser.add_argument(
        "--auto-manage-resources",
        dest="auto_manage_resources",
        action="store_true",
        help="Let the BASE environment advance traffic and perform allocations automatically",
    )
    parser.add_argument(
        "--no-auto-manage-resources",
        dest="auto_manage_resources",
        action="store_false",
        help="Let the training runner control allocations and traffic advancement manually",
    )
    parser.set_defaults(plot_reward=True, auto_manage_resources=True)
    parser.add_argument("--plot-every", type=int, default=1, help="Episodes per block when plotting rewards")
    parser.add_argument("--batch-size", type=int, default=256, help="PPO minibatch size")
    parser.add_argument("--k", type=int, default=3, help="Candidate path count requested by the env")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.episode_load_scheduler = build_episode_load_scheduler(args)
    run_training(args)


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO)
    main()
