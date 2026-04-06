"""Plot how much bandwidth PPO saves over a heuristic using .bp files."""
from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np


def parse_loads(text: str) -> list[int]:
    if not text:
        raise ValueError("load list must not be empty")
    values: list[int] = []
    for chunk in text.split(','):
        chunk = chunk.strip()
        if not chunk:
            continue
        values.append(int(chunk))
    if not values:
        raise ValueError("at least one load is required")
    return values


def parse_args(argv: Sequence[str] | None = None):
    parser = ArgumentParser(description="Plot PPO's saved bandwidth over a heuristic")
    parser.add_argument(
        "--result-dir",
        default="results",
        help="Directory containing the `.bp` files",
    )
    parser.add_argument(
        "--heuristic-file",
        default="BASE_ksp_first-fit_8ch.bp",
        help="Filename of the heuristic .bp dump",
    )
    parser.add_argument(
        "--ppo-file",
        default="PPO_ppo_env_8ch.bp",
        help="Filename of the PPO .bp dump",
    )
    parser.add_argument("--load-min", type=int, default=50, help="Load value for the first column")
    parser.add_argument("--load-step", type=int, default=10, help="Load increment between columns")
    parser.add_argument(
        "--load-max",
        type=int,
        default=None,
        help="Maximum load value to render (optional)",
    )
    parser.add_argument(
        "--requests",
        type=int,
        default=10000,
        help="Number of setup requests issued per run",
    )
    parser.add_argument(
        "--channel-rate",
        type=float,
        default=10.0,
        help="Per-wavelength rate in Gb/s",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=10.0,
        help="Duration of the measurement window in seconds",
    )
    parser.add_argument("--loads", help="Comma separated list of loads; overrides load-min/step/max")
    parser.add_argument("--dpi", type=int, default=120, help="DPI for the saved figure")
    parser.add_argument("--output", help="File path for saving the figure")
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not display the matplotlib window",
    )
    return parser.parse_args(argv)


def read_bp_file(path: Path) -> list[float]:
    text = path.read_text().splitlines()
    rows: list[list[float]] = []
    for line in text:
        parts = line.strip().split()
        if not parts:
            continue
        try:
            row = [float(value) for value in parts]
        except ValueError:
            continue
        rows.append(row)

    if not rows:
        raise ValueError(f"{path} does not contain any data")

    max_cols = max(len(row) for row in rows)
    data = np.full((len(rows), max_cols), np.nan)
    for i, row in enumerate(rows):
        data[i, : len(row)] = row

    if data.shape[0] == 1:
        return data[0, :].tolist()
    return np.nanmean(data, axis=0).tolist()


def build_load_axis(
    load_min: int, load_step: int, ncols: int, load_max: int | None = None
) -> list[int]:
    if load_step <= 0:
        raise ValueError("load_step must be positive")
    loads = [load_min + i * load_step for i in range(ncols)]
    if load_max is not None and loads and loads[-1] > load_max:
        loads = [value for value in loads if value <= load_max]
    return loads


def compute_saved_gbits(
    heuristic: Iterable[float],
    ppo: Iterable[float],
    requests: int,
    channel_rate_gbps: float,
    duration_s: float,
) -> list[float]:
    saved = []
    for h_val, p_val in zip(heuristic, ppo):
        delta_percent = h_val - p_val
        saved_requests = delta_percent / 100.0 * requests
        saved_bits = saved_requests * channel_rate_gbps * 1e9 * duration_s
        saved.append(saved_bits / 1e9)
    return saved


def plot_saved_resource(load_axis: list[int], saved_gbits: list[float], dpi: int) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 6), dpi=dpi)
    ax.plot(load_axis, saved_gbits, '-o', linewidth=2, markersize=6, label='Saved blocked resource')
    ax.set_xlabel('Load (Erlangs)')
    ax.set_ylabel('Saved capacity (Gbit/10000 requests)')
    ax.set_title('PPO saved resource compared to heuristic')
    ax.grid(True, linestyle='--', alpha=0.6)
    for x, y in zip(load_axis, saved_gbits):
        ax.text(x, y, f'{y:.1f}', ha='center', va='bottom', fontsize=8)
    ax.legend()
    fig.tight_layout()
    return fig


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    result_dir = Path(args.result_dir)
    heuristic_path = result_dir / args.heuristic_file
    ppo_path = result_dir / args.ppo_file
    if not heuristic_path.exists():
        raise FileNotFoundError(f"Heuristic file not found: {heuristic_path}")
    if not ppo_path.exists():
        raise FileNotFoundError(f"PPO file not found: {ppo_path}")

    if args.loads:
        load_axis = parse_loads(args.loads)
        ncols = len(load_axis)
    else:
        heuristic = read_bp_file(heuristic_path)
        ppo = read_bp_file(ppo_path)
        ncols = min(len(heuristic), len(ppo))
        if args.load_max is not None and args.load_max < args.load_min:
            raise ValueError("load_max must be greater than or equal to load_min")
        if args.load_max is not None:
            expected_cols = int((args.load_max - args.load_min) // args.load_step + 1)
            ncols = min(ncols, expected_cols)
        load_axis = build_load_axis(args.load_min, args.load_step, ncols, args.load_max)
        heuristic = heuristic[: len(load_axis)]
        ppo = ppo[: len(load_axis)]
        saved_gbits = compute_saved_gbits(
            heuristic,
            ppo,
            args.requests,
            args.channel_rate,
            args.duration,
        )
        fig = plot_saved_resource(load_axis, saved_gbits, args.dpi)
        if args.output:
            fig.savefig(args.output, dpi=args.dpi)
            print(f"Saved figure to {args.output}")
        if not args.no_show:
            plt.show()
        return

    heuristic = read_bp_file(heuristic_path)
    ppo = read_bp_file(ppo_path)
    if len(load_axis) > min(len(heuristic), len(ppo)):
        raise ValueError("Explicit loads list is longer than the available columns in the .bp files")
    heuristic = [heuristic[i] for i in range(len(load_axis))]
    ppo = [ppo[i] for i in range(len(load_axis))]
    saved_gbits = compute_saved_gbits(
        heuristic,
        ppo,
        args.requests,
        args.channel_rate,
        args.duration,
    )
    fig = plot_saved_resource(load_axis, saved_gbits, args.dpi)
    if args.output:
        fig.savefig(args.output, dpi=args.dpi)
        print(f"Saved figure to {args.output}")
    if not args.no_show:
        plt.show()


if __name__ == '__main__':
    main()
