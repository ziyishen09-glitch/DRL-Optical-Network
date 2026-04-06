"""I/O related operations such as R/W data from disk and viz-related ops

"""

import glob
import logging
import math
import os
from typing import Callable, List

import numpy as np
import matplotlib.pyplot as plt

__all__ = (
    'write_bp_to_disk',
    'write_it_to_disk',
    'write_SP_A_to_disk',
    'write_SP_R_to_disk',
    'write_rutil_to_disk',
    'write_sbp_to_disk',
    'plot_bp',
    'plot_sbp',
    'plot_sp_a',
    'plot_sp_r',
    'plot_rutil',
)

logger = logging.getLogger(__name__)


def write_bp_to_disk(result_dir: str,
                     filename: str, bplist: List[float]) -> None:
    """Writes blocking probabilities to text file

    Args:
        result_dir: directory to write files to
        filename: name of the file to be written
        itlist: list of blocking probability values, as percentages, to be
            dumped to file

    """
    if not os.path.isdir(result_dir):
        logger.info('Creating result dir in %s' % result_dir)
        os.mkdir(result_dir)

    filepath = os.path.join(result_dir, filename)
    logger.info('Writing blocking probability results to file "%s"' % filepath)
    with open(filepath, 'a') as f:
        for bp in bplist:
            f.write(' %7.3f' % bp)
        f.write('\n')
        
def write_SP_A_to_disk(result_dir: str,
                     filename: str, SPA: List[float]) -> None:
    """Writes resource utilization to text file

    Args:
        result_dir: directory to write files to
        filename: name of the file to be written
        itlist: list of blocking probability values, as percentages, to be
            dumped to file

    """
    if not os.path.isdir(result_dir):
        logger.info('Creating result dir in %s' % result_dir)
        os.mkdir(result_dir)

    filepath = os.path.join(result_dir, filename)
    logger.info('Writing sp_a results to file "%s"' % filepath)
    with open(filepath, 'a') as f:
        for spa in SPA:
            f.write(' %7.3f' % spa)
        f.write('\n')

def write_SP_R_to_disk(result_dir: str,
                     filename: str, SPR: List[float]) -> None:
    """Writes resource utilization to text file

    Args:
        result_dir: directory to write files to
        filename: name of the file to be written
        itlist: list of blocking probability values, as percentages, to be
            dumped to file

    """
    if not os.path.isdir(result_dir):
        logger.info('Creating result dir in %s' % result_dir)
        os.mkdir(result_dir)

    filepath = os.path.join(result_dir, filename)
    logger.info('Writing resource utilization results to file "%s"' % filepath)
    with open(filepath, 'a') as f:
        for spr in SPR:
            f.write(' %7.3f' % spr)
        f.write('\n')



def write_it_to_disk(result_dir: str,
                     filename: str, itlist: List[float]) -> None:
    """Writes profiling time information to text file

    Args:
        result_dir: directory to write files to
        filename: name of the file to be written
        itlist: list of times, in seconds, to be dumped to file

    """
    if not os.path.isdir(result_dir):
        logger.info('Creating result dir in %s' % result_dir)
        os.mkdir(result_dir)

    filepath = os.path.join(result_dir, filename)
    logger.info('Writing simulation profiling times to file "%s"' % filepath)
    with open(filepath, 'a') as f:
        for it in itlist:
            f.write(' %7.7f' % it)
        # ensure each run ends with a newline so files contain one run per line
        f.write('\n')


def write_rutil_to_disk(result_dir: str,
                        filename: str, rutil: List[float]) -> None:
    """Writes resource utilization (proportion) to text file

    Args:
        result_dir: directory to write files to
        filename: name of the file to be written
        rutil: list of utilization values (proportions in [0,1])
    """
    if not os.path.isdir(result_dir):
        logger.info('Creating result dir in %s' % result_dir)
        os.mkdir(result_dir)

    filepath = os.path.join(result_dir, filename)
    logger.info('Writing resource utilization results to file "%s"' % filepath)
    with open(filepath, 'a') as f:
        for val in rutil:
            f.write(' %7.5f' % val)
        f.write('\n')


def write_sbp_to_disk(result_dir: str, filename: str, sbp: List[float]) -> None:
    """Writes single-link blocking probabilities (percent) to disk.

    Args:
        result_dir: directory where files are written.
        filename: name of the file to append to.
        sbp: list of blocking percentages, one per load.
    """
    if not os.path.isdir(result_dir):
        logger.info('Creating result dir in %s' % result_dir)
        os.mkdir(result_dir)

    filepath = os.path.join(result_dir, filename)
    logger.info('Writing single-link blocking stats to file "%s"' % filepath)
    with open(filepath, 'a') as f:
        for rate in sbp:
            if rate is None or math.isnan(rate):
                f.write('    - ')
            else:
                f.write(' %7.3f' % rate)
        f.write('\n')


def _default_token_parser(value: str) -> float | None:
    """Parse a token into a float, returning None if parsing fails."""
    try:
        return float(value)
    except ValueError:
        return None


def _sbp_token_parser(value: str) -> float | None:
    """Parse SBP tokens, treating '-' or 'nan' as NaN."""
    stripped = value.strip()
    if not stripped:
        return None
    if stripped == '-' or stripped.lower() == 'nan':
        return math.nan
    try:
        return float(stripped)
    except ValueError:
        return None


def plot_bp(
    result_dir: str,
    load_min: int = 1,
    load_max: int | None = None,
    load_step: int = 1,
    filename_prefix: str | None = None,
) -> None:
    """Plot mean blocking probability curves stored in `.bp` files.

    Args:
        result_dir: directory containing the `.bp` files.
        load_min / load_max / load_step: range to associate with each column.
        filename_prefix: optional prefix to filter filenames.
    """
    prefix = filename_prefix or ''
    pattern = f"{prefix}*.bp"
    _generic_plot_from_files(
        result_dir,
        pattern,
        'Blocking probability (%)',
        'Average mean blocking probability',
        load_min,
        load_max,
        load_step,
    )


def _generic_plot_from_files(
    result_dir: str,
    pattern: str,
    ylabel: str,
    title: str,
    load_min: int = 1,
    load_max: int | None = None,
    load_step: int = 1,
    ylim_zero: bool = False,
    token_parser: Callable[[str], float | None] | None = None,
) -> None:
    filelist = []
    for f in glob.glob(os.path.join(result_dir, pattern)):
        filelist.append(os.path.basename(f))
        rows: list[list[float]] = []
        parser = token_parser or _default_token_parser
        with open(f, 'r') as fh:
            for line in fh:
                parts = line.strip().split()
                if not parts:
                    continue
                nums: list[float] = []
                for part in parts:
                    value = parser(part)
                    if value is None:
                        nums = []
                        break
                    nums.append(value)
                if not nums:
                    continue
                rows.append(nums)

        if not rows:
            continue

        max_cols = max(len(r) for r in rows)
        data = np.full((len(rows), max_cols), np.nan)
        for i, r in enumerate(rows):
            data[i, :len(r)] = r

        if load_max is not None:
            if load_max < load_min:
                logger.warning('load_max < load_min, ignoring load_max')
            else:
                expected_cols = int((load_max - load_min) // load_step + 1)
                if expected_cols > data.shape[1]:
                    new_data = np.full((data.shape[0], expected_cols), np.nan)
                    new_data[:, : data.shape[1]] = data
                    data = new_data
                elif expected_cols < data.shape[1]:
                    data = data[:, :expected_cols]

        ncols = data.shape[1]
        x = load_min + np.arange(ncols) * load_step

        if data.shape[0] == 1:
            y = data[0, :]
        else:
            y = np.nanmean(data, axis=0)

        plt.plot(x, y, '--', linewidth=2, marker='o', markersize=6)

        half = max(0.5, load_step / 2.0)
        plt.xlim(load_min - half, load_min + (ncols - 1) * load_step + half)
        if data.shape[0] < 10:
            logger.warning(
                'Remember you should simulate at least 10 times '
                '(found only %d in %s)' % (data.shape[0], f)
            )

    if not filelist:
        logger.warning('No files found for pattern %s in %s' % (pattern, result_dir))
        return

    plt.grid()
    if ylim_zero:
        plt.ylim(bottom=0)
    plt.ylabel(ylabel, fontsize=18)
    plt.xlabel('Load (Erlangs)', fontsize=18)
    plt.title(title, fontsize=20)
    plt.legend(filelist)
    plt.show(block=True)


def plot_sbp(
    result_dir: str,
    load_min: int = 1,
    load_max: int | None = None,
    load_step: int = 1,
    filename_prefix: str | None = None,
) -> None:
    """Plot single-link blocking probabilities aggregated in `.sbp` files."""
    prefix = filename_prefix or ''
    pattern = f"{prefix}*.sbp"
    _generic_plot_from_files(
        result_dir,
        pattern,
        'Single-link blocking probability (%)',
        'Average single-link blocking probability',
        load_min,
        load_max,
        load_step,
        token_parser=_sbp_token_parser,
    )


def plot_sp_a(result_dir: str, load_min: int = 1, load_max: int | None = None, load_step: int = 1) -> None:
    """Plot SP_A results from `.spa` files."""
    _generic_plot_from_files(result_dir, '*.spa', 'SP_A (proportion)', 'Average SP_A', load_min, load_max, load_step)


def plot_sp_r(result_dir: str, load_min: int = 1, load_max: int | None = None, load_step: int = 1) -> None:
    """Plot SP_R results from `.spr` files."""
    _generic_plot_from_files(result_dir, '*.spr', 'SP_R (proportion)', 'Average SP_R', load_min, load_max, load_step)


def plot_rutil(result_dir: str, load_min: int = 1, load_max: int | None = None, load_step: int = 1) -> None:
    """Plot resource utilization results from `.rutil` files."""
    _generic_plot_from_files(result_dir, '*.rutil', 'Utilization (proportion)', 'Average Resource Utilization', load_min, load_max, load_step, ylim_zero=True)
