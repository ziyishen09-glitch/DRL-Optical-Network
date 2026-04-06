
"""Small helper to plot simulation results on demand.

Usage examples:
    python plotter.py --result-dir results --prefix BASE_ --plots all
    python plotter.py --result-dir results --prefix PB_ --plots bp,spa
"""
from __future__ import annotations
import argparse
import sys
from typing import Sequence


def parse_args(argv: Sequence[str] | None = None):
    p = argparse.ArgumentParser(description='Plot simulation results (wrapper around rwa_wdm.io)')
    p.add_argument('--result-dir', default='results', help='Directory where result files are stored')
    p.add_argument('--prefix', default=None, help="Filename prefix for a simulator variant, e.g. 'BASE_' or 'PB_'. If omitted, all files will be considered.")
    p.add_argument('--plots', default='all', help="Comma-separated list of plots to draw: bp,spa,spr,rutil or 'all' (default: all)")
    p.add_argument('--load-min', type=int, default=1, help='Minimum load when plotting (used by plotters)')
    p.add_argument('--load-max', type=int, default=10, help='Maximum load when plotting (used by plotters)')
    p.add_argument('--load-step', type=int, default=1, help='Load step when plotting (used by plotters)')
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None):
    args = parse_args(argv)
    # If the user didn't override load range flags (they are the plotter's
    # defaults), try to read the DEFAULT_CONFIG from run_quick_sim.py so the
    # plotter uses the same load_min/load_max/load_step used by quick-run.
    try:
        default_flags = (args.load_min == 1 and args.load_max == 10 and args.load_step == 1)
    except Exception:
        default_flags = False
    if default_flags:
        try:
            import run_quick_sim
            cfg = getattr(run_quick_sim, 'DEFAULT_CONFIG', None)
            if isinstance(cfg, dict):
                # map keys: load_min, load, load_step
                args.load_min = cfg.get('load_min', args.load_min)
                args.load_max = cfg.get('load', args.load_max)
                args.load_step = cfg.get('load_step', args.load_step)
        except Exception:
            # if import fails, just keep the CLI defaults
            pass

    try:
        from rwa_wdm.io import plot_bp, plot_sbp, plot_sp_a, plot_sp_r
        try:
            from rwa_wdm.io import plot_rutil
        except Exception:
            plot_rutil = None
    except Exception as exc:
        print('Failed to import plotting utilities from rwa_wdm.io:', exc)
        sys.exit(2)

    wanted = {p.strip().lower() for p in args.plots.split(',')} if args.plots else {'all'}
    if 'all' in wanted:
        wanted = {'bp', 'sbp', 'spa', 'spr', 'rutil'}

    kw = dict(load_min=args.load_min, load_max=args.load_max, load_step=args.load_step)

    if 'bp' in wanted:
        try:
            plot_bp(args.result_dir, filename_prefix=args.prefix, **kw)
        except TypeError:
            # fallback if older signature doesn't accept filename_prefix
            plot_bp(args.result_dir, **kw)
    if 'sbp' in wanted:
        try:
            plot_sbp(args.result_dir, filename_prefix=args.prefix, **kw)
        except TypeError:
            plot_sbp(args.result_dir, **kw)
    if 'spa' in wanted:
        try:
            plot_sp_a(args.result_dir, filename_prefix=args.prefix, **kw)
        except TypeError:
            plot_sp_a(args.result_dir, **kw)
    if 'spr' in wanted:
        try:
            plot_sp_r(args.result_dir, filename_prefix=args.prefix, **kw)
        except TypeError:
            plot_sp_r(args.result_dir, **kw)
    if 'rutil' in wanted and plot_rutil is not None:
        try:
            plot_rutil(args.result_dir, filename_prefix=args.prefix, **kw)
        except TypeError:
            plot_rutil(args.result_dir, **kw)

    print('Plotting complete.')


if __name__ == '__main__':
    main()
