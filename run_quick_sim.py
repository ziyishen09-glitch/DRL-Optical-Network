"""Small helper script to run the simulator with an embedded config.

This script builds an argparse.Namespace containing the parameters the
simulator expects, validates them using `validate_args`, then calls
`simulator(args)`. Edit DEFAULT_CONFIG to change parameters.
"""
from argparse import Namespace
import traceback
import sys

from rwa_wdm.BASE_Heuristic import simulator as base_simulator
from rwa_wdm.util import validate_args

# Ensure logging is configured so simulator info messages (QKP logs) are visible.
import logging as _logging
if not _logging.getLogger().handlers:
    _logging.basicConfig(level=_logging.INFO, format='[%(levelname)s] %(message)s')

# Default quick-run config — edit as needed
DEFAULT_CONFIG = {
    'topology': 'COST239', 
    'channels': 4,  #according to the paper
    'r': 'dijkstra',
    'w': 'first-fit',
    'y': 3,
    'rwa': None,
    'load':150,
    'load_min': 50,
    'load_step': 10,
    'calls': 10000,
    'result_dir': './results',
    'num_sim': 2,
    'plot': True,
    'debug_adjacency':False,  #是否显示邻接矩阵
    'debug_dijkstra':False, #是否显示dijkstra调试信息
    'debug_lightpath':False, #是否显示lightpath调试信息
    'plot_topo':True,
    'runner': 'base_no_upd',  # 'base_no_upd' or 'fb_no_upd' 
}


def main(config: dict | None = None) -> int:
    try:
        cfg = DEFAULT_CONFIG.copy()
        if config:
            cfg.update(config)

        # Construct Namespace with keys expected by simulator
        args = Namespace(
            topology=cfg['topology'],
            channels=cfg['channels'],
            r=cfg.get('r'),
            w=cfg.get('w'),
            rwa=cfg.get('rwa'),
            y=cfg.get('y'),
            load=cfg['load'],
            load_min=cfg['load_min'],
            load_step=cfg['load_step'],
            calls=cfg['calls'],
            result_dir=cfg['result_dir'],
            num_sim=cfg['num_sim'],
            plot=cfg['plot'],
            debug_adjacency=cfg['debug_adjacency'],
            debug_dijkstra=cfg['debug_dijkstra'],
            debug_lightpath=cfg['debug_lightpath'],
            plot_topo=cfg['plot_topo'],
            runner=cfg['runner'],
            write_qkp_log=cfg.get('write_qkp_log', False),
            write_qkp_usage_log=cfg.get('write_qkp_usage_log', False),
        )

        # Validate and run
        validate_args(args)
        if args.runner == 'base_no_upd':
            simulator = base_simulator
        else:
            raise ValueError('Invalid runner specified: %s' % args.runner)
        simulator(args)
        return 0
    
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
 