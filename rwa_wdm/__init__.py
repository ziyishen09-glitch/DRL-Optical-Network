"""rwa_wdm is a Python simulator package aiming to solve the routing and
wavelength assignment (RWA) problem over wavelength division multiplexing (WDM)
optical networks.

"""
import logging
import numpy as _np

if not hasattr(_np, 'float_'):
    _np.float_ = _np.float64

from .BASE_Heuristic import simulator

# https://stackoverflow.com/questions/15727420/using-logging-in-multiple-modules/15729700#15729700
logging.basicConfig(
    format='[%(asctime)s] [%(module)s] %(levelname)-8s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO)

# https://stackoverflow.com/questions/35325042/python-logging-disable-logging-from-imported-modules
logging.getLogger('matplotlib').setLevel(logging.WARNING)

__version__ = '0.2.1'
__author__ = 'Cassio Batista'
