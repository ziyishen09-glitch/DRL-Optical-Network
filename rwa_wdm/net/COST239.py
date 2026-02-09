from collections import OrderedDict
from typing import Dict, List, Tuple

from . import Network


class COST239(Network):
    """COST239 pan-European backbone topology."""

    def __init__(self, num_channels: int):
        self._name = 'cost239'
        self._fullname = 'COST239 European backbone'
        self._s = 0
        self._d = 1
        super().__init__(num_channels, len(self.get_nodes_2D_pos()), len(self.get_edges()))

    def get_edges(self) -> List[Tuple[int, int, int]]:
        return [
            (0, 1, 1),  # London-Amsterdam
            (0, 4, 1),  # London-Brussels
            (0, 7, 1),  # London-Paris
            (0, 2, 1),  # London-Copenhagen
            (1, 2, 1),  # Amsterdam-Copenhagen
            (1, 3, 1),  # Amsterdam-Berlin
            (1, 5, 1),  # Amsterdam-Luxembourg
            (1, 4, 1),  # Amsterdam-Brussels
            (2, 3, 1),  # Copenhagen-Berlin
            (2, 6, 1),  # Copenhagen-Prague
            (3, 6, 1),  # Berlin-Prague
            (3, 10, 1), # Berlin-Vienna
            (3, 7, 1),  # Berlin-Paris
            (4, 7, 1),  # Brussels-Paris
            (4, 5, 1),  # Brussels-Luxembourg
            (4, 6, 1),  # Brussels-Prague
            (4, 9, 1),  # Brussels-Milan
            (5, 7, 1),  # Luxembourg-Paris
            (5, 6, 1),  # Luxembourg-Prague
            (5, 8, 1),  # Luxembourg-Zurich
            (6, 8, 1),  # Prague-Zurich
            (6, 10, 1), # Prague-Vienna
            (7, 9, 1),  # Paris-Milan
            (7, 8, 1),  # Paris-Zurich
            (8, 9, 1),  # Zurich-Milan
            (9, 10, 1), # Milan-Vienna
            (8, 10, 1), # Zurich-Vienna
        ]

    def get_nodes_2D_pos(self) -> Dict[str, Tuple[float, float]]:
        return OrderedDict([
            ('0', (-0.7, 1.6)),   # London
            ('1', (-0.2, 1.6)),   # Amsterdam
            ('2', (0.3, 2.0)),    # Copenhagen
            ('3', (0.8, 1.6)),    # Berlin
            ('4', (-0.5, 0.9)),   # Brussels
            ('5', (0.0, 0.9)),    # Luxembourg
            ('6', (0.5, 0.9)),    # Prague
            ('7', (-0.7, 0.2)),   # Paris
            ('8', (0.2, 0.2)),    # Zurich
            ('10', (0.7, 0.2)),   # Vienna
            ('9', (0.2, -0.4)),   # Milan
        ])

