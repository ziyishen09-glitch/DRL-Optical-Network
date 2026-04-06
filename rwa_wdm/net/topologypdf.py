from typing import Dict, List, Tuple
from collections import OrderedDict

from . import Network  # Assurez-vous que le module Network est importé correctement depuis votre package
class MyTopology(Network):
    def __init__(self, ch_n):
        self._name = 'pdf'
        self._fullname = u'pdf'
        self._s = 1
        self._d = 6
        super().__init__(ch_n, len(self.get_nodes_2D_pos()), len(self.get_edges()))

    def get_edges(self) -> List[Tuple[int, int]]:
        """Définition des liens dans votre topologie"""
        return [
            (1, 8), (8, 3), (8, 9), (3, 5), (5, 7), (5, 9), (7, 4),
              (4, 0), (6, 4), (6, 2), (2, 0), (0, 9),  (9, 7), 
            (9, 3)
        ]

    def get_nodes_2D_pos(self) -> Dict[int, Tuple[float, float]]:
        """Position des nœuds dans votre topologie"""
        return OrderedDict([
            (0, (0.70, 2.70)), (1, (1.20, 1.70)), (2, (1.00, 4.00)),
            (3, (3.10, 1.00)), (4, (4.90, 0.70)), (5, (2.00, 2.74)),
            (6, (2.90, 2.66)), (7, (3.70, 2.80)), (8, (4.60, 2.80)),
            (9, (5.80, 3.10))
        ])
