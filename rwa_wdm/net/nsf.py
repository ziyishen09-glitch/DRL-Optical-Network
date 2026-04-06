from typing import Dict, List, Tuple
from collections import OrderedDict

from . import Network


class NationalScienceFoundation(Network):
    """U.S. National Science Foundation Network (NSFNET)"""

    def __init__(self, ch_n):
        self._name = 'nsf'
        self._fullname = u'National Science Foundation'
        self._s = 1
        self._d = 9
        super().__init__(ch_n,
                         len(self.get_nodes_2D_pos()),
                         len(self.get_edges()))

    def get_edges(self) -> List[Tuple[int, int]]:
        """get"""

        return [
            (1, 8), (8, 3), (8, 9), (3, 5), (3, 9), (5, 7), (5, 9), (7, 4), (7, 9), (4, 6), (4, 0), (6, 4), (6, 2), (2, 0), (0, 9), (0, 2), (9, 7), (9, 5), (9, 3), (9, 8)
        ]

    def get_nodes_2D_pos(self) -> Dict[str, Tuple[float, float]]:
        """Get position of the nodes on the bidimensional Cartesian plan"""

        return OrderedDict([
            ('0', (0.01322179, -0.42508968)),   # 0
            ('1', (-0.81033115, 1.0)),   # 1
            ('2', (0.07394747, -0.9297942)),   # 2
            ('3', (-0.10759903, 0.59666911)),   # 3
            ('4', (0.40306749, -0.50795438)),   # 4
            ('5', (0.21384686, 0.40213917)),   # 5
            ('6', (0.44676966, -0.97552313)),   # 6
            ('7', (0.31675486, -0.00359664)),   # 7
            ('8', (-0.46077252, 0.64851051)),   # 8
            ('9', (-0.08890543, 0.19463922)),   # 9
            
        ])



# from typing import Dict, List, Tuple
# from collections import OrderedDict

# from . import Network


# class NationalScienceFoundation(Network):
#     """U.S. National Science Foundation Network (NSFNET)"""

#     def __init__(self, ch_n):
#         self._name = 'nsf'
#         self._fullname = u'National Science Foundation'
#         self._s = 0
#         self._d = 12
#         super().__init__(ch_n,
#                          len(self.get_nodes_2D_pos()),
#                          len(self.get_edges()))

#     def get_edges(self) -> List[Tuple[int, int]]:
#         """get"""

#         return [
#             (0, 1), (0, 2), (0, 5),
#             (1, 2), (1, 3),
#             (2, 8),
#             (3, 4), (3, 6), (3, 13),
#             (4, 9),
#             (5, 6), (5, 10),
#             (6, 7),
#             (7, 8),
#             (8, 9),
#             (9, 11), (9, 12),
#             (10, 11), (10, 12),
#             (11, 13)
#         ]

#     def get_nodes_2D_pos(self) -> Dict[str, Tuple[float, float]]:
#         """Get position of the nodes on the bidimensional Cartesian plan"""

#         return OrderedDict([
#             ('0', (0.70, 2.70)),   # 0
#             ('1', (1.20, 1.70)),   # 1
#             ('2', (1.00, 4.00)),   # 2
#             ('3', (3.10, 1.00)),   # 3
#             ('4', (4.90, 0.70)),   # 4
#             ('5', (2.00, 2.74)),   # 5
#             ('6', (2.90, 2.66)),   # 6
#             ('7', (3.70, 2.80)),   # 7
#             ('8', (4.60, 2.80)),   # 8
#             ('9', (5.80, 3.10)),   # 9
#             ('10', (5.50, 3.90)),  # 10
#             ('11', (6.60, 4.60)),  # 11
#             ('12', (7.40, 3.30)),  # 12
#             ('13', (6.50, 2.40))   # 13
#         ])
