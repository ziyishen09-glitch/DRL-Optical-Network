import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List

def dijkstra(mat: np.ndarray, s: int, d: int) -> List[int]:
    """Dijkstra routing algorithm

    Args:
        mat: Network's adjacency matrix graph
        s: source node index
        d: destination node index

    Returns:
        :obj:`list` of :obj:`int`: sequence of router indices encoding a path

    """
    if s < 0 or d < 0:
        raise ValueError('Source nor destination nodes cannot be negative')
    elif s > mat.shape[0] or d > mat.shape[0]:
        raise ValueError('Source nor destination nodes should not exceed '
                         'adjacency matrix dimensions')

    G = nx.from_numpy_array(mat, create_using=nx.Graph())
    hops, path = nx.bidirectional_dijkstra(G, s, d, weight=None)
    return path

if __name__ == "__main__":
    G = nx.Graph()
    G.add_nodes_from(range(1, 10))  # 9 nodes
    edges = [
        (1, 8), (8, 3), (8, 9), (3, 5), (3, 9), (5, 7), (5, 9), (7, 4),
        (7, 9), (4, 6), (4, 0), (6, 4), (6, 2), (2, 0), (0, 9), (0, 2),
        (9, 7), (9, 5), (9, 3), (9, 8)
    ]
    G.add_edges_from(edges)

    adjacency_matrix = nx.adjacency_matrix(G).toarray()
    shortest_path = dijkstra(adjacency_matrix, 1, 6)
    communication_edges = [(shortest_path[i], shortest_path[i + 1]) for i in range(len(shortest_path) - 1)]

    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_size=500, node_color="skyblue")

    for edge in edges:
        plt.plot(
            [pos[edge[0]][0], pos[edge[1]][0]],
            [pos[edge[0]][1], pos[edge[1]][1]],
            'k-',
            alpha=0.3,
        )

    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=communication_edges,
        width=3,
        edge_color='r',
    )

    plt.axis("off")
    plt.show()
