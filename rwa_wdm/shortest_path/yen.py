import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from typing import List

def yen(mat: np.ndarray, s: int, d: int, k: int) -> List[List[int]]:
    """Yen's routing algorithm, a.k.a. K-shortest paths

    Args:
        mat: Network's adjacency matrix graph
        s: source node index
        d: destination node index
        k: number of alternate paths

    Returns:
        :obj:`list` of :obj:`list`: a sequence of `k` paths

    """
    if s < 0 or d < 0:
        raise ValueError('Source nor destination nodes cannot be negative')
    elif s > mat.shape[0] or d > mat.shape[0]:
        raise ValueError('Source nor destination nodes should exceed '
                         'adjacency matrix dimensions')
    if k < 0:
        raise ValueError('Number of alternate paths should be positive')

    G = nx.from_numpy_array(mat, create_using=nx.Graph())
    paths = list(nx.shortest_simple_paths(G, s, d, weight=None))
    return paths[:k]

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
    k_shortest_paths = yen(adjacency_matrix, 1, 6, 2)

    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_size=500, node_color="skyblue")

    for edge in edges:
        plt.plot([
            pos[edge[0]][0], pos[edge[1]][0]
        ], [
            pos[edge[0]][1], pos[edge[1]][1]
        ], 'k-', alpha=0.3)

    colors = ['r', 'b']
    for i, path in enumerate(k_shortest_paths):
        communication_edges = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=communication_edges,
            width=3,
            edge_color=colors[i],
        )

    plt.axis("off")
    plt.show()
