import heapq
import math
import networkx as nx
import matplotlib.pyplot as plt

def shortest_path(G, start, goal):
    def distance(node1, node2):
        pos1 = pos[node1]
        pos2 = pos[node2]
        return math.sqrt((pos2[0] - pos1[0])**2 + (pos2[1] - pos1[1])**2)
    
    # Initialisation des variables
    node_before = dict()
    explored = {start}
    g_cost = {start: 0}
    f_total = {distance(start, goal): start}
    frontier_heap = list(f_total.keys())
    heapq.heapify(frontier_heap)
    
    while len(frontier_heap) > 0:
        current_node = f_total[heapq.heappop(frontier_heap)]
        
        if current_node == goal:
            return construct_path(node_before, current_node)
        else:
            for next_node in G.neighbors(current_node):
                neighboor_g_value = g_cost[current_node] + G[current_node][next_node].get('weight', 1)  # Use edge weight if available
                if next_node not in g_cost or neighboor_g_value < g_cost[next_node]:
                    node_before[next_node] = current_node
                    g_cost[next_node] = neighboor_g_value
                    f_value = g_cost[next_node] + distance(next_node, goal)
                    f_total[f_value] = next_node
                    heapq.heappush(frontier_heap, f_value)
    
    return -1 

def construct_path(dictionary, last_node):
    path = [last_node]
    while last_node in dictionary:
        last_node = dictionary[last_node]
        path = [last_node] + path
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

    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_size=500, node_color="skyblue")
    for edge in edges:
        plt.plot(
            [pos[edge[0]][0], pos[edge[1]][0]],
            [pos[edge[0]][1], pos[edge[1]][1]],
            'k-',
            alpha=0.3,
        )

    shortest_path_result = shortest_path(G, 1, 6)
    print("Shortest Path:", shortest_path_result)

    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=[
            (shortest_path_result[i], shortest_path_result[i + 1])
            for i in range(len(shortest_path_result) - 1)
        ],
        width=3,
        edge_color='r',
    )

    plt.axis("off")
    plt.show()
