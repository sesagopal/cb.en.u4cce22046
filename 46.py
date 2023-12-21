# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 10:32:55 2023

@author: hp
"""

class ShortestPathFinder:
    def __init__(self, vertices):
        self.V = vertices
        self.graph = [[0 for _ in range(vertices)] for _ in range(vertices)]

    def print_shortest_paths(self, distances):
        print("Vertex \tDistance from Source")
        for node, distance in enumerate(distances):
            print(node, "\t\t ", distance)
        print()

    def find_min_distance_vertex(self, distances, visited):
        min_distance = float('inf')
        min_vertex = -1

        for v in range(self.V):
            if distances[v] < min_distance and not visited[v]:
                min_distance = distances[v]
                min_vertex = v

        return min_vertex

    def dijkstra_algorithm(self, source):
        distances = [float('inf')] * self.V
        distances[source] = 0
        visited = [False] * self.V

        for _ in range(self.V):
            current_vertex = self.find_min_distance_vertex(distances, visited)
            visited[current_vertex] = True

            for neighbor_vertex in range(self.V):
                if (
                    not visited[neighbor_vertex]
                    and self.graph[current_vertex][neighbor_vertex] > 0
                    and distances[neighbor_vertex] > distances[current_vertex] + self.graph[current_vertex][
                        neighbor_vertex]
                ):
                    distances[neighbor_vertex] = distances[current_vertex] + self.graph[current_vertex][neighbor_vertex]

        self.print_shortest_paths(distances)


# Test Case 1
graph1 = ShortestPathFinder(6)
graph1.graph = [
    [0, 5, 2, 0, 4, 0],
    [5, 0, 5, 0, 0, 0],
    [2, 5, 0, 0, 0, 0],
    [0, 3, 0, 0, 0, 2],
    [4, 0, 0, 0, 0, 0],
    [0, 0, 0, 3, 0, 0]
]
print("Test Case 1:")
graph1.dijkstra_algorithm(5)

# Test Case 2
graph2 = ShortestPathFinder(5)
graph2.graph = [
    [0, 4, 0, 0, 0],
    [4, 0, 7, 0, 0],
    [0, 7, 0, 3, 0],
    [0, 0, 3, 0, 5],
    [0, 0, 0, 5, 0]
]
print("Test Case 2:")
graph2.dijkstra_algorithm(0)


INF = float('inf')

def floyd_warshall(graph):
    vertices = len(graph)
    distance_matrix = [row[:] for row in graph]

    for k in range(vertices):
        for i in range(vertices):
            for j in range(vertices):
                if distance_matrix[i][k] + distance_matrix[k][j] < distance_matrix[i][j]:
                    distance_matrix[i][j] = distance_matrix[i][k] + distance_matrix[k][j]

    print_solution(distance_matrix)

def print_solution(distance_matrix):
    print("Shortest Distances between all pairs of vertices:")
    for row in distance_matrix:
        print(row)

# Test Case 1
graph1 = [
    [0, 6, INF, 0, 8, 0],
    [6, 0, 7, 0, 0, 0],
    [3, 7, 0, 0, 0, 0],
    [0, 4, 0, 0, 0, 5],
    [8, 0, 0, 0, 0, 0],
    [0, 0, 0, 9, 0, 0]
]
print("Test Case 1:")
floyd_warshall(graph1)

# Test Case 2
graph2 = [
    [0, 3, INF, INF, INF],
    [3, 0, 8, INF, INF],
    [INF, 8, 0, 4, INF],
    [INF, INF, 4, 0, 6],
    [INF, INF, INF, 6, 0]
]
print("Test Case 2:")
floyd_warshall(graph2)


from collections import defaultdict, deque

class Graph:
    def __init__(self):
        self.graph = defaultdict(list)

    def add_edge(self, u, v):
        self.graph[u].append(v)

    def bfs(self, start_node):
        visited = set()
        queue = deque([start_node])

        while queue:
            current_node = queue.popleft()

            if current_node not in visited:
                print(current_node, end=" ")
                visited.add(current_node)

                for neighbor in self.graph[current_node]:
                    if neighbor not in visited:
                        queue.append(neighbor)

# Test Case 1
g1 = Graph()
g1.add_edge('A', 'B')
g1.add_edge('A', 'C')
g1.add_edge('B', 'D')
g1.add_edge('B', 'E')
g1.add_edge('C', 'F')

print("BFS traversal starting from node 'A' (Test Case 1):")
g1.bfs('A')

# Test Case 2
g2 = Graph()
g2.add_edge('X', 'Y')
g2.add_edge('X', 'Z')
g2.add_edge('Y', 'W')
g2.add_edge('Z', 'V')
g2.add_edge('W', 'U')

print("\nBFS traversal starting from node 'X' (Test Case 2):")
g2.bfs('X')



class Graph:
    def __init__(self):
        self.graph = {}

    def add_edge(self, u, v):
        if u not in self.graph:
            self.graph[u] = []
        if v not in self.graph:
            self.graph[v] = []

        self.graph[u].append(v)
        self.graph[v].append(u)

    def dfs(self, start_node, visited=None):
        if visited is None:
            visited = set()

        if start_node not in visited:
            print(start_node, end=" ")
            visited.add(start_node)

            for neighbor in self.graph[start_node]:
                self.dfs(neighbor, visited)

# Test Case 1
g1 = Graph()
g1.add_edge('A', 'B')
g1.add_edge('A', 'C')
g1.add_edge('B', 'D')
g1.add_edge('B', 'E')
g1.add_edge('C', 'F')

print("DFS traversal starting from node 'A' (Test Case 1):")
g1.dfs('A')

# Test Case 2
g2 = Graph()
g2.add_edge('X', 'Y')
g2.add_edge('X', 'Z')
g2.add_edge('Y', 'W')
g2.add_edge('Z', 'V')
g2.add_edge('W', 'U')

print("\nDFS traversal starting from node 'X' (Test Case 2):")
g2.dfs('X')


class PrimMST:
    def __init__(self, vertices):
        self.V = vertices
        self.graph = [[0 for _ in range(vertices)] for _ in range(vertices)]

    def add_edge(self, u, v, weight):
        self.graph[u][v] = weight
        self.graph[v][u] = weight

    def prim_mst(self):
        INF = float('inf')
        selected = [False] * self.V

        # Initialize the key values with infinity
        key = [INF] * self.V
        key[0] = 0  # Start with the first vertex

        for _ in range(self.V):
            # Find the minimum key value vertex from the set of vertices not yet included in MST
            u = self.minimum_key(key, selected)
            selected[u] = True

            # Update key values of the adjacent vertices
            for v in range(self.V):
                if self.graph[u][v] > 0 and not selected[v] and key[v] > self.graph[u][v]:
                    key[v] = self.graph[u][v]

        self.print_mst(key)

    def minimum_key(self, key, selected):
        INF = float('inf')
        min_key = INF
        min_index = -1

        for v in range(self.V):
            if key[v] < min_key and not selected[v]:
                min_key = key[v]
                min_index = v

        return min_index

    def print_mst(self, key):
        print("Edge \tWeight")
        for i in range(1, self.V):
            print(f"{i} - {key[i]}")

# Test Case 1
g1 = PrimMST(5)
g1.add_edge(0, 1, 2)
g1.add_edge(0, 2, 3)
g1.add_edge(1, 2, 1)
g1.add_edge(1, 3, 1)
g1.add_edge(2, 4, 4)
print("Minimum Spanning Tree (Test Case 1):")
g1.prim_mst()

# Test Case 2
g2 = PrimMST(4)
g2.add_edge(0, 1, 1)
g2.add_edge(1, 2, 3)
g2.add_edge(2, 3, 4)
print("\nMinimum Spanning Tree (Test Case 2):")
g2.prim_mst()



class BellmanFord:
    def __init__(self, vertices):
        self.V = vertices
        self.graph = []

    def add_edge(self, u, v, weight):
        self.graph.append((u, v, weight))

    def bellman_ford(self, source):
        # Step 1: Initialize distances and predecessors
        distances = [float('inf')] * self.V
        predecessors = [None] * self.V
        distances[source] = 0

        # Step 2: Relax edges repeatedly
        for _ in range(self.V - 1):
            for u, v, weight in self.graph:
                if distances[u] + weight < distances[v]:
                    distances[v] = distances[u] + weight
                    predecessors[v] = u

        # Step 3: Check for negative cycles
        for u, v, weight in self.graph:
            if distances[u] + weight < distances[v]:
                print("Graph contains a negative cycle")
                return

        # Step 4: Print the result
        self.print_result(distances, predecessors)

    def print_result(self, distances, predecessors):
        print("Vertex\tDistance\tPredecessor")
        for i in range(self.V):
            print(f"{i}\t{distances[i]}\t\t{predecessors[i]}")

# Test Case 1
g1 = BellmanFord(5)
g1.add_edge(0, 1, 2)
g1.add_edge(0, 2, 3)
g1.add_edge(1, 2, 1)
g1.add_edge(1, 3, 1)
g1.add_edge(2, 4, 4)
print("Shortest Paths from Source 0 (Test Case 1):")
g1.bellman_ford(0)

# Test Case 2
g2 = BellmanFord(4)
g2.add_edge(0, 1, 1)
g2.add_edge(1, 2, 3)
g2.add_edge(2, 3, 4)
print("\nShortest Paths from Source 0 (Test Case 2):")
g2.bellman_ford(0)


class KruskalMST:
    def __init__(self, vertices):
        self.V = vertices
        self.graph = []

    def add_edge(self, u, v, weight):
        self.graph.append((u, v, weight))

    def kruskal_mst(self):
        self.graph.sort(key=lambda edge: edge[2])  # Sort edges in ascending order of weight
        parent = [i for i in range(self.V)]
        result = []

        def find_set(node):
            if parent[node] == node:
                return node
            parent[node] = find_set(parent[node])  # Path compression
            return parent[node]

        def union_sets(u, v):
            root_u = find_set(u)
            root_v = find_set(v)
            parent[root_u] = root_v

        for edge in self.graph:
            u, v, weight = edge
            if find_set(u) != find_set(v):
                result.append((u, v, weight))
                union_sets(u, v)

        self.print_mst(result)

    def print_mst(self, result):
        print("Edge \tWeight")
        for edge in result:
            print(f"{edge[0]} - {edge[1]}\t{edge[2]}")

# Test Case 1
g1 = KruskalMST(5)
g1.add_edge(0, 1, 2)
g1.add_edge(0, 2, 3)
g1.add_edge(1, 2, 1)
g1.add_edge(1, 3, 1)
g1.add_edge(2, 4, 4)
print("Minimum Spanning Tree (Test Case 1):")
g1.kruskal_mst()

# Test Case 2
g2 = KruskalMST(4)
g2.add_edge(0, 1, 1)
g2.add_edge(1, 2, 3)
g2.add_edge(2, 3, 4)
print("\nMinimum Spanning Tree (Test Case 2):")
g2.kruskal_mst()
