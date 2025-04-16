from libcpp.vector cimport vector
from libcpp.queue cimport queue
import numpy as np
cimport numpy as np
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
def bfs(int source, int target, int n_nodes,
        np.ndarray[np.int32_t, ndim=2] edge_index):

    cdef:
        int i, u, v, idx
        vector[vector[int]] adj = vector[vector[int]](n_nodes)
        vector[int] visited = vector[int](n_nodes, 0)
        vector[int] prev = vector[int](n_nodes, -1)
        queue[int] q

    # Build adjacency list (directed)
    for i in range(edge_index.shape[0]):
        u = edge_index[i, 0]
        v = edge_index[i, 1]
        adj[u].push_back(v)

    # BFS
    visited[source] = 1
    q.push(source)

    while not q.empty():
        u = q.front()
        q.pop()

        if u == target:
            break

        for i in range(adj[u].size()):
            v = adj[u][i]
            if not visited[v]:
                visited[v] = 1
                prev[v] = u
                q.push(v)

    # Reconstruct edge index path
    cdef vector[int] edge_path
    v = target
    while prev[v] != -1:
        u = prev[v]

        # Find index in edge_index where (u, v) occurs
        for i in range(edge_index.shape[0]):
            if edge_index[i, 0] == u and edge_index[i, 1] == v:
                edge_path.push_back(i)
                break
        v = u

    # Reverse to get source → target order
    cdef vector[int] reversed_path
    for i in range(edge_path.size() - 1, -1, -1):
        reversed_path.push_back(edge_path[i])

    return [i for i in reversed_path]
