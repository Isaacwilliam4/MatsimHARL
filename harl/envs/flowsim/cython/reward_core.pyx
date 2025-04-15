# distutils: language = c++
# cython: boundscheck=False, wraparound=False, cdivision=True, nonecheck=False

from libc.stdlib cimport rand, srand
from libcpp.vector cimport vector
import numpy as np
cimport numpy as np
from cython.parallel import prange
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
def sample_od_pairs(np.ndarray[np.float32_t, ndim=3] actions,
                    dict cluster_lists,
                    int n_agents):
    cdef:
        int cluster1, cluster2, hour, count, i
        int origin, dest
        dict od_result = {}
        tuple key
        list origins, dests

    for cluster1 in range(n_agents):
        for cluster2 in range(n_agents):
            if cluster1 == cluster2:
                continue
            for hour in range(24):
                count = int(10 ** actions[cluster1, cluster2, hour])
                origins = cluster_lists[cluster1]
                dests = cluster_lists[cluster2]

                for i in range(count):
                    origin = origins[np.random.randint(0, len(origins))]
                    dest = dests[np.random.randint(0, len(dests))]
                    key = (hour, origin, dest)
                    if key in od_result:
                        od_result[key] += 1
                    else:
                        od_result[key] = 1

    return od_result
