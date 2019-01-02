cimport sklearn.neighbors.dist_metrics

#cdef class YumiPoseDistanceMetric(DistanceMetric):

#    cdef inline DTYPE_t dist(self, DTYPE_t* x1, DTYPE_t* x2, ITYPE_t size) nogil except -1:
#        return euclidean_dist(x1, x2, size)

#    cdef inline DTYPE_t rdist(self, DTYPE_t* x1, DTYPE_t* x2, ITYPE_t size) nogil except -1:
#        return euclidean_rdist(x1, x2, size)

#    cdef inline DTYPE_t _rdist_to_dist(self, DTYPE_t rdist) nogil except -1:
#        return sqrt(rdist)

#    cdef inline DTYPE_t _dist_to_rdist(self, DTYPE_t dist) nogil except -1:
#        return dist * dist

#    def rdist_to_dist(self, rdist):
#        return np.sqrt(rdist)

#    def dist_to_rdist(self, dist):
#        return dist**2

cdef get_distance_fn():
    return sklearn.neighbors.DistanceMetric()