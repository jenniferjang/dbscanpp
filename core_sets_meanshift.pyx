import numpy as np
cimport numpy as np
import random
from sklearn.metrics.pairwise import pairwise_distances


cdef extern from "quickshift.h":
    void quickshift_cy(int m, int n,
                       double kernel_size,
                       int * X_sampled,
                       double * distances,
                       int * result)
    void cluster_remaining_cy(int m, int n,
                              int * X_sampled,
                              double * distances,
                              int * result)


cdef quickshift_np(m, n, kernel_size,
                   np.ndarray[int, ndim=1, mode="c"] X_sampled,
                   np.ndarray[double, ndim=2, mode="c"] distances,
                   np.ndarray[np.int32_t, ndim=1, mode="c"] result):
    quickshift_cy(m, n, kernel_size,
                  <int *> np.PyArray_DATA(X_sampled),
                  <double *> np.PyArray_DATA(distances),
                  <int *> np.PyArray_DATA(result))

cdef cluster_remaining_np(m, n, 
                          np.ndarray[int, ndim=1, mode="c"] X_sampled,
                          np.ndarray[double, ndim=2, mode="c"] distances,
                          np.ndarray[np.int32_t, ndim=1, mode="c"] result):
    cluster_remaining_cy(m, n,
                         <int *> np.PyArray_DATA(X_sampled),
                         <double *> np.PyArray_DATA(distances),
                         <int *> np.PyArray_DATA(result))


class CoreSetsMeanshift:
    """

    """


    def __init__(self, p, kernel_size):
        self.p = p
        self.kernel_size = kernel_size



    def fit_predict(self, X):
        """
        
        """
        X = np.array(X)
        n, d = X.shape
        m = int(self.p * n)

        X_sampled = np.random.choice(np.arange(n, dtype=np.int32), m, replace=False)
        X_sampled.sort()
        X_sampled_pts = X[X_sampled]
        
        distances = pairwise_distances(X_sampled_pts, X)

        result = np.full(n, -1, dtype=np.int32)
        quickshift_np(m, n, self.kernel_size, X_sampled, distances, result)
        cluster_remaining_np(m, n, X_sampled, distances, result)

        return result




