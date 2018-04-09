import numpy as np
cimport numpy as np
import random
from sklearn.neighbors import KDTree
from sklearn.neighbors.kde import KernelDensity
from sklearn.metrics.pairwise import pairwise_distances


cdef extern from "quickshift.h":
    void quickshift_cy(int m, int n,
                       double kernel_size,
                       int * X_sampled,
                       double * distances,
                       double * densities,
                       int * result)
    void cluster_remaining_cy(int n,
                              int * closest_point,
                              int * result)


cdef quickshift_np(m, n, kernel_size,
                   np.ndarray[int, ndim=1, mode="c"] X_sampled,
                   np.ndarray[double, ndim=2, mode="c"] distances,
                   np.ndarray[double, ndim=1, mode="c"] densities,
                   np.ndarray[np.int32_t, ndim=1, mode="c"] result):
    quickshift_cy(m, n, kernel_size,
                  <int *> np.PyArray_DATA(X_sampled),
                  <double *> np.PyArray_DATA(distances),
                  <double *> np.PyArray_DATA(densities),
                  <int *> np.PyArray_DATA(result))


cdef cluster_remaining_np(n, 
                          np.ndarray[int, ndim=1, mode="c"] closest_point,
                          np.ndarray[np.int32_t, ndim=1, mode="c"] result):
    cluster_remaining_cy(n,
                         <int *> np.PyArray_DATA(closest_point),
                         <int *> np.PyArray_DATA(result))


class CoreSetsMeanshift:
    """

    """

    def __init__(self, p, kernel_size, kernel="gaussian"):
        self.p = p
        self.kernel_size = kernel_size
        self.kernel = kernel


    def fit_predict(self, X):
        """
        
        """
        X = np.array(X)
        n, d = X.shape
        m = int(self.p * n)

        if m < 1:
          raise ValueError("p is too small, so sampling did not produce any points.")

        # Find a random subset of m points 
        X_sampled = np.random.choice(np.arange(n, dtype=np.int32), m, replace=False)
        X_sampled.sort()
        X_sampled_pts = X[X_sampled]
        distances = pairwise_distances(X_sampled_pts, X_sampled_pts)
        
        kde = KernelDensity(kernel=self.kernel, bandwidth=self.kernel_size).fit(X)
        densities = kde.score_samples(X_sampled_pts)

        result = np.full(n, -1, dtype=np.int32)
        quickshift_np(m, n, self.kernel_size, X_sampled, distances, densities, result)

        # Find the closest core point to every data point
        tree = KDTree(X_sampled_pts)
        closest_point = tree.query(X, k=1)[1]
        closest_point = X_sampled[closest_point[:,0]]

        # Cluster the remaining points
        cluster_remaining_np(n, closest_point, result)

        return result




