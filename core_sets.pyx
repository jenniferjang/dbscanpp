import numpy as np
cimport numpy as np
import random
import math
from sklearn.neighbors import KDTree
from sklearn.neighbors.kde import KernelDensity
from sklearn.metrics.pairwise import pairwise_distances

cdef extern from "DBSCAN.h":
    void DBSCAN_cy(int c, int n,
                   double eps_density,
                   double eps_clustering,
                   int * X_core,
                   double * distances,
                   int * result)


cdef DBSCAN_np(c, n, eps_density, eps_clustering,
               np.ndarray[int,  ndim=1, mode="c"] X_core,
               np.ndarray[double, ndim=2, mode="c"] distances,
               np.ndarray[np.int32_t, ndim=1, mode="c"] result):
    DBSCAN_cy(c, n, eps_density, eps_clustering,
              <int *> np.PyArray_DATA(X_core),
              <double *> np.PyArray_DATA(distances),
              <int *> np.PyArray_DATA(result))


cdef extern from "quickshift.h":
    void quickshift_cy(int m, int n,
                       double kernel_size,
                       int * X_sampled,
                       double * distances,
                       double * densities,
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


cdef extern from "cluster_remaining.h":
    void cluster_remaining_cy(int n,
                              int * closest_point,
                              int * result)

cdef cluster_remaining_np(n, 
                          np.ndarray[int, ndim=1, mode="c"] closest_point,
                          np.ndarray[np.int32_t, ndim=1, mode="c"] result):
    cluster_remaining_cy(n,
                         <int *> np.PyArray_DATA(closest_point),
                         <int *> np.PyArray_DATA(result))


class CoreSetsDBSCAN:
    """
    

    """

    def __init__(self, p, eps_density,
                 eps_clustering, minPts):
        self.p = p
        self.eps_density = eps_density
        self.eps_clustering = eps_clustering
        self.minPts = minPts


    def fit_predict(self, X, sample=True):
        """
        """
        from datetime import datetime

        print datetime.now()
        X = np.array(X)
        n, d = X.shape
        if sample:
          m = int(self.p * math.pow(n, d/(d + 4.0)))
          if m < 1:
            raise ValueError("p is too small, so sampling did not produce any points.")
        else:
          m = int(self.p * n)

        # Find a random subset of m points 
        X_sampled = np.random.choice(np.arange(n, dtype=np.int32), m, replace=False)
        X_sampled.sort()
        X_sampled_pts = X[X_sampled]
        print datetime.now()

        # Find the core points and calculate distances between core points and rest of data set
        tree = KDTree(X)
        radii = tree.query(X_sampled_pts, k=self.minPts)[0]
        X_core = X_sampled[radii[:,-1] <= self.eps_density]
        c = X_core.shape[0]
        X_core_pts = X[X_core]
        distances = pairwise_distances(X_core_pts, X_core_pts)
        print datetime.now()
        
        # Cluster the core points
        result = np.full(n, -1, dtype=np.int32)
        DBSCAN_np(c, n, self.eps_density, self.eps_clustering, X_core, distances, result)
        print datetime.now()

        # Find the closest core point to every data point
        tree_core_pts = KDTree(X_core_pts)
        closest_point = tree_core_pts.query(X, k=1)[1]
        closest_point = X_core[closest_point[:,0]]
        print datetime.now()

        # Cluster the remaining points
        cluster_remaining_np(n, closest_point, result)
        print datetime.now()

        return result 


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
        m = int(self.p * math.pow(n, d/(d + 4.0)))

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
