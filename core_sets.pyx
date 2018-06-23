import numpy as np
cimport numpy as np
import random
import math
from sklearn.neighbors import KDTree
from sklearn.neighbors.kde import KernelDensity
from sklearn.metrics.pairwise import pairwise_distances, euclidean_distances
from datetime import datetime

cdef extern from "k_init.h":
    void k_init_cy(int m, int n,
                   double * distances,
                   double * closest_dist_sq,
                   int * result)


cdef k_init_np(m, n, 
               np.ndarray[double,  ndim=2, mode="c"] distances,
               np.ndarray[double, ndim=1, mode="c"] closest_dist_sq,
               np.ndarray[np.int32_t, ndim=1, mode="c"] result):
    k_init_cy(m, n, 
              <double *> np.PyArray_DATA(distances),
              <double *> np.PyArray_DATA(closest_dist_sq),
              <int *> np.PyArray_DATA(result))

cdef extern from "DBSCAN.h":
    void DBSCAN_cy(int c, int n,
                   double eps_density,
                   double eps_clustering,
                   int * X_core,
                   int * neighbors,
                   int * neighbors_ind,
                   int * result)


cdef DBSCAN_np(c, n, eps_density, eps_clustering,
               np.ndarray[int,  ndim=1, mode="c"] X_core,
               np.ndarray[int, ndim=1, mode="c"] neighbors,
               np.ndarray[int, ndim=1, mode="c"] neighbors_ind,
               np.ndarray[np.int32_t, ndim=1, mode="c"] result):
    DBSCAN_cy(c, n, eps_density, eps_clustering,
              <int *> np.PyArray_DATA(X_core),
              <int *> np.PyArray_DATA(neighbors),
              <int *> np.PyArray_DATA(neighbors_ind),
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

    def k_init(self, X, m):
      n, d = X.shape

      indices = np.empty(m, dtype=np.int32)
      closest_dist_sq = np.empty(n, dtype=np.float64)
      distances = euclidean_distances(X, squared=True)

      k_init_np(m,
                n,
                distances,
                closest_dist_sq,
                indices)

      return indices

    def fit_predict(self, X, sample=True, init="k-means++", technique="exponential"):
        """
        """

        X = np.array(X)
        n, d = X.shape
        if sample:
          if technique == "exponential":
            m = int(self.p * math.pow(n, d/(d + 4.0)))
          elif technique == "linear":
            m = int(self.p * n)
          else:
            raise ValueError("Technique %s is not defined." % technique)
          if m < 1: 
            raise ValueError("p is too small, so sampling did not produce any points.")

          # Find a random subset of m points 
          if init == "k-means++":
            X_sampled = self.k_init(X, m)
          else:
            X_sampled = np.random.choice(np.arange(n, dtype=np.int32), m, replace=False)
          X_sampled_pts = X[X_sampled]
        else:
          X_sampled = np.arange(n, dtype=np.int32)
          X_sampled_pts = X

        # Find the core points
        radii = KDTree(X).query(X_sampled_pts, k=self.minPts)[0]
        X_core = X_sampled[radii[:,-1] <= self.eps_density]
        X_core_pts = X[X_core]

        # Get the list of core neighbors for each core point
        core_pts_tree = KDTree(X_core_pts)
        neighbors = core_pts_tree.query_radius(X_core_pts, self.eps_clustering)
        neighbors = np.asarray(np.concatenate(neighbors), dtype=np.int32)
        neighbors_ct = core_pts_tree.query_radius(X_core_pts, self.eps_clustering, count_only=True)
        neighbors_ind = np.cumsum(neighbors_ct, dtype=np.int32)
        
        # Cluster the core points
        c = X_core.shape[0]
        result = np.full(n, -1, dtype=np.int32)
        DBSCAN_np(c,
                  n,
                  self.eps_density, 
                  self.eps_clustering, 
                  X_core,
                  neighbors,
                  neighbors_ind,
                  result)

        # Find the closest core point to every data point
        closest_core_pts = core_pts_tree.query(X, k=1)[1]
        closest = X_core[closest_core_pts[:,0]] 

        # Cluster the remaining points
        cluster_remaining_np(n, closest, result)

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
