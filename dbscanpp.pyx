import numpy as np
cimport numpy as np
import random
import math
from sklearn.neighbors import KDTree
from datetime import datetime

cdef extern from "k_centers.h":
    void k_centers_cy(int m, int n, int d,
                      double * X,
                      double * closest_dist_sq,
                      int * result)


cdef k_centers_np(m, n, d,
                  np.ndarray[double,  ndim=2, mode="c"] X,
                  np.ndarray[double, ndim=1, mode="c"] closest_dist_sq,
                  np.ndarray[np.int32_t, ndim=1, mode="c"] result):
    k_centers_cy(m, n, d,
                 <double *> np.PyArray_DATA(X),
                 <double *> np.PyArray_DATA(closest_dist_sq),
                 <int *> np.PyArray_DATA(result))

cdef extern from "DBSCAN.h":
    void DBSCAN_cy(int c, int n,
                   int * X_core,
                   int * neighbors,
                   int * num_neighbors,
                   int * result)


cdef DBSCAN_np(c, n,
               np.ndarray[np.int32_t,  ndim=1, mode="c"] X_core,
               np.ndarray[np.int32_t, ndim=1, mode="c"] neighbors,
               np.ndarray[np.int32_t, ndim=1, mode="c"] num_neighbors,
               np.ndarray[np.int32_t, ndim=1, mode="c"] result):
    DBSCAN_cy(c, n,
              <int *> np.PyArray_DATA(X_core),
              <int *> np.PyArray_DATA(neighbors),
              <int *> np.PyArray_DATA(num_neighbors),
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


class DBSCANPP:
    """
    Parameters
    ----------
    
    p: The sample fraction, which determines m, the number of points to sample

    eps_density: Radius for determining core points; points that have greater than
                 minPts in its eps_density-radii ball will be considered core points

    eps_clustering: Radius for determining neighbors and edges in the density graph

    minPts: Number of neighbors required for a point to be labeled a core point. Works
            in conjunction with eps_density

    """

    def __init__(self, p, eps_density,
                 eps_clustering, minPts):
        self.p = p
        self.eps_density = eps_density
        self.eps_clustering = eps_clustering
        self.minPts = minPts

    def k_centers(self, m, n, d, X):
      """
      Return m points from X that are far away from each other

      Parameters
      ----------
      m: Number of points to sample
      n: Size of original dataset
      d: Dimensions of original dataset
      X: (m, d) dataset

      Returns
      ----------
      (m, ) list of indices
      """

      n, d = X.shape

      indices = np.empty(m, dtype=np.int32)
      closest_dist_sq = np.empty(n, dtype=np.float64)

      k_centers_np(m, n, d,
                   X,
                   closest_dist_sq,
                   indices)

      return indices

    def fit_predict(self, X, init="k-centers", cluster_outliers=True):
        """
        Determines the clusters in three steps.
        First step is to sample points from X using either the
        k-centers greedy sampling technique or a uniform
        sample technique. The next step is to run DBSCAN on the
        sampled points using the k-NN densities. Finally, all the 
        remaining points are clustered to the closest cluster.

        Parameters
        ----------
        X: Data matrix. Each row should represent a datapoint in 
           Euclidean space
        init: String. Either "k-centers" for the k-centers greedy
              sampling technique or "uniform" for a uniform random
              sampling technique
        cluster_outliers: Boolean. Whether we should cluster the 
              remaining points

        Returns
        ----------
        (n, ) cluster labels
        """

        X = np.ascontiguousarray(X)
        n, d = X.shape
        m = int(self.p * math.pow(n, d/(d + 4.0)))

        # Find a random subset of m points 
        if init == "uniform":
          X_sampled_ind = np.random.choice(np.arange(n, dtype=np.int32), m, replace=False)
          X_sampled_ind = np.sort(X_sampled_ind)
        elif init == "k-centers":
          X_sampled_ind = self.k_centers(m, n, d, X)
        else:
          raise ValueError("Initialization technique %s is not defined." % init)
        
        X_sampled_pts = X[X_sampled_ind]

        # Find the core points
        X_tree = KDTree(X)
        radii, indices = X_tree.query(X_sampled_pts, k=self.minPts)
        X_core_ind = X_sampled_ind[radii[:,-1] <= self.eps_density]
        X_core_pts = X[X_core_ind]

        # Get the list of core neighbors for each core point
        core_pts_tree = KDTree(X_core_pts)

        neighbors = core_pts_tree.query_radius(X_core_pts, self.eps_clustering)
        neighbors = np.asarray(np.concatenate(neighbors), dtype=np.int32)
        neighbors_ct = core_pts_tree.query_radius(X_core_pts, self.eps_clustering, count_only=True)
        num_neighbors = np.cumsum(neighbors_ct, dtype=np.int32)
        
        # Cluster the core points
        c = X_core_ind.shape[0]
        result = np.full(n, -1, dtype=np.int32)
        DBSCAN_np(c,
                  n,
                  X_core_ind,
                  neighbors,
                  num_neighbors,
                  result)

        # Find the closest core point to every data point
        dist_to_core_pt, closest_core_pt = core_pts_tree.query(X, k=1)
        closest = X_core_ind[closest_core_pt[:,0]]

        # Cluster the remaining points
        cluster_remaining_np(n, closest, result)
        
        # Set outliers
        if not cluster_outliers:
          result[dist_to_core_pt[:,0] > self.eps_density] = -1

        return result
