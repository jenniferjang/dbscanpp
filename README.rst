DBSCAN++
=====
Fast and robust density-based clustering algorithm.


Usage
======

**Initializiation**:

.. code-block:: python

  DBSCANPP(p, eps_density, eps_clustering, minPts)
  
p: The sample fraction, which determines the number of points to sample

eps_density: Radius for determining core points; points that have greater than minPts in its eps_density-radii ball will be considered core points

eps_clustering: Radius for determining neighbors and edges in the density graph

minPts: Number of neighbors required for a point to be labeled a core point. Works in conjunction with eps_density

**Finding Clusters**:

.. code-block:: python

  fit_predict(X, init="k-center", cluster_outliers=True)
  
X: Data matrix. Each row should represent a datapoint in Euclidean space

init: String. Either "k-center" for the k-center greedy sampling technique or "uniform" for a uniform random sampling technique

cluster_outliers: Boolean. Whether we should cluster the remaining points

fit_predict performs the clustering and returns the cluster labels.

**Example** (mixture of two gaussians):

.. code-block:: python

  from DBSCANPP import DBSCANPP
  import numpy as np

  # Mixture of three multivariate Gaussians
  cov = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

  X = np.concatenate([np.random.multivariate_normal([5, 5, 5], cov, 1000), 
            np.random.multivariate_normal([0, 0, 0], cov, 1000), 
            np.random.multivariate_normal([-5, -5, -5], cov, 1000)])
  y = np.concatenate([np.full(1000, 0), np.full(1000, 1), np.full(1000, 2)])

  # Declare a DBSCAN++ model with tuning hyperparameters
  dbscanpp = DBSCANPP(p=0.1, eps_density=5.0, eps_clustering=5.0, minPts=10)
  y_hat = dbscanpp.fit_predict(X, init="k-center")

  # Score the clustering
  from sklearn.metrics.cluster import adjusted_rand_score, adjusted_mutual_info_score
  print("Adj. Rand Index Score: %f." % adjusted_rand_score(y_hat, y))
  print("Adj. Mutual Info Score: %f." % adjusted_mutual_info_score(y_hat, y))


Install
=======

This package uses distutils, which is the default way of installing
python modules.

To install for all users on Unix/Linux::

  sudo python setup.py build; python setup.py install



Dependencies
=======

Python 2.7, NumPy, scikit-learn



