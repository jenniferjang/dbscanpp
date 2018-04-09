#include <vector>
#include <algo.h>
#include <cfloat>
#include <map>
#include <cmath>
using namespace std;

vector<double> _get_densities(int m, int n,
                              int * X_sampled,
                              double kernel_size,
                              double * distances) {
    /*
        Calculates the kernel density of each point in the sample with
        respect to every point in the dataset

        Returns
        ----------
        1-dimensional array of size m where each element is the kernel density 
        of the corresponding sampled point
    */

    vector<double> densities(m, 0.0);
    double inv_kernel_size_sqr = -0.5 / pow(kernel_size, 2);

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++ ) {
            if (distances[i * n + j] <= kernel_size) {
                densities[i] += exp(distances[i * n + j] * inv_kernel_size_sqr);
            }
        }
    }

    for (int i = 0; i < m; i++) {
        cout << densities[i] << ", ";
    }
    cout << endl;
    return densities;
}

void quickshift_cy(int m, int n,
                   double kernel_size,
                   int * X_sampled,
                   double * distances,
                   double * densities,
                   int * result) {
    /*
        Runs quickshift on subset of the original dataset using
        densities with respect to the whole dataset.

        Parameters
        ----------
        m: Number of sampled points
        n: Size of original dataset
        kernel_size:
        X_sampled: 
        distances: m x m distance matrix of distances between pairwise
                   sampled points
        result: Cluster result

    */

    double closest, dist;

    vector<int> old(m), parent(m);
    //vector<double> densities = _get_densities(m, n, X_sampled, kernel_size, distances);

    for (int i = 0; i < m; i++) {
        parent[i] = i;
        closest = DBL_MAX;
        for (int j = 0; j < m; j++) {
            dist = distances[i * m + j];
            if (dist <= kernel_size && dist < closest && densities[j] > densities[i]) {
                closest = distances[i * m + j];
                parent[i] = j;
            }
        }
    }

    while (old != parent) {
        old = parent;
        for (int i = 0; i < m; i++) {
            parent[i] = old[old[i]];
        }
    }

    map<int,int> clusters_index;
    int k, cnt = 0;

    for (int i = 0; i < m; i++) {
        k = parent[i];
        if (clusters_index.find(k) == clusters_index.end()) {
            clusters_index[k] = cnt;
            cnt++;
        } 
        result[X_sampled[i]] = clusters_index.find(k) -> second;
    }

}

void cluster_remaining_cy(int n,
                          int * closest_point,
                          int * result) {
    /*
        Label the remaining points by clustering them with the nearest
        clustered neighbor.

        Parameters
        ----------
        n: Size of original dataset
        closest_point: (n, ) array with the index of the closest core point
        result: (n, ) array of cluster results to be calculated

    */

    for (int i = 0; i < n; i++) {
        if (result[i] < 0) {
            result[i] = result[closest_point[i]];
        }
    }
}
