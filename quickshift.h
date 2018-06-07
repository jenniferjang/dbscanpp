#include <vector>
#include <algorithm>
#include <cfloat>
#include <map>
#include <cmath>
#include <assert.h> 
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
        kernel_size
        X_sampled: (m, ) array of indices of each sampled point
        distances: (m, m) distance matrix of distances between pairwise
                   sampled points
        densities: (m, ) array of kernel density estimates for each sampled 
                   point
        result: (n, ) array of cluster results to be calculated

    */

    assert(X_sampled.size() == m);
    assert(distances.size() == m*m);
    assert(densities.size() == m);
    assert(result.size() == m);

    double closest, dist;

    vector<int> old(m), parent(m);
    //vector<double> densities = _get_densities(m, n, X_sampled, kernel_size, distances);

    for (int i = 0; i < m; i++) {
        parent[i] = i;
        closest = DBL_MAX;
        for (int j = 0; j < m; j++) {
            dist = distances[i * m + j];
            if (dist <= 2.0*kernel_size && dist < closest && densities[j] > densities[i]) {
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
