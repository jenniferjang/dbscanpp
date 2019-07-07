#include <iterator>
#include <algorithm>

using namespace std;


double sq_euclidean_distance(int d, int i, int j, double * X) {
    /*
        Return the squared distance between points i and j in X

        Parameters
        ----------
        d: dimensions of dataset X
        i: index of first point in X
        j: index of second point in X
        X: (m, d) dataset
    */

    double distance = 0;
    for (int k = 0; k < d; k++) {
        distance += pow(X[i * d + k] - X[j * d + k], 2);
    }
    return distance;
}

void k_centers_cy(int m, int n, int d,
                  double * X,
                  double * closest_dist_sq,
                  int * result) {
    /*
        Return m points from X that are far away from each other

        Parameters
        ----------
        m: Number of points to sample
        n: Size of original dataset
        d: Dimensions of original dataset
        X: (m, d) dataset
        closest_dist_sq: (n, ) empty array to be filled with the minimum
                         distance to any points selected thus far
        result: (m, ) indices of the m selected points in X
    */

    // Pick the first point
    int center_id = 0;
    result[0] = center_id;

    // Initialize list of closest distances and calculate current potential
    for (int i = 0; i < n; i++) {
        closest_dist_sq[i] = sq_euclidean_distance(d, i, center_id, X);
    }

    // Pick the remaining m-1 points
    for (int c = 1; c < m; c++) {
        // Choose center that is farthest from already selected centers
        center_id = distance(closest_dist_sq, max_element(closest_dist_sq, closest_dist_sq + n));

        // Compute distances to center candidates
        for (int i = 0; i < n; i++) {
            closest_dist_sq[i] = min(sq_euclidean_distance(d, i, center_id, X), closest_dist_sq[i]);
        }

        // Permanently add best center candidate found in local tries
        result[c] = center_id;
    }
}