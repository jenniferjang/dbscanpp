using namespace std;

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
