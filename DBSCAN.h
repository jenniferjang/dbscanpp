#include <vector>
#include <set>
#include <queue>
#include <cfloat>
#include <map>
#include <iostream>
#include <assert.h> 
using namespace std;


void DBSCAN_cy(int c, int n,
               double eps_density,
               double eps_clustering,
               int * X_core,
               double * distances,
               int * result) {
    /*
        Cluster the core points using density-based methods

        Parameters
        ----------
        c: Number of core points
        n: Size of original dataset
        eps_density: Minimum distance to be considered neighbors for
                     determining core points
        eps_clustering: Minimum distance to be considered a child for
                        breadth-first search 
        minPts: Minimum number of neighbors to be considered core point
        X_core: (c, ) array of the indices of the core points
        distances: (c, c) distance matrix of distances between pairwise
                   core points
        result: (n, ) array of cluster results to be calculated
    */

    // these literally don't work?
    assert(X_core.size() == c);
    assert(distances.size() == c * n);
    assert(result.size() == n);

    set<int> seen;
    queue<int> q = queue<int>();
    int point;
    double distance;

    for (int i = 0; i < c; i++) {
        q = queue<int>();
        if (seen.find(i) == seen.end()) {
            q.push(i);
        }

        while (!q.empty()) {
            point = q.front();
            q.pop();
            for (int j = 0; j < c; j++) {
                distance = distances[point * c + j];
                if (distance <= eps_clustering && seen.find(j) == seen.end()) {
                    q.push(j);
                    result[X_core[j]] = i;
                }
            }
            seen.insert(point);
        }
    }

    map<int,int> clusters_index;
    int k, cnt = 0;

    for (int i = 0; i < c; i++) {
        k = X_core[i];
        if (result[k] >= 0) {
            if (clusters_index.find(result[k]) == clusters_index.end()) {
                clusters_index[result[k]] = cnt;
                cnt++;
            } 
            result[k] = clusters_index.find(result[k]) -> second;
        }
    }
}
