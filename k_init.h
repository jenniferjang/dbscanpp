#include <random>
#include <iterator>
#include <iostream>
#include <time.h>

using namespace std;


// Get current date/time, format is YYYY-MM-DD.HH:mm:ss
const std::string currentDateTime() {
    time_t     now = time(0);
    struct tm  tstruct;
    char       buf[80];
    tstruct = *localtime(&now);
    // Visit http://en.cppreference.com/w/cpp/chrono/c/strftime
    // for more information about date/time format
    strftime(buf, sizeof(buf), "%Y-%m-%d.%X", &tstruct);

    return buf;
}

double sq_euclidean_distance(int d, int i, int j, double * X) {
    double distance = 0;
    for (int k = 0; k < d; k++) {
        distance += pow(X[i * d + k] - X[j * d + k], 2);
    }
    return distance;
}

void k_init_cy2(int m, int n, int d,
               double * X,
               double * closest_dist_sq,
               int * result) {
    // Pick first center randomly
    int center_id = rand() % n;
    result[0] = center_id;

    // Initialize list of closest distances and calculate current potential
    double distance, current_pot = 0;
    for (int i = 0; i < n; i++) {
        distance = sq_euclidean_distance(d, i, center_id, X);
        closest_dist_sq[i] = distance;
        current_pot += distance;
    }

    double new_distance;
    int candidate_ind;
    default_random_engine generator;

    // Pick the remaining m-1 points
    for (int c = 1; c < m; c++) {
        // Choose center candidates by sampling with probability proportional
        // to the squared distance to the closest existing center
        discrete_distribution<> distribution(closest_dist_sq, closest_dist_sq + n);
        candidate_ind = distribution(generator);

        // Compute distances to center candidates
        current_pot = 0;
        for (int i = 0; i < n; i++) {
            new_distance = min(sq_euclidean_distance(d, i, candidate_ind, X), closest_dist_sq[i]);
            closest_dist_sq[i] = new_distance;
            current_pot += new_distance;
        }

        // Permanently add best center candidate found in local tries
        result[c] = candidate_ind;
    }
}

void k_centers_cy(int m, int n, int d,
                  double * X,
                  double * closest_dist_sq,
                  int * result) {
    // Pick first center randomly
    int center_id = rand() % n;
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