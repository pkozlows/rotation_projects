#include <iostream>
#include <armadillo>
#include <vector>
#include <unordered_map>
#include <tuple>
#include <cassert>
#include "scf.h"
#include "matrix_utils.h"

using namespace std;


Scf::Scf(const arma::mat& kinetic_integral, const arma::mat& coulombIntegral, const int& nelec, const int& npws, const vector<tuple<int, int, int>>& plane_waves) {
    kinetic = kinetic_integral;
    exchange = coulombIntegral;

    this->nelec = nelec;
    this->n_pw = npws;
    this->plane_waves = plane_waves;
}

// generate a rhf initial guess for the density matrix
arma::mat Scf::identity_guess() {
    // Initialize the density matrix to zeros
    arma::mat density_matrix(n_pw, n_pw, arma::fill::zeros);
    for (int i = 0; i < nelec / 2; ++i) {
        density_matrix(i, i) = 2.0;
    }

    return density_matrix;
        
}

arma::mat Scf::zeros_guess() {
    // Initialize the density matrix to zeros
    arma::mat density_matrix(n_pw, n_pw, arma::fill::zeros);


    return density_matrix;
        
}



arma::mat Scf::make_fock_matrix(arma::mat &density_matrix) {
    int npws = density_matrix.n_rows;
    arma::mat hcore = kinetic;

    arma::mat exchange_matrix(npws, npws, arma::fill::zeros);

    // Define a hash function for tuples
    struct TupleHash {
        template <typename T1, typename T2, typename T3>
        std::size_t operator()(const std::tuple<T1, T2, T3> &t) const {
            auto h1 = std::hash<T1>{}(std::get<0>(t));
            auto h2 = std::hash<T2>{}(std::get<1>(t));
            auto h3 = std::hash<T3>{}(std::get<2>(t));
            return h1 ^ h2 ^ h3;
        }
    };

    // Define an equality function for tuples
    struct TupleEqual {
        template <typename T1, typename T2, typename T3>
        bool operator()(const std::tuple<T1, T2, T3> &t1, const std::tuple<T1, T2, T3> &t2) const {
            return t1 == t2;
        }
    };

    // Create an unordered map for plane waves
    std::unordered_map<std::tuple<int, int, int>, int, TupleHash, TupleEqual> plane_wave_map;
    for (int i = 0; i < npws; ++i) {
        plane_wave_map[plane_waves[i]] = i;
    }

    // Calculate the Coulomb contribution
    for (int Q = 0; Q < npws; ++Q) {
        auto g_Q = plane_waves[Q];

        for (int i = 0; i < npws; ++i) {
            auto g_i = plane_waves[i];

            // Compute g_p = g_i + g_Q
            auto g_p = std::make_tuple(
                std::get<0>(g_i) + std::get<0>(g_Q),
                std::get<1>(g_i) + std::get<1>(g_Q),
                std::get<2>(g_i) + std::get<2>(g_Q)
            );

            // Check if g_p exists in the map
            auto it = plane_wave_map.find(g_p);
            if (it != plane_wave_map.end()) {
                int p_index = it->second;

                for (int q = 0; q < npws; ++q) {
                    auto g_q = plane_waves[q];

                    exchange_matrix(p_index, q) = density_matrix(p_index - Q, q - Q) * exchange(i, Q);
                }
            }
        }
    }

    // Since we are just considering the exchange contribution, we subtract out 0.5 * exchange_matrix
    arma::mat fock_matrix = hcore - 0.5 * exchange_matrix;
    return fock_matrix;
}
