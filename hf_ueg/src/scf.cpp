#include <iostream>
#include <armadillo>
#include <vector>
#include <unordered_map>
#include <tuple>
#include <cassert>
#include "scf.h"
#include "matrix_utils.h"

using namespace std;


Scf::Scf(const arma::mat& kinetic_integral, const arma::vec& coulombIntegral, const int& nelec, const int& npws, const vector<tuple<int, int, int>>& plane_waves, const arma::mat& lookup_table) {
    kinetic = kinetic_integral;
    exchange = coulombIntegral;

    this->nelec = nelec;
    this->n_pw = npws;
    this->plane_waves = plane_waves;
    this->lookup_table = lookup_table;
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
    arma::mat hcore = kinetic;  // Ensure kinetic is properly initialized
    arma::mat lookup = lookup_table;

    arma::mat exchange_matrix(npws, npws, arma::fill::zeros);

    // Calculate the Coulomb contribution
    for (size_t p = 0; p < npws; ++p) {
        auto g_p = plane_waves[p];

        for (size_t q = 0; q < npws; ++q) {
            auto g_q = plane_waves[q];

            double sum = 0.0;
            for (size_t Q = 0; Q < npws; ++Q) {
                auto g_Q = plane_waves[Q];

                // Determine if this is an allowed index in the lookup table
                if (lookup(p, Q) != -1 && lookup(q, Q) != -1) {
                    size_t p_minus_q = lookup(p, Q);
                    size_t q_minus_q = lookup(q, Q);
                    double density_val = density_matrix(p_minus_q, q_minus_q);
                    double exchange_val = exchange(Q);


                    sum += density_val * exchange_val;
                }
            }
            exchange_matrix(p, q) = sum;
        }
    }

    return hcore - 0.5 * exchange_matrix;
}


// //based on the some of the eigenvalues, compute total RHF energy 
// double Scf::compute_rhf_energy_old(arma::mat &density_matrix, arma::mat &fock_matrix) {
//     double energy = 0.0;
//     size_t npws = density_matrix.n_rows;
//     for (size_t i = 0; i < npws; ++i) {
//         for (size_t j = 0; j < npws; ++j) {
//             energy += density_matrix(j, i) * (kinetic(i, j) + fock_matrix(i, j));
//         }
//     }
//     return 0.5 * energy;
// }

double Scf::compute_rhf_energy(arma::mat &density_matrix, arma::mat &fock_matrix) {
    double energy = 0.0;
    for (size_t i = 0; i < nelec / 2; ++i) {
        energy += fock_matrix(i, i) + kinetic(i, i);
    return energy;
    }
}

// Construct the density matrix
arma::mat Scf::generate_density_matrix(arma::mat &eigenvectors) {
    size_t n_occ = nelec / 2;
    arma::mat occupied_eigenvectors = eigenvectors.cols(0, n_occ - 1);
    arma::mat density_matrix = 2 * (occupied_eigenvectors * occupied_eigenvectors.t());
    return density_matrix;
}


