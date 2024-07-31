#include <iostream>
#include <armadillo>
#include <vector>
#include <unordered_map>
#include <tuple>
#include <cassert>
#include "scf.h"
#include "matrix_utils.h"

using namespace std;


Scf::Scf(const arma::mat& kinetic_integral, const arma::vec& coulombIntegral, const int& nelec, const int& npws, const vector<tuple<int, int, int>>& plane_waves, const arma::mat& lookup_table, const double& madeleung_constant) {
    kinetic = kinetic_integral;
    exchange = coulombIntegral;

    this->nelec = nelec;
    this->n_pw = npws;
    this->plane_waves = plane_waves;
    this->lookup_table = lookup_table;
    this->madeleung_constant = madeleung_constant;
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

    arma::mat exchange_matrix(npws, npws, arma::fill::zeros);

    // Calculate the Coulomb contribution
    for (int p = 0; p < npws; ++p) {
        auto g_p = plane_waves[p];

        for (int q = 0; q < npws; ++q) {
            auto g_q = plane_waves[q];

            double sum = 0.0;
            for (int Q = 0; Q < npws; ++Q) {
                auto g_Q = plane_waves[Q];

                // Determine if this is an allowed index in the lookup_table table
                if (lookup_table(p, Q) != -1 && lookup_table(q, Q) != -1) {
                    int p_minus_q = lookup_table(p, Q);
                    int q_minus_q = lookup_table(q, Q);
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


double Scf::compute_rhf_energy(arma::mat &density_matrix, arma::mat &fock_matrix) {
    double energy = 0.0;
    for (int i = 0; i < nelec / 2; ++i) {
        energy += fock_matrix(i, i) + kinetic(i, i);
    }
    // for (int i = 0; i < n_pw; ++i) {
    //     for (int j = 0; j < n_pw; ++j) {
    //         energy += density_matrix(j, i) * (kinetic(i, j) + fock_matrix(i, j));
    //     }
    // }
    return energy;// + madeleung_constant;
}

// Construct the density matrix
arma::mat Scf::generate_density_matrix(arma::mat &eigenvectors) {
    int n_occ = nelec / 2;
    arma::mat occupied_eigenvectors = eigenvectors.cols(0, n_occ - 1);
    arma::mat density_matrix = 2 * (occupied_eigenvectors * occupied_eigenvectors.t());
    return density_matrix;
}


