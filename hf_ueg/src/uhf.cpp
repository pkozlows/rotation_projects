#include "uhf.h"
#include "matrix_utils.h"
#include <cassert>
#include <cmath> // For std::round

using namespace std;

pair<arma::mat, arma::mat> UHF::guess_uhf() {
    // Determine the number of alpha and beta electrons based on polarization
    size_t n_alpha = std::round(n_elec / 2.0 * (1 + spin_polarisation));
    size_t n_beta = n_elec - n_alpha;

    // Create density matrices
    arma::mat density_matrix_alpha(n_pw, n_pw, arma::fill::zeros);
    arma::mat density_matrix_beta(n_pw, n_pw, arma::fill::zeros);

    // Fill the alpha density matrix with 1's on the diagonal up to n_alpha
    for (size_t i = 0; i < n_alpha; ++i) {
        density_matrix_alpha(i, i) = 1.0;
    }
    // cout << "The alpha density matrix is: " << density_matrix_alpha << endl;

    // Fill the beta density matrix with 1's on the diagonal up to n_beta
    for (size_t i = 0; i < n_beta; ++i) {
        density_matrix_beta(i, i) = 1.0;
    }
    //I want to add a random perturbation so I need to make a matrix of the same size as my guesses for alpha and beta but I want its entries to be random numbers between 0 and 1
    // arma::mat perturbation_alpha = arma::randu<arma::mat>(n_pw, n_pw);
    // arma::mat perturbation_beta = arma::randu<arma::mat>(n_pw, n_pw);
    // density_matrix_alpha += perturbation_alpha + perturbation_alpha.t();
    // density_matrix_beta += perturbation_beta + perturbation_beta.t();
    return {density_matrix_alpha, density_matrix_beta};
}



pair<arma::mat, arma::mat> UHF::make_uhf_fock_matrix(const pair<arma::mat, arma::mat> &guess_density) {
    arma::mat total_density = guess_density.first + guess_density.second;
    // cout << "The total density is: " << total_density << endl;

    arma::mat hartree(n_pw, n_pw, arma::fill::zeros);

    arma::mat exchange_alpha(n_pw, n_pw, arma::fill::zeros);
    arma::mat exchange_beta(n_pw, n_pw, arma::fill::zeros);

    //we can do the hartree and exchange terms in the same loops
    for (size_t p = 0; p < n_pw; ++p) {
        for (size_t q = 0; q < n_pw; ++q) {
            // start by calculating the hartree term

            //use the second local table to compute the index of the momentum transfer vector
            size_t index = lookup_tables.second(p, q);
            double hartree_sum = 0.0;
            for (size_t r = 0; r < n_pw; ++r) {
                // compute the index of j-Q
                int idx = lookup_tables.first(r, index);
                if (idx != -1) {
                    hartree_sum += total_density(r, idx);
                }
            }
            hartree(p, q) = interaction(index) * hartree_sum;

            // now calculate the exchange term
            double exchange_sum_alpha = 0.0;
            double exchange_sum_beta = 0.0;
            for (size_t r = 0; r < n_mom; ++r) {
                //only append if we have valid indices
                if (lookup_tables.first(p, r) != -1 && lookup_tables.first(q, r) != -1) {
                    exchange_sum_alpha += interaction(r) * guess_density.first(lookup_tables.first(p, r), lookup_tables.first(q, r));
                    exchange_sum_beta += interaction(r) * guess_density.second(lookup_tables.first(p, r), lookup_tables.first(q, r));
                }
            }
            exchange_alpha(p, q) = exchange_sum_alpha;
            exchange_beta(p, q) = exchange_sum_beta;
        }    
    }

    arma::mat fock_alpha = kinetic + (hartree - exchange_alpha)/volume;
    arma::mat fock_beta = kinetic + (hartree - exchange_beta)/volume;

    return {fock_alpha, fock_beta};
}


double UHF::compute_uhf_energy(const pair<arma::mat, arma::mat> &density_matrix, const pair<arma::mat, arma::mat> &fock_matrices) {
    double energy = 0.0;
    arma::mat total_density = density_matrix.first + density_matrix.second;
    for (size_t i = 0; i < n_pw; ++i) {
        for (size_t j = 0; j < n_pw; ++j) {
            energy += (total_density(i, j) * (kinetic(i, j))) + (density_matrix.first(i, j) * fock_matrices.first(i, j)) + (density_matrix.second(i, j) * fock_matrices.second(i, j));
        }
    }
    return 0.5 * energy;
}

pair<arma::mat, arma::mat> UHF::generate_uhf_density_matrix(const pair<arma::mat, arma::mat> &eigenvectors) {
    int n_alpha = n_elec / 2 * (1 + spin_polarisation);
    int n_beta = n_elec - n_alpha;
    arma::mat occupied_eigenvectors_alpha = eigenvectors.first.cols(0, n_alpha - 1);
    arma::mat occupied_eigenvectors_beta = eigenvectors.second.cols(0, n_beta - 1);
    // arma::mat occupied_eigenvectors_alpha = eigenvectors.first.cols(0, n_alpha - 1);
    // arma::mat occupied_eigenvectors_beta(n_pw, n_pw, arma::fill::zeros);
    // if (spin_polarisation != 1) {
    //     arma::mat occupied_eigenvectors_beta = eigenvectors.second.cols(0, n_beta - 1);
    // }
    // cout << "The occupied eigenvectors for alpha are: " << occupied_eigenvectors_alpha << endl;
    // cout << "The occupied eigenvectors for beta are: " << occupied_eigenvectors_beta << endl;

    arma::mat density_matrix_alpha = (occupied_eigenvectors_alpha * occupied_eigenvectors_alpha.t());
    arma::mat density_matrix_beta = (occupied_eigenvectors_beta * occupied_eigenvectors_beta.t());
    return {density_matrix_alpha, density_matrix_beta};
}