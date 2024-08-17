#include "uhf.h"
#include "matrix_utils.h"
#include <cassert>

using namespace std;

pair<arma::mat, arma::mat> UHF::guess_uhf() {
    // Generate initial guesses
    arma::mat density_matrix_alpha(n_pw, n_pw, arma::fill::zeros);
    //fill the alpha density matrix with n_elec diag 1s
    for (size_t i = 0; i < n_elec; ++i) {
        density_matrix_alpha(i, i) = 1.0;
    }
    arma::mat density_matrix_beta(n_pw, n_pw, arma::fill::zeros);
    return {density_matrix_alpha, density_matrix_beta};
}

pair<arma::mat, arma::mat> UHF::make_uhf_fock_matrix(const pair<arma::mat, arma::mat> &guess_density) {
    arma::mat total_density = guess_density.first + guess_density.second;

    arma::mat hartree(n_pw, n_pw, arma::fill::zeros);

    arma::mat exchange_alpha(n_pw, n_pw, arma::fill::zeros);
    arma::mat exchange_beta(n_pw, n_pw, arma::fill::zeros);

    //we can do the hartree and exchange terms in the same loops
    for (size_t p = 0; p < n_pw; ++p) {
        for (size_t q = 0; q < n_pw; ++q) {
            // start by calculating the hartree term

            // calculate the momentum transfer vector
            arma::Col<int> p_m_q(3);
            p_m_q(0) = plane_waves(0, p) - plane_waves(0, q);
            p_m_q(1) = plane_waves(1, p) - plane_waves(1, q);
            p_m_q(2) = plane_waves(2, p) - plane_waves(2, q);

            // Find the index of p_m_q in momentum_transfer_vectors
            size_t index = static_cast<size_t>(-1); // Default to -1 (not found)
            for (size_t r = 0; r < n_mom; ++r) {
                if (arma::all(momentum_transfer_vectors.col(r) == p_m_q)) {
                    index = r;
                    break;
                }
            }

            if (index != static_cast<size_t>(-1)) {
                double hartree_sum = 0.0;
                for (size_t r = 0; r < n_pw; ++r) {
                    // compute the index of j-Q
                    int idx = lookup_table(r, index);
                    if (idx != -1) {
                        hartree_sum += total_density(r, idx);
                    }
                }
                hartree(p, q) = interaction(index) * hartree_sum;
            }

            // now calculate the exchange term
            double exchange_sum_alpha = 0.0;
            double exchange_sum_beta = 0.0;
            for (size_t r = 0; r < n_mom; ++r) {
                int p_idx = lookup_table(p, r);
                int q_idx = lookup_table(q, r);
                if (p_idx != -1 && q_idx != -1) {
                    exchange_sum_alpha += interaction(r) * guess_density.first(p_idx, q_idx);
                    exchange_sum_beta += interaction(r) * guess_density.second(p_idx, q_idx);
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
    size_t n_spin = n_elec / 2;
    arma::mat occupied_eigenvectors_alpha = eigenvectors.first.cols(0, n_spin - 1);
    arma::mat occupied_eigenvectors_beta = eigenvectors.second.cols(0, n_spin - 1);
    arma::mat density_matrix_alpha = (occupied_eigenvectors_alpha * occupied_eigenvectors_alpha.t());
    arma::mat density_matrix_beta = (occupied_eigenvectors_beta * occupied_eigenvectors_beta.t());
    return {density_matrix_alpha, density_matrix_beta};
}