#include "uhf.h"

using namespace std;

pair<arma::mat, arma::mat> UHF::guess_uhf() {
    //do a combination guess
    arma::mat alpha_density(n_pw, n_pw, arma::fill::zeros);
    arma::mat beta_density(n_pw, n_pw, arma::fill::zeros);
    for (int i = 0; i < n_elec / 2; ++i) {
        alpha_density(i, i) = 1.0;
    }
    return {alpha_density, beta_density};
}

pair<arma::mat, arma::mat> UHF::make_uhf_fock_matrix(const pair<arma::mat, arma::mat> &guess_density) {
    arma::mat exchange_alpha(n_pw, n_pw, arma::fill::zeros);
    arma::mat exchange_beta(n_pw, n_pw, arma::fill::zeros);

    for (size_t p = 0; p < n_pw; ++p) {
        for (size_t q = 0; q < n_pw; ++q) {
            double sum_alpha = 0.0;
            double sum_beta = 0.0;
            //iterate over all possible momentum transfers
            for (size_t r = 0; r < n_mom; ++r) {
                //only append if we have valid indices
                if (lookup_table(p, r) != -1 && lookup_table(q, r) != -1) {
                    sum_alpha += exchange(r) * guess_density.first(lookup_table(p, r), lookup_table(q, r));
                    sum_beta += exchange(r) * guess_density.second(lookup_table(p, r), lookup_table(q, r));
                }
            }
            exchange_alpha(p, q) = sum_alpha;
            exchange_beta(p, q) = sum_beta;
        }    
    }
    arma::mat fock_alpha = kinetic - exchange_alpha / volume;
    arma::mat fock_beta = kinetic - exchange_beta / volume;
    return {fock_alpha, fock_beta};
}

double UHF::compute_uhf_energy(const pair<arma::mat, arma::mat> &density_matrix, const pair<arma::mat, arma::mat> &fock_matrices) {
    double energy = 0.0;
    for (int i = 0; i < n_pw; ++i) {
        for (int j = 0; j < n_pw; ++j) {
            energy += density_matrix.first(i, j) * (kinetic(i, j) + fock_matrices.first(i, j));
            energy += density_matrix.second(i, j) * (kinetic(i, j) + fock_matrices.second(i, j));
        }
    }
    return 0.5 * energy;
}

pair<arma::mat, arma::mat> UHF::generate_uhf_density_matrix(const pair<arma::mat, arma::mat> &eigenvectors) {
    int n_occ = n_elec / 2;
    arma::mat occupied_eigenvectors_alpha = eigenvectors.first.cols(0, n_occ - 1);
    arma::mat occupied_eigenvectors_beta = eigenvectors.second.cols(0, n_occ - 1);
    arma::mat density_matrix_alpha = (occupied_eigenvectors_alpha * occupied_eigenvectors_alpha.t());
    arma::mat density_matrix_beta = (occupied_eigenvectors_beta * occupied_eigenvectors_beta.t());
    return {density_matrix_alpha, density_matrix_beta};
}