#include "uhf.h"
#include "matrix_utils.h"

using namespace std;

pair<arma::mat, arma::mat> UHF::guess_uhf() {
    //do guess of random positive numbers between 0 and 1 for alpha and beta
    arma::mat density_matrix_alpha(n_pw, n_pw, arma::fill::randu);
    arma::mat density_matrix_beta(n_pw, n_pw, arma::fill::randu);
    return {density_matrix_alpha, density_matrix_beta};
}

pair<arma::mat, arma::mat> UHF::make_uhf_fock_matrix(const pair<arma::mat, arma::mat> &guess_density) {

    arma::mat total_density = guess_density.first + guess_density.second;

    arma::mat hartree(n_pw, n_pw, arma::fill::zeros);

    arma::mat exchange_alpha(n_pw, n_pw, arma::fill::zeros);
    arma::mat exchange_beta(n_pw, n_pw, arma::fill::zeros);

    for (size_t p = 0; p < n_pw; ++p) {
        for (size_t q = 0; q < n_pw; ++q) {
            //compute the difference put tween the plane wave indexed by p and q
            auto [px, py, pz] = plane_waves[p];
            auto [qx, qy, qz] = plane_waves[q];
            tuple<int, int, int> p_m_q = make_tuple(px - qx, py - qy, pz - qz);
            // search for index of p_m_q in momentum transfer vectors
            int index = distance(momentum_transfer_vectors.begin(), find(momentum_transfer_vectors.begin(), momentum_transfer_vectors.end(), p_m_q));

            double hartree_sum = 0.0;
            for (size_t r = 0; r < n_pw; ++r) {
                //only append if we have valid index
                if (lookup_table(r, index) != -1) {
                    hartree_sum += total_density(r, lookup_table(r, index));
                }
            }
            hartree(p, q) = hartree_sum*exchange(index);
            
            double exchange_sum_alpha = 0.0;
            double exchange_sum_beta = 0.0;
            //iterate over all possible momentum transfers
            for (size_t r = 0; r < n_mom; ++r) {
                //only append if we have valid indices
                if (lookup_table(p, r) != -1 && lookup_table(q, r) != -1) {
                    exchange_sum_alpha += exchange(r) * guess_density.first(lookup_table(p, r), lookup_table(q, r));
                    exchange_sum_beta += exchange(r) * guess_density.second(lookup_table(p, r), lookup_table(q, r));
                }
            }
            exchange_alpha(p, q) = exchange_sum_alpha;
            exchange_beta(p, q) = exchange_sum_beta;
        


        }    
    }

    print_matrix(hartree);

    arma::mat fock_alpha = kinetic + (hartree - exchange_alpha) / volume;
    arma::mat fock_beta = kinetic + (hartree - exchange_beta) / volume;
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