#include "uhf.h"
#include "matrix_utils.h"
#include <cassert>
#include <cmath> // For std::round

using namespace std;

pair<arma::mat, arma::mat> UHF::guess(const std::string &guess_type) {
    pair<arma::mat, arma::mat> density_matrices;
    if (guess_type == "random") {
        density_matrices.first = arma::randu<arma::mat>(n_pw, n_pw);
        density_matrices.first += density_matrices.first.t();
        density_matrices.second = arma::randu<arma::mat>(n_pw, n_pw);
        density_matrices.second += density_matrices.second.t();
    }
    return density_matrices;
}



pair<arma::mat, arma::mat> UHF::make_fock_matrix(const pair<arma::mat, arma::mat> &density_matrix) {
    arma::mat total_density = density_matrix.first + density_matrix.second;
    // cout << "The total density is: " << total_density << endl;

    arma::mat hartree(n_pw, n_pw, arma::fill::zeros);

    arma::mat exchange_alpha(n_pw, n_pw, arma::fill::zeros);
    arma::mat exchange_beta(n_pw, n_pw, arma::fill::zeros);

    //we can do the hartree and exchange terms in the same loops
    for (size_t p = 0; p < n_pw; ++p) {
        for (size_t q = 0; q < n_pw; ++q) {
            // start by calculating the hartree term

            //use the second local table to compute the index of the momentum transfer vector
            size_t index = pw_lookup_table(p, q);
            double hartree_sum = 0.0;
            for (size_t r = 0; r < n_pw; ++r) {
                // compute the index of j-Q
                int idx = momentum_lookup_table(r, index);
                if (idx != -1) {
                    hartree_sum += total_density(r, idx);
                }
            }
            hartree(p, q) = interaction_integrals(index) * hartree_sum;

            // now calculate the exchange term
            double exchange_sum_alpha = 0.0;
            double exchange_sum_beta = 0.0;
            for (size_t r = 0; r < n_mom; ++r) {
                //only append if we have valid indices
                if (momentum_lookup_table(p, r) != -1 && momentum_lookup_table(q, r) != -1) {
                    exchange_sum_alpha += interaction_integrals(r) * density_matrix.first(momentum_lookup_table(p, r), momentum_lookup_table(q, r));
                    exchange_sum_beta += interaction_integrals(r) * density_matrix.second(momentum_lookup_table(p, r), momentum_lookup_table(q, r));
                }
            }
            exchange_alpha(p, q) = exchange_sum_alpha;
            exchange_beta(p, q) = exchange_sum_beta;
        }    
    }

    arma::mat fock_alpha = kinetic_integrals + (hartree - exchange_alpha)/volume;
    arma::mat fock_beta = kinetic_integrals + (hartree - exchange_beta)/volume;

    return {fock_alpha, fock_beta};
}


double UHF::compute_energy(const pair<arma::mat, arma::mat> &density_matrix, const pair<arma::mat, arma::mat> &fock_matrices) {
    double energy = 0.0;
    arma::mat total_density = density_matrix.first + density_matrix.second;
    for (size_t i = 0; i < n_pw; ++i) {
        for (size_t j = 0; j < n_pw; ++j) {
            energy += (total_density(i, j) * (kinetic_integrals(i, j))) + (density_matrix.first(i, j) * fock_matrices.first(i, j)) + (density_matrix.second(i, j) * fock_matrices.second(i, j));
        }
    }
    return 0.5 * energy;
}

pair<arma::mat, arma::mat> UHF::generate_density_matrix(const pair<arma::mat, arma::mat> &fock_matrix) {
    int n_alpha = n_elec / 2 * (1 + spin_polarisation);
    int n_beta = n_elec - n_alpha;
    
    arma::vec eigval_alpha;
    arma::mat eigvec_alpha;
    arma::eig_sym(eigval_alpha, eigvec_alpha, fock_matrix.first);
    arma::mat occupied_eigenvectors_alpha = eigvec_alpha.cols(0, n_alpha - 1);
    arma::mat density_matrix_alpha = (occupied_eigenvectors_alpha * occupied_eigenvectors_alpha.t());

    arma::vec eigval_beta;
    arma::mat eigvec_beta;
    arma::eig_sym(eigval_beta, eigvec_beta, fock_matrix.second);
    arma::mat occupied_eigenvectors_beta = eigvec_beta.cols(0, n_beta - 1);
    arma::mat density_matrix_beta = (occupied_eigenvectors_beta * occupied_eigenvectors_beta.t());
    return {density_matrix_alpha, density_matrix_beta};
}

double UHF::calculate_density_difference(const pair<arma::mat, arma::mat> &new_density, const pair<arma::mat, arma::mat> &previous_guess) {
    return arma::accu(arma::abs(new_density.first - previous_guess.first)) + arma::accu(arma::abs(new_density.second - previous_guess.second));
}

void UHF::print_density_matrix(const pair<arma::mat, arma::mat> &density_matrix) {
    cout << "The diagonal of the alpha density matrix is: " << endl;
    cout << arma::diagvec(density_matrix.first).t() << endl;
    cout << "The diagonal of the beta density matrix is: " << endl;
    cout << arma::diagvec(density_matrix.second).t() << endl;

}

void UHF::update_density_matrix(pair<arma::mat, arma::mat> &previous_guess, const pair<arma::mat, arma::mat> &new_density) {
    previous_guess.first += new_density.first;
    previous_guess.first /= 2;

    previous_guess.second += new_density.second;
    previous_guess.second /= 2;
}