#include "rhf.h"
#include <cassert>

using namespace std;


arma::mat RHF::guess(const std::string &guess_type) {
    arma::mat density_matrix;  // Declare density_matrix before the if-else ladder

    if (guess_type == "identity") {
        density_matrix = arma::mat(n_pw, n_pw, arma::fill::zeros);
        for (size_t i = 0; i < n_elec / 2; ++i) {
            density_matrix(i, i) = 2.0;
        }
    } else if (guess_type == "random") {
        density_matrix = arma::randu<arma::mat>(n_pw, n_pw);
        density_matrix += density_matrix.t();  // Ensure the matrix is symmetric
    } else if (guess_type == "zeros") {
        density_matrix = arma::mat(n_pw, n_pw, arma::fill::zeros);
    } else {
        // Handle unexpected guess_type values
        throw std::invalid_argument("Invalid guess_type provided.");
    }

    return density_matrix;
}

arma::mat RHF::compute_hartree_matrix(const arma::mat &density_matrix) {
    arma::mat hartree(n_pw, n_pw, arma::fill::zeros);

    for (int p = 0; p < n_pw; ++p) {
        for (int q = 0; q < n_pw; ++q) {
            int index = pw_lookup_table(p, q);
            double hartree_sum = 0.0;

            for (int r = 0; r < n_pw; ++r) {
                int idx = momentum_lookup_table(r, index);
                if (idx != -1) {
                    hartree_sum += density_matrix(r, idx);
                }
            }

            hartree(p, q) = interaction_integrals(index) * hartree_sum;
        }
    }

    return hartree;
}

arma::mat RHF::compute_exchange_matrix(const arma::mat &density_matrix) {
    arma::mat exchange_matrix(n_pw, n_pw, arma::fill::zeros);

    for (size_t p = 0; p < n_pw; ++p) {
        for (size_t q = 0; q < n_pw; ++q) {
            double exchange_sum = 0.0;

            for (size_t r = 0; r < n_mom; ++r) {
                if (momentum_lookup_table(p, r) != -1 && momentum_lookup_table(q, r) != -1) {
                    exchange_sum += interaction_integrals(r) * density_matrix(momentum_lookup_table(p, r), momentum_lookup_table(q, r));
                }
            }

            exchange_matrix(p, q) = exchange_sum;
        }
    }

    return exchange_matrix;
}
arma::mat RHF::make_fock_matrix(const arma::mat &density_matrix) {
    
    arma::mat hartree = compute_hartree_matrix(density_matrix);
    arma::mat exchange_matrix = compute_exchange_matrix(density_matrix);

    // Optionally print the Hartree and exchange matrices for debugging
    
    // cout << "Kinetic matrix is: " << endl;
    // cout << kinetic_integrals << endl;
    // cout << "The normalized Hartree matrix is: " << endl;
    // cout << hartree / volume << endl;
    // cout << "The normalized Exchange matrix is: " << endl;
    // cout << exchange_matrix / volume << endl;


    return kinetic_integrals + (hartree - 0.5 * exchange_matrix) / volume;
}






double RHF::compute_energy(const arma::mat &density_matrix, const arma::mat &fock_matrix) {
    double energy = 0.0;
    for (size_t i = 0; i < n_pw; ++i) {
        for (size_t j = 0; j < n_pw; ++j) {
            energy += density_matrix(i, j) * (kinetic_integrals(i, j) + fock_matrix(i, j));
        }
    }
    return 0.5 * energy;
}

arma::mat RHF::generate_density_matrix(const arma::mat &fock_matrix) {
    size_t n_occ = n_elec / 2;
    arma::vec eigval;
    arma::mat eigvec;
    arma::eig_sym(eigval, eigvec, fock_matrix);   
    arma::mat occupied_eigenvectors = eigvec.cols(0, n_occ - 1);
    arma::mat density_matrix = 2 * (occupied_eigenvectors * occupied_eigenvectors.t());
    return density_matrix;
}

double RHF::calculate_density_difference(const arma::mat &new_density, const arma::mat &previous_guess) {
    return arma::accu(arma::abs(new_density - previous_guess));
}

void RHF::print_density_matrix(const arma::mat &density_matrix) {
    cout << "The diagonal of the density matrix is: " << endl;
    cout << arma::diagvec(density_matrix).t() << endl;
}

void RHF::update_density_matrix(arma::mat &previous_guess, const arma::mat &new_density) {
    previous_guess += new_density;
    previous_guess /= 2;
}
