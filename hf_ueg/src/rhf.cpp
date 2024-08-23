#include "rhf.h"

using namespace std;


arma::mat RHF::guess_rhf(const std::string &guess_type) {
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

arma::mat RHF::compute_hartree_matrix(const arma::mat &guess_density) {
    arma::mat hartree(n_pw, n_pw, arma::fill::zeros);

    for (int p = 0; p < n_pw; ++p) {
        for (int q = 0; q < n_pw; ++q) {
            int index = lookup_tables.second(p, q);
            double hartree_sum = 0.0;

            for (int r = 0; r < n_pw; ++r) {
                int idx = lookup_tables.first(r, index);
                if (idx != -1) {
                    hartree_sum += guess_density(r, idx);
                }
            }

            hartree(p, q) = interaction(index) * hartree_sum;
        }
    }

    return hartree;
}

arma::mat RHF::compute_exchange_matrix(const arma::mat &guess_density) {
    arma::mat exchange_matrix(n_pw, n_pw, arma::fill::zeros);

    for (size_t p = 0; p < n_pw; ++p) {
        for (size_t q = 0; q < n_pw; ++q) {
            double exchange_sum = 0.0;

            for (size_t r = 0; r < n_mom; ++r) {
                if (lookup_tables.first(p, r) != -1 && lookup_tables.first(q, r) != -1) {
                    exchange_sum += interaction(r) * guess_density(lookup_tables.first(p, r), lookup_tables.first(q, r));
                }
            }

            exchange_matrix(p, q) = exchange_sum;
        }
    }

    return exchange_matrix;
}
arma::mat RHF::make_fock_matrix(arma::mat &guess_density) {
    arma::mat hartree = compute_hartree_matrix(guess_density);
    arma::mat exchange_matrix = compute_exchange_matrix(guess_density);

    // Optionally print the Hartree and exchange matrices for debugging
    
    // cout << "Kinetic matrix is: " << endl;
    // cout << kinetic << endl;
    // cout << "The normalized Hartree matrix is: " << endl;
    // cout << hartree / volume << endl;
    // cout << "The normalized Exchange matrix is: " << endl;
    // cout << exchange_matrix / volume << endl;


    return kinetic + (hartree - 0.5 * exchange_matrix) / volume;
}






double RHF::compute_energy(arma::mat &density_matrix, arma::mat &fock_matrix) {
    double energy = 0.0;
    for (size_t i = 0; i < n_pw; ++i) {
        for (size_t j = 0; j < n_pw; ++j) {
            energy += density_matrix(i, j) * (kinetic(i, j) + fock_matrix(i, j));
        }
    }
    return 0.5 * energy;
}

arma::mat RHF::generate_density_matrix(arma::mat &eigenvectors) {
    size_t n_occ = n_elec / 2;
    arma::mat occupied_eigenvectors = eigenvectors.cols(0, n_occ - 1);
    arma::mat density_matrix = 2 * (occupied_eigenvectors * occupied_eigenvectors.t());
    return density_matrix;
}