#include "rhf.h"

using namespace std;


arma::mat RHF::guess_rhf(const string &guess_type) {
    //for this guess of the density matrix I want the elements to be random numbers between 0 and 1
    arma::mat density_matrix = arma::randu<arma::mat>(n_pw, n_pw);
    //I want to make the density matrix symmetric so I add the transpose of the matrix to itself
    density_matrix += density_matrix.t();
    // arma::mat density_matrix(n_pw, n_pw, arma::fill::zeros);


    return density_matrix;
}

arma::mat RHF::make_fock_matrix(arma::mat &guess_density) {

    arma::mat hartree(n_pw, n_pw, arma::fill::zeros);
    arma::mat exchange_matrix(n_pw, n_pw, arma::fill::zeros);
    
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
                    hartree_sum += guess_density(r, idx);
                }
            }
            hartree(p, q) = interaction(index) * hartree_sum;

            if (index != static_cast<size_t>(-1)) {
            }
            
            double exchange_sum = 0.0;
            //iterate over all possible momentum transfers
            for (size_t r = 0; r < n_mom; ++r) {
                //only append if we have valid indices
                if (lookup_tables.first(p, r) != -1 && lookup_tables.first(q, r) != -1) {
                    exchange_sum += interaction(r) * guess_density(lookup_tables.first(p, r), lookup_tables.first(q, r));
                }
            }
            exchange_matrix(p, q) = exchange_sum;
        }    
    }
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