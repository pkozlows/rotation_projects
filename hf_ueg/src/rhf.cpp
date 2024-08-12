#include "rhf.h"

using namespace std;


arma::mat RHF::guess_rhf(const string &guess_type) {
    arma::mat density_matrix(n_pw, n_pw, arma::fill::zeros);
    if (guess_type == "identity") {
        for (int i = 0; i < n_elec / 2; ++i) {
            density_matrix(i, i) = 2.0;
        }
    }
    return density_matrix;
}

arma::mat RHF::make_fock_matrix(arma::mat &guess_density) {

    arma::mat exchange_matrix(n_pw, n_pw, arma::fill::zeros);

    for (size_t p = 0; p < n_pw; ++p) {
        for (size_t q = 0; q < n_pw; ++q) {
            double sum = 0.0;
            //iterate over all possible momentum transfers
            for (size_t r = 0; r < n_mom; ++r) {
                //only append if we have valid indices
                if (lookup_table_minus(p, r) != -1 && lookup_table_minus(q, r) != -1) {
                    sum += exchange(r) * guess_density(lookup_table_minus(p, r), lookup_table_minus(q, r));
                }
            }
            exchange_matrix(p, q) = sum;
        }    
    }
    // cout << "The volume is: " << volume << endl;
    cout << "The fork matrix is " << endl;
    cout << kinetic - 0.5 * (exchange_matrix / volume) << endl;
    return kinetic - 0.5 * (exchange_matrix / volume);
}


double RHF::compute_energy(arma::mat &density_matrix, arma::mat &fock_matrix) {
    double energy = 0.0;
    for (int i = 0; i < n_pw; ++i) {
        for (int j = 0; j < n_pw; ++j) {
            energy += density_matrix(i, j) * (kinetic(i, j) + fock_matrix(i, j));
        }
    }
    return 0.5 * energy;
}

arma::mat RHF::generate_density_matrix(arma::mat &eigenvectors) {
    int n_occ = n_elec / 2;
    arma::mat occupied_eigenvectors = eigenvectors.cols(0, n_occ - 1);
    arma::mat density_matrix = 2 * (occupied_eigenvectors * occupied_eigenvectors.t());
    return density_matrix;
}