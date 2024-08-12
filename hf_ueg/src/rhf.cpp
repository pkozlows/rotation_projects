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

    for (int p = 0; p < n_pw; ++p) {
        for (int q = 0; q < n_pw; ++q) {
            double sum = 0.0;
            for (int Q = 0; Q < n_mom; ++Q) {
                if (lookup_table(p, Q) != -1 && lookup_table(q, Q) != -1) {
                    sum += guess_density(lookup_table(p, Q), lookup_table(q, Q)) * exchange(Q);
                    // cout << "The sum for Q= " << Q << "when p= " << p << " and q= " << q << " is " << sum << endl;
                }
            }
            exchange_matrix(p, q) = sum;
        }
    }
    return kinetic - 0.5 * exchange_matrix;
}

double RHF::compute_energy(arma::mat &density_matrix, arma::mat &fock_matrix) {
    double energy = 0.0;
    for (int i = 0; i < n_elec / 2; ++i) {
        energy += fock_matrix(i, i);
    }
    return energy;
}

arma::mat RHF::generate_density_matrix(arma::mat &eigenvectors) {
    int n_occ = n_elec / 2;
    arma::mat occupied_eigenvectors = eigenvectors.cols(0, n_occ - 1);
    arma::mat density_matrix = 2 * (occupied_eigenvectors * occupied_eigenvectors.t());
    return density_matrix;
}