#include "uhf.h"

// UHF class constructor
UHF::UHF(const arma::mat &kinetic, const arma::vec &coulomb, const int &nelec, const int &npws, 
         const vector<tuple<int, int, int>> &plane_waves, const arma::mat &lookup_table, 
         const double &madeleung_constant)
    : kinetic(kinetic), exchange(coulomb), nelec(nelec), n_pw(npws), plane_waves(plane_waves), 
      lookup_table(lookup_table), madeleung_constant(madeleung_constant) {}


// UHF methods
pair<arma::mat, arma::mat> UHF::combination_guess() {
    //first provide the gas for the alpha spins
    arma::mat alpha_density_matrix(n_pw, n_pw, arma::fill::zeros);
    for (int i = 0; i < nelec / 2; ++i) {
        alpha_density_matrix(i, i) = 1.0;
    }
    //now provide the guess for the beta spins
    arma::mat beta_density_matrix(n_pw, n_pw, arma::fill::zeros);
    return make_pair(alpha_density_matrix, beta_density_matrix);
}

arma::mat UHF::generate_exchange_matrix(const arma::mat &density, const vector<tuple<int, int, int>> plane_waves, const arma::mat &lookup_table, int n_pw) {
    arma::mat exchange_matrix(n_pw, n_pw, arma::fill::zeros);
    for (int p = 0; p < n_pw; ++p) {
        for (int q = 0; q < n_pw; ++q) {
            double sum = 0.0;
            for (int Q = 0; Q < n_pw; ++Q) {
                if (lookup_table(p, Q) != -1 && lookup_table(q, Q) != -1) {
                    int p_minus_q = lookup_table(p, Q);
                    int q_minus_q = lookup_table(q, Q);
                    double density_val = density(p_minus_q, q_minus_q);
                    double exchange_val = exchange(Q);
                    sum += density_val * exchange_val;
                }
            }
            exchange_matrix(p, q) = sum;
        }
    }
    return exchange_matrix;
}


pair<arma::mat, arma::mat> UHF::make_fock_matrices(const pair<arma::mat, arma::mat> &guess_density) {
    arma::mat hcore = kinetic;

    // Use the helper function for both alpha and beta spins
    arma::mat alpha_exchange_matrix = generate_exchange_matrix(guess_density.first, plane_waves, lookup_table, n_pw);
    arma::mat beta_exchange_matrix = generate_exchange_matrix(guess_density.second, plane_waves, lookup_table, n_pw);

    return make_pair(hcore - 0.5 * alpha_exchange_matrix, hcore - 0.5 * beta_exchange_matrix);
}

double UHF::compute_uhf_energy(const pair<arma::mat, arma::mat> &density_matrix, const pair<arma::mat, arma::mat> &fock_matrices) {
    double energy = 0.0;
    for (int i = 0; i < n_pw; ++i) {
        for (int j = 0; j < n_pw; ++j) {
            energy += ((density_matrix.first(j, i) + density_matrix.second(j, i)) * kinetic(i, j) + 
                             (density_matrix.first(j, i) * fock_matrices.first(i, j) + density_matrix.second(j, i) * fock_matrices.second(i, j)));
        }
    }
    return 0.5 * energy;
}

pair<arma::mat, arma::mat> UHF::generate_density_matrices(const pair<arma::mat, arma::mat> &eigenvectors) {
    int n_occ = nelec / 2;
    arma::mat occupied_alpha_eigenvectors = eigenvectors.first.cols(0, n_occ - 1);
    arma::mat occupied_beta_eigenvectors = eigenvectors.second.cols(0, n_occ - 1);
    arma::mat alpha_density_matrix = 2 * (occupied_alpha_eigenvectors * occupied_alpha_eigenvectors.t());
    arma::mat beta_density_matrix = 2 * (occupied_beta_eigenvectors * occupied_beta_eigenvectors.t());
    return make_pair(alpha_density_matrix, beta_density_matrix);    
}
