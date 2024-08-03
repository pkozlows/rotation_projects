#include "scf.h"

using namespace std;

Scf::Scf(const arma::mat &kinetic, const arma::vec &exchange, const int &nelec, const int &npws, 
         const vector<tuple<int, int, int>> &plane_waves, const arma::mat &lookup_table, 
         const double &madeleung_constant)
    : kinetic(kinetic), exchange(exchange), nelec(nelec), n_pw(npws), plane_waves(plane_waves), 
      lookup_table(lookup_table), madeleung_constant(madeleung_constant) {}

// RHF class constructor
RHF::RHF(const arma::mat &kinetic, const arma::vec &exchange, const int &nelec, const int &npws, 
         const vector<tuple<int, int, int>> &plane_waves, const arma::mat &lookup_table, 
         const double &madeleung_constant)
    : Scf(kinetic, exchange, nelec, npws, plane_waves, lookup_table, madeleung_constant) {}

arma::mat RHF::guess_rhf(const string &guess_type) {
    arma::mat density_matrix(n_pw, n_pw, arma::fill::zeros);
    if (guess_type == "identity") {
        for (int i = 0; i < nelec / 2; ++i) {
            density_matrix(i, i) = 2.0;
        }
    }
    return density_matrix;
}

arma::mat RHF::make_fock_matrix(arma::mat &guess_density) {
    int npws = guess_density.n_rows;
    arma::mat hcore = kinetic;

    arma::mat exchange_matrix(npws, npws, arma::fill::zeros);

    for (int p = 0; p < npws; ++p) {
        for (int q = 0; q < npws; ++q) {
            double sum = 0.0;
            for (int Q = 0; Q < npws; ++Q) {
                if (lookup_table(p, Q) != -1 && lookup_table(q, Q) != -1) {
                    int p_minus_q = lookup_table(p, Q);
                    int q_minus_q = lookup_table(q, Q);
                    double density_val = guess_density(p_minus_q, q_minus_q);
                    double exchange_val = exchange(Q);

                    sum += density_val * exchange_val;
                }
            }
            exchange_matrix(p, q) = sum;
        }
    }

    return hcore - 0.5 * exchange_matrix;
}

double RHF::compute_energy(arma::mat &density_matrix, arma::mat &fock_matrix) {
    double energy = 0.0;
    for (int i = 0; i < nelec / 2; ++i) {
        energy += fock_matrix(i, i) + kinetic(i, i);
    }
    return energy;
}

arma::mat RHF::generate_density_matrix(arma::mat &eigenvectors) {
    int n_occ = nelec / 2;
    arma::mat occupied_eigenvectors = eigenvectors.cols(0, n_occ - 1);
    arma::mat density_matrix = 2 * (occupied_eigenvectors * occupied_eigenvectors.t());
    return density_matrix;
}

// UHF class constructor
UHF::UHF(const arma::mat &kinetic, const arma::vec &exchange, const int &nelec, const int &npws, 
         const vector<tuple<int, int, int>> &plane_waves, const arma::mat &lookup_table, 
         const double &madeleung_constant)
    : Scf(kinetic, exchange, nelec, npws, plane_waves, lookup_table, madeleung_constant) {}

pair<arma::mat, arma::mat> UHF::guess_uhf() {
    //first provide the guess for the alpha spins
    arma::mat alpha_density_matrix(n_pw, n_pw, arma::fill::zeros);
    for (int i = 0; i < nelec / 2; ++i) {
        alpha_density_matrix(i, i) = 1.0;
    }
    //now provide the guess for the beta spins
    arma::mat beta_density_matrix(n_pw, n_pw, arma::fill::zeros);
    return make_pair(alpha_density_matrix, beta_density_matrix);
}

pair<arma::mat, arma::mat> UHF::make_uhf_fock_matrix(pair<arma::mat, arma::mat> &guess_density) {
    arma::mat hcore = kinetic;

    // Use the helper function for both alpha and beta spins
    arma::mat alpha_exchange_matrix = generate_exchange_matrix(guess_density.first, plane_waves, lookup_table);
    arma::mat beta_exchange_matrix = generate_exchange_matrix(guess_density.second, plane_waves, lookup_table);

    return make_pair(hcore - 0.5 * alpha_exchange_matrix, hcore - 0.5 * beta_exchange_matrix);
}

arma::mat UHF::generate_exchange_matrix(arma::mat &density, const vector<tuple<int, int, int>> &plane_waves, const arma::mat &lookup_table) {
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

double UHF::compute_uhf_energy(pair<arma::mat, arma::mat> &density_matrix, pair<arma::mat, arma::mat> &fock_matrices) {
    double energy = 0.0;
    for (int i = 0; i < n_pw; ++i) {
        for (int j = 0; j < n_pw; ++j) {
            energy += ((density_matrix.first(j, i) + density_matrix.second(j, i)) * kinetic(i, j) + 
                             (density_matrix.first(j, i) * fock_matrices.first(i, j) + density_matrix.second(j, i) * fock_matrices.second(i, j)));
        }
    }
    return 0.5 * energy;
}

pair<arma::mat, arma::mat> UHF::generate_uhf_density_matrix(pair<arma::mat, arma::mat> &eigenvectors) {
    int n_occ = nelec / 2;
    arma::mat occupied_alpha_eigenvectors = eigenvectors.first.cols(0, n_occ - 1);
    arma::mat occupied_beta_eigenvectors = eigenvectors.second.cols(0, n_occ - 1);
    arma::mat alpha_density_matrix = (occupied_alpha_eigenvectors * occupied_alpha_eigenvectors.t());
    arma::mat beta_density_matrix = (occupied_beta_eigenvectors * occupied_beta_eigenvectors.t());
    return make_pair(alpha_density_matrix, beta_density_matrix);    
}