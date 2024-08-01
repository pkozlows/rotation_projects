#include "rhf.h"

// RHF class constructor
RHF::RHF(const arma::mat &kinetic, const arma::vec &coulomb, const int &nelec, const int &npws, 
         const vector<tuple<int, int, int>> &plane_waves, const arma::mat &lookup_table, 
         const double &madeleung_constant)
    : kinetic(kinetic), exchange(coulomb), nelec(nelec), n_pw(npws), plane_waves(plane_waves), 
      lookup_table(lookup_table), madeleung_constant(madeleung_constant) {}
// RHF methods
arma::mat RHF::identity_guess() {
    arma::mat density_matrix(n_pw, n_pw, arma::fill::zeros);
    for (int i = 0; i < nelec / 2; ++i) {
        density_matrix(i, i) = 2.0;
    }
    return density_matrix;
}

arma::mat RHF::zeros_guess() {
    arma::mat density_matrix(n_pw, n_pw, arma::fill::zeros);
    return density_matrix;
}

arma::mat RHF::make_fock_matrix(const arma::mat &guess_density) {
    int npws = guess_density.n_rows;
    arma::mat hcore = kinetic;  // Ensure kinetic is properly initialized

    arma::mat exchange_matrix(npws, npws, arma::fill::zeros);

    for (int p = 0; p < npws; ++p) {
        auto g_p = plane_waves[p];

        for (int q = 0; q < npws; ++q) {
            auto g_q = plane_waves[q];

            double sum = 0.0;
            for (int Q = 0; Q < npws; ++Q) {
                auto g_Q = plane_waves[Q];

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

double RHF::compute_energy(const arma::mat &density_matrix, const arma::mat &fock_matrix) {
    double energy = 0.0;
    for (int i = 0; i < nelec / 2; ++i) {
        energy += fock_matrix(i, i) + kinetic(i, i);
    }
    return energy;
}

arma::mat RHF::generate_density_matrix(const arma::mat &eigenvectors) {
    int n_occ = nelec / 2;
    arma::mat occupied_eigenvectors = eigenvectors.cols(0, n_occ - 1);
    arma::mat density_matrix = 2 * (occupied_eigenvectors * occupied_eigenvectors.t());
    return density_matrix;
}
