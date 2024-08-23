#ifndef RHF_H
#define RHF_H

#include <armadillo>
#include <vector>
#include <tuple>
#include <utility> // for std::pair

using namespace std;

// RHF class
class RHF {
public:
    RHF(const arma::mat &kinetic, const arma::vec &exchange, const size_t &n_elec, size_t &n_pw, const size_t &n_mom, arma::Mat<int> &plane_waves, arma::Mat<int> &momentum_transfer_vectors, const pair<arma::Mat<int>, arma::Mat<int>> &lookup_tables, const double &volume)
        : kinetic(kinetic), interaction(exchange), n_elec(n_elec), n_pw(n_pw), n_mom(n_mom), plane_waves(plane_waves), momentum_transfer_vectors(momentum_transfer_vectors), lookup_tables(lookup_tables), volume(volume) {}
    // RHF-specific methods
    arma::mat guess_rhf(const string &guess_type);
    arma::mat make_fock_matrix(arma::mat &guess_density);
    double compute_energy(arma::mat &density_matrix, arma::mat &fock_matrix);
    arma::mat generate_density_matrix(arma::mat &eigenvectors);

private:
    arma::mat compute_hartree_matrix(const arma::mat &guess_density);
    arma::mat compute_exchange_matrix(const arma::mat &guess_density);

    arma::mat kinetic;
    arma::vec interaction;
    size_t n_elec;
    size_t n_pw;
    size_t n_mom;
    arma::Mat<int> plane_waves;
    arma::Mat<int> momentum_transfer_vectors;
    pair<arma::Mat<int>, arma::Mat<int>> lookup_tables;
    double volume;
};

#endif // SCF_H
