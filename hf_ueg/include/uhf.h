#ifndef UHF_H
#define UHF_H

#include <armadillo>
#include <vector>
#include <tuple>
#include <utility> // for std::pair

using namespace std;
// UHF class
class UHF {
public:
    UHF(const arma::mat &kinetic, const arma::vec &exchange, const size_t &n_elec, size_t &n_pw, const size_t &n_mom, arma::Mat<int> &plane_waves, arma::Mat<int> &momentum_transfer_vectors, const arma::Mat<int> &lookup_table, const double &volume)
        : kinetic(kinetic), interaction(exchange), n_elec(n_elec), n_pw(n_pw), n_mom(n_mom), plane_waves(plane_waves), momentum_transfer_vectors(momentum_transfer_vectors), lookup_table(lookup_table), volume(volume) {}

    pair<arma::mat, arma::mat> guess_uhf();
    pair<arma::mat, arma::mat> make_uhf_fock_matrix(const pair<arma::mat, arma::mat> &guess_density);
    double compute_uhf_energy(const pair<arma::mat, arma::mat> &density_matrix, const pair<arma::mat, arma::mat> &fock_matrices);
    pair<arma::mat, arma::mat> generate_uhf_density_matrix(const pair<arma::mat, arma::mat> &eigenvectors);

private:
    arma::mat kinetic;
    arma::vec interaction;
    size_t n_elec;
    size_t n_pw;
    size_t n_mom;
    arma::Mat<int> plane_waves;
    arma::Mat<int> momentum_transfer_vectors;
    arma::Mat<int> lookup_table;
    double volume;
};

#endif // UHF_H