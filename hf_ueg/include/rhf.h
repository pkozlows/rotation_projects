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
    RHF(const arma::mat &kinetic, const arma::vec &exchange, const int &n_elec, int &n_pw, const size_t &n_mom, vector<tuple<int, int, int>> &plane_waves, vector<tuple<int, int, int>> &momentum_transfer_vectors, const arma::Mat<int> &lookup_table, const double &volume)
        : kinetic(kinetic), interaction(exchange), n_elec(n_elec), n_pw(n_pw), n_mom(n_mom), plane_waves(plane_waves), momentum_transfer_vectors(momentum_transfer_vectors), lookup_table(lookup_table), volume(volume) {}
    // RHF-specific methods
    arma::mat guess_rhf(const string &guess_type);
    arma::mat make_fock_matrix(arma::mat &guess_density);
    double compute_energy(arma::mat &density_matrix, arma::mat &fock_matrix);
    arma::mat generate_density_matrix(arma::mat &eigenvectors);

private:
    arma::mat kinetic;
    arma::vec interaction;
    int n_elec;
    int n_pw;
    size_t n_mom;
    vector<tuple<int, int, int>> plane_waves;
    arma::Mat<int> lookup_table;
    double volume;
    vector<tuple<int, int, int>> momentum_transfer_vectors;
};

#endif // SCF_H
