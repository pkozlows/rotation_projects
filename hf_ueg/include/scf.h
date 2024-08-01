#ifndef SCF_H
#define SCF_H
#include <armadillo>
#include <vector>
#include <tuple>

using namespace std;

class Scf {
public:
    Scf(const arma::mat &kinetic, const arma::vec &exchange, const int &nelec, const int &npws, const vector<tuple<int, int, int>> &plane_waves, const arma::mat &lookup_table, const double &madeleung_constant)
        : kinetic(kinetic), exchange(exchange), nelec(nelec), n_pw(npws), plane_waves(plane_waves), lookup_table(lookup_table), madeleung_constant(madeleung_constant) {}
    
    virtual ~Scf() {}

    virtual arma::mat zeros_guess() = 0;

    // Overloaded method for RHF
    virtual auto make_fock_matrix(const auto &guess_density) = 0;

    virtual double compute_energy(const arma::mat &density_matrix, const arma::mat &fock_matrix) = 0;
    virtual double compute_energy(const pair<arma::mat, arma::mat> &density_matrix, const pair<arma::mat, arma::mat> &fock_matrix) = 0;
    virtual arma::mat generate_density_matrix(const arma::mat &eigenvectors) = 0;
    virtual pair<arma::mat, arma::mat> generate_density_matrix(const pair<arma::mat, arma::mat> &eigenvectors) = 0;

protected:
    arma::mat kinetic;
    arma::vec exchange;
    arma::mat lookup_table;
    double madeleung_constant;
    int nelec;
    int n_pw;
    vector<tuple<int, int, int>> plane_waves;
};

class RHF : public Scf {
public:
    RHF(const arma::mat &kinetic, const arma::vec &exchange, const int &nelec, const int &npws, const vector<tuple<int, int, int>> &plane_waves, const arma::mat &lookup_table, const double &madeleung_constant)
        : Scf(kinetic, exchange, nelec, npws, plane_waves, lookup_table, madeleung_constant) {}

    arma::mat identity_guess();
    arma::mat zeros_guess();
    arma::mat make_fock_matrix(const arma::mat &guess_density);
    double compute_energy(const arma::mat &density_matrix, const arma::mat &fock_matrix);
    arma::mat generate_density_matrix(const arma::mat &eigenvectors);
};

class UHF : public Scf {
public:
    UHF(const arma::mat &kinetic, const arma::vec &exchange, const int &nelec, const int &npws, const vector<tuple<int, int, int>> &plane_waves, const arma::mat &lookup_table, const double &madeleung_constant)
        : Scf(kinetic, exchange, nelec, npws, plane_waves, lookup_table, madeleung_constant) {}

    pair<arma::mat, arma::mat> UHF::combination_gas();
    pair<arma::mat, arma::mat> make_fock_matrix(const pair<arma::mat, arma::mat> &guess_density);
    double compute_energy(const pair<arma::mat, arma::mat> &density_matrix, const pair<arma::mat, arma::mat> &fock_matrix);
    pair<arma::mat, arma::mat> UHF::generate_density_matrix(const pair<arma::mat, arma::mat> &eigenvectors);

    arma::mat generate_exchange_matrix(const arma::mat &density, const vector<int> &plane_waves, const arma::imat &lookup_table, int n_pw);
};

#endif // SCF_H
