#ifndef SCF_H
#define SCF_H

#include <armadillo>
#include <vector>
#include <tuple>
#include <utility> // for std::pair

using namespace std;

// Base class
class Scf {
public:
    Scf(const arma::mat &kinetic, const arma::vec &exchange, const int &nelec, const int &npws,
        const vector<tuple<int, int, int>> &plane_waves, const arma::mat &lookup_table, 
        const double &madeleung_constant);

    virtual ~Scf() {}

    // Pure virtual functions for RHF
    virtual arma::mat guess_rhf(const string &guess_type) = 0;
    virtual arma::mat make_fock_matrix(const arma::mat &guess_density) = 0;
    virtual double compute_energy(const arma::mat &density_matrix, const arma::mat &fock_matrix) = 0;
    virtual arma::mat generate_density_matrix(const arma::mat &eigenvectors) = 0;

    // Pure virtual functions for UHF
    virtual pair<arma::mat, arma::mat> guess_uhf() = 0;
    virtual pair<arma::mat, arma::mat> make_uhf_fock_matrix(const pair<arma::mat, arma::mat> &guess_density) = 0;
    virtual double compute_uhf_energy(const pair<arma::mat, arma::mat> &density_matrix, const pair<arma::mat, arma::mat> &fock_matrices) = 0;
    virtual pair<arma::mat, arma::mat> generate_uhf_density_matrix(const pair<arma::mat, arma::mat> &eigenvectors) = 0;

protected:
    arma::mat kinetic;
    arma::vec exchange;
    arma::mat lookup_table;
    double madeleung_constant;
    int nelec;
    int n_pw;
    vector<tuple<int, int, int>> plane_waves;
};

// RHF class
class RHF : public Scf {
public:
    RHF(const arma::mat &kinetic, const arma::vec &exchange, const int &nelec, const int &npws, 
        const vector<tuple<int, int, int>> &plane_waves, const arma::mat &lookup_table, 
        const double &madeleung_constant);

    arma::mat guess_rhf(const string &guess_type) override;
    arma::mat make_fock_matrix(const arma::mat &guess_density) override;
    double compute_energy(const arma::mat &density_matrix, const arma::mat &fock_matrix) override;
    arma::mat generate_density_matrix(const arma::mat &eigenvectors) override;

    // UHF methods return empty pairs or defaults for RHF
    pair<arma::mat, arma::mat> guess_uhf() override { return {}; }
    pair<arma::mat, arma::mat> make_uhf_fock_matrix(const pair<arma::mat, arma::mat> &guess_density) override { return {}; }
    double compute_uhf_energy(const pair<arma::mat, arma::mat> &density_matrix, const pair<arma::mat, arma::mat> &fock_matrices) override { return 0.0; }
    pair<arma::mat, arma::mat> generate_uhf_density_matrix(const pair<arma::mat, arma::mat> &eigenvectors) override { return {}; }
};

// UHF class
class UHF : public Scf {
public:
    UHF(const arma::mat &kinetic, const arma::vec &exchange, const int &nelec, const int &npws, 
        const vector<tuple<int, int, int>> &plane_waves, const arma::mat &lookup_table, 
        const double &madeleung_constant);

    pair<arma::mat, arma::mat> guess_uhf() override;
    pair<arma::mat, arma::mat> make_uhf_fock_matrix(const pair<arma::mat, arma::mat> &guess_density) override;
    double compute_uhf_energy(const pair<arma::mat, arma::mat> &density_matrix, const pair<arma::mat, arma::mat> &fock_matrices) override;
    pair<arma::mat, arma::mat> generate_uhf_density_matrix(const pair<arma::mat, arma::mat> &eigenvectors) override;

    // RHF methods return empty matrices for UHF
    arma::mat guess_rhf(const string &guess_type) override { return {}; }
    arma::mat make_fock_matrix(const arma::mat &guess_density) override { return {}; }
    double compute_energy(const arma::mat &density_matrix, const arma::mat &fock_matrix) override { return 0.0; }
    arma::mat generate_density_matrix(const arma::mat &eigenvectors) override { return {}; }

private:
    arma::mat generate_exchange_matrix(const arma::mat &density, const vector<tuple<int, int, int>> &plane_waves, const arma::mat &lookup_table);
};

#endif // SCF_H
