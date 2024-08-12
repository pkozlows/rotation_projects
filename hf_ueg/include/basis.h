#ifndef BASIS_H
#define BASIS_H
#include <armadillo>
#include <vector>
#include <tuple>


using namespace std;

class Basis_3D {
public:
    Basis_3D(const double &rs, const int &n_elec);
    pair<int, vector<tuple<int, int, int>>> generate_plan_waves();
    pair<size_t, vector<tuple<int, int, int>>> generate_momentum_transfer_vectors();
    arma::Mat<int> make_lookup_table();
    arma::mat kinetic_integrals();
    arma::vec exchangeIntegrals();
    double compute_madeleung_constant();
    double compute_fermi_energy();
    

private:
    double rs;
    int n_elec;
    vector<double> kinetic_energies;
    vector<tuple<int, int, int>> plane_waves;
    vector<tuple<int, int, int>> momentum_transfer_vectors;
    int n_pw;
    size_t n_mom;
    int max_n;
};





#endif