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
    arma::mat make_lookup_table();
    arma::mat kinetic_integrals();
    arma::vec exchangeIntegrals();
    double compute_madeleung_constant();
    double compute_fermi_energy();
    

private:
    double rs;
    int n_elec;
    vector<double> kinetic_energies;
    vector<tuple<int, int, int>> plane_waves;
    int n_pw;
    int max_n;
};





#endif