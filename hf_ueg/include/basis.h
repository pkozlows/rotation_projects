#ifndef BASIS_H
#define BASIS_H
#include <armadillo>
#include <vector>
#include <tuple>


using namespace std;

class Basis_3D {
public:
    Basis_3D(const double &ke_cutoff, const double &rs, const int &n_elec);
    pair<int, arma::Mat<int>> generate_plan_waves();
    arma::mat make_lookup_table();
    arma::mat kinetic_integrals();
    arma::vec exchangeIntegrals();
    double compute_madeleung_constant();
    double compute_fermi_energy();
    

private:
    double ke_cutoff;
    double rs;
    int n_elec;
    arma::vec kinetic_energies;
    arma::Mat<int> plane_waves;
    int n_pw;
};





#endif