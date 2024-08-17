#ifndef BASIS_H
#define BASIS_H
#include <armadillo>
#include <vector>


using namespace std;

class Basis_3D {
public:
    Basis_3D(const float &rs, const size_t &n_elec);
   
    pair<size_t, arma::Mat<int>> generate_plan_waves();
    pair<size_t, arma::Mat<int>> generate_momentum_transfer_vectors();
    arma::Mat<int> make_lookup_table();
    arma::mat kinetic_integrals();
    arma::vec interaction_integrals();
    double compute_madeleung_constant();
    double compute_fermi_energy();
    

private:
    float rs;
    size_t n_elec;
    arma::vec kinetic_energies;
    arma::Mat<int> plane_waves;
    arma::Mat<int> momentum_transfer_vectors;
    size_t n_pw;
    size_t n_mom;
    int max_n;
};





#endif