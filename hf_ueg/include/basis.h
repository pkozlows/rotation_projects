#ifndef BASIS_H
#define BASIS_H

#include <armadillo>
#include <utility>

class Basis {
public:
    Basis(const float &rs, const size_t &n_elec);

protected: // Protected to allow access in derived classes
    float rs;
    size_t n_elec, n_pw, n_mom;
    int max_n;
    arma::Mat<int> plane_waves;
    arma::Mat<int> momentum_transfer_vectors;
    arma::mat kinetic_integrals;
    arma::vec interaction_integrals, kinetic_energies;
    double madeleung_constant, volume;
    arma::Mat<int> pw_lookup_table;
    arma::Mat<int> momentum_lookup_table;

    void generate_basis();
    void generate_plan_waves();
    void generate_momentum_transfer_vectors();
    void generate_momentum_lookup_table();
    void generate_pw_lookup_table();
    void compute_kinetic_integrals();
    void compute_interaction_integrals();
    void compute_madeleung_constant();
};

#endif // BASIS_H
