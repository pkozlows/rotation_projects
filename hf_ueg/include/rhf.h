#ifndef RHF_H
#define RHF_H

#include "basis.h"

class RHF : public Basis {
public:
    RHF(const Basis &basis) : Basis(basis) {}

    arma::mat guess(const std::string &guess_type);

    arma::mat make_fock_matrix(const arma::mat &density_matrix);

    arma::mat generate_density_matrix(const arma::mat &fock_matrix);

    double compute_energy(const arma::mat &density_matrix, const arma::mat &fock_matrix);

    double calculate_density_difference(const arma::mat &new_density, const arma::mat &previous_guess);

    void print_density_matrix(const arma::mat &density_matrix);

    void update_density_matrix(arma::mat &previous_guess, const arma::mat &new_density);


private:
arma::mat compute_hartree_matrix(const arma::mat &density_matrix);
arma::mat compute_exchange_matrix(const arma::mat &density_matrix);
};





#endif // RHF_H
