#ifndef UHF_H
#define UHF_H

#include "basis.h"

class UHF : public Basis {
public:
    UHF(const Basis &basis, float spin_polarisation) : Basis(basis), spin_polarisation(spin_polarisation) {}

    std::pair<arma::mat, arma::mat> guess(const std::string &guess_type);

    std::pair<arma::mat, arma::mat> make_fock_matrix(const std::pair<arma::mat, arma::mat> &density_matrix);

    std::pair<arma::mat, arma::mat> generate_density_matrix(const std::pair<arma::mat, arma::mat> &fock_matrix);

    double compute_energy(const std::pair<arma::mat, arma::mat> &density_matrix, const std::pair<arma::mat, arma::mat> &fock_matrices);

    double calculate_density_difference(const std::pair<arma::mat, arma::mat> &new_density, const std::pair<arma::mat, arma::mat> &previous_guess);

    void print_density_matrix(const std::pair<arma::mat, arma::mat> &density_matrix);

    void update_density_matrix(std::pair<arma::mat, arma::mat> &previous_guess, const std::pair<arma::mat, arma::mat> &new_density);

private:
    float spin_polarisation;
};


#endif // UHF_H
