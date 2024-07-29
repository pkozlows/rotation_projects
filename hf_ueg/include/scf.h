#ifndef SCF_H
#define SCF_H
#include <armadillo>
#include <vector>

using namespace std;
class Scf {
    public:
    //make a constructor to take in the kinetic and clump matrices
    Scf(const arma::mat &kinetic, const arma::mat &coulomb, const int &nelec, const int &npws);
    //make a function that generates an initial guess for the density matrix
    arma::mat identity_guess();
    arma::mat zeros_guess();
    //make a function that generates the Fock matrix from the guess density
    arma::mat make_fock_matrix(arma::mat &guess_density);
    //compute the RHF energy from the eigenvalues
    double compute_rhf_energy(arma::mat &density_matrix, arma::mat &fock_matrix);
    //generate a matrix of coefficients and density by diagonalizing the fock matrix 
    arma::mat generate_density_matrix(arma::mat &eigenvectors);
    private:
        arma::mat kinetic;
        arma::mat exchange;
        int nelec;
        int n_pw;
};
#endif