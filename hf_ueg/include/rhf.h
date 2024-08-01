#ifndef RHF_H
#define RHF_H
#include <armadillo>
#include <vector>

using namespace std;
class RHF {
    public:
    //make a constructor to take in the kinetic and clump matrices
    RHF(const arma::mat &kinetic, const arma::vec &coulomb, const int &nelec, const int &npws, const vector<tuple<int, int, int>> &plane_waves, const arma::mat &lookup_table, const double &madeleung_constant);
    //make a function that generates an initial guess for the density matrix
    arma::mat identity_guess();
    arma::mat zeros_guess();
    //make a function that generates the Fock matrix from the guess density
    arma::mat make_fock_matrix(const arma::mat &guess_density);
    //compute the RHF energy from the eigenvalues
    double compute_energy(const arma::mat &density_matrix, const arma::mat &fock_matrix);
    //generate a matrix of coefficients and density by diagonalizing the fock matrix 
    arma::mat generate_density_matrix(const arma::mat &eigenvectors);
    private:
        arma::mat kinetic;
        arma::vec exchange;
        arma::mat lookup_table;
        double madeleung_constant;
        int nelec;
        int n_pw;
        vector<tuple<int, int, int>> plane_waves;
};
#endif
