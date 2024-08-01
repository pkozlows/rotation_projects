#ifndef UHF_H
#define UHF_H
#include <armadillo>
#include <vector>

using namespace std;
class UHF {
    public:
    //make a constructor to take in the kinetic and clump matrices
    UHF(const arma::mat &kinetic, const arma::vec &coulomb, const int &nelec, const int &npws, const vector<tuple<int, int, int>> &plane_waves, const arma::mat &lookup_table, const double &madeleung_constant);
    //make a function that generates an initial guess for the density matrix
    pair<arma::mat, arma::mat> combination_guess();
    //make a function that generates the Fock matrix from the guess density
    //make a function that generates the Fock matrix from the guess density
    pair<arma::mat, arma::mat> make_fock_matrices(const pair<arma::mat, arma::mat> &guess_density);
    
    //compute the UHF energy from the eigenvalues
    double compute_uhf_energy(const pair<arma::mat, arma::mat> &density_matrix, const pair<arma::mat, arma::mat> &fock_matrices);
    
    //generate a matrix of coefficients and density by diagonalizing the fock matrix 
    pair<arma::mat, arma::mat> generate_density_matrices(const pair<arma::mat, arma::mat> &eigenvectors);
    private:
        arma::mat generate_exchange_matrix(const arma::mat &density, const vector<tuple<int, int, int>> plane_waves, const arma::mat &lookup_table, int n_pw);
        arma::mat kinetic;
        arma::vec exchange;
        arma::mat lookup_table;
        double madeleung_constant;
        int nelec;
        int n_pw;
        vector<tuple<int, int, int>> plane_waves;
};
#endif
