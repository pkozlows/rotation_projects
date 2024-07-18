#include <iostream>
#include <armadillo>
#include <vector>
#include "scf.h"
#include "matrix_utils.h"

using namespace std;


Scf::Scf(const arma::mat& kinetic_integral, const arma::mat& coulombIntegral, const int& nelec) {
    kinetic = kinetic_integral;
    coulomb = coulombIntegral;
    this->nelec = nelec;
}




arma::mat Scf::make_fock_matrix(arma::mat &density_matrix) {
    int npws = density_matrix.n_rows;
    arma::mat hcore = kinetic;
    
    arma::mat exchange_matrix(npws, npws, arma::fill::zeros);

    // Calculate the Coulomb contribution
    for (int i = 0; i < npws; ++i) {
        for (int j = 0; j < npws; ++j) {
            double sum = 0.0;
            for (int k = 0; k < npws; ++k) {
                for (int l = 0; l < npws; ++l) {
                    sum += density_matrix(k, l) * coulomb(i * npws + k, l * npws + j);
                }
            }
            // if (i != j) {
            //     //fund out all of the of diagonal elements
            //     cout << "The sum for the off diagonal element" << i << " " << j << " is: " << sum << endl;
            // }
            exchange_matrix(i, j) = sum;
        }
    }

    // Since we are just considering the exchange contribution, we can subtract out 0.5 * exchange_matrix
    arma::mat fock_matrix = hcore - 0.5 * exchange_matrix;
    return fock_matrix;
}


//based on the some of the eigenvalues, compute total RHF energy 
double Scf::compute_rhf_energy(arma::vec &eigenvalues) {
    double energy = arma::sum(eigenvalues);
    return energy;
}

// Construct the density matrix
arma::mat Scf::generate_density_matrix(arma::mat &eigenvectors) {
    int n_occ = nelec / 2;
    arma::mat occupied_eigenvectors = eigenvectors.cols(0, n_occ - 1);
    arma::mat density_matrix = 2 * (occupied_eigenvectors * occupied_eigenvectors.t());
    return density_matrix;
}
