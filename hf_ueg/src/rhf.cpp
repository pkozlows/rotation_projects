#include "rhf.h"
#include <cassert>

using namespace std;


arma::mat RHF::guess_rhf(const string &guess_type) {
    //for this guess of the density matrix I want the elements to be random numbers between 0 and 1
    //make all the elements of the matrix trust 2s
    // arma::mat density_matrix(n_pw, n_pw, arma::fill::ones);
    
    arma::mat density_matrix = arma::randu<arma::mat>(n_pw, n_pw);
    // I want to make the density matrix symmetric so I add the transpose of the matrix to itself
    density_matrix += density_matrix.t();
    // arma::mat density_matrix(n_pw, n_pw, arma::fill::zeros);
    // add the identity matrix to the density matrix but only for the occupied orbitals
    // for (size_t i = 0; i < n_elec / 2; ++i) {
    //     density_matrix(i, i) = 2;
    // }


    return density_matrix;
}

arma::mat RHF::make_fock_matrix(arma::mat &guess_density) {

    arma::mat hartree(n_pw, n_pw, arma::fill::zeros);
    arma::mat exchange_matrix(n_pw, n_pw, arma::fill::zeros);
    
    // we can do the hartree and exchange terms in the same loops
    for (size_t mu = 0; mu < n_pw; ++mu) {
        for (size_t nu = 0; nu < n_pw; ++nu) {
            //defend the momentum transfer vector
            int mu_m_nu_x = plane_waves(0, mu) - plane_waves(0, nu);
            int mu_m_nu_y = plane_waves(1, mu) - plane_waves(1, nu);
            int mu_m_nu_z = plane_waves(2, mu) - plane_waves(2, nu);
            //find the index of this momentum transfer vector
            int index = -1;
            for (size_t i = 0; i < n_mom; ++i) {
                if (momentum_transfer_vectors(0, i) == mu_m_nu_x && momentum_transfer_vectors(1, i) == mu_m_nu_y && momentum_transfer_vectors(2, i) == mu_m_nu_z) {
                    index = i;
                    break;
                }
            }
            assert(index != -1);
            for (size_t lambda = 0; lambda < n_pw; ++lambda) {

                for (size_t sigma = 0; sigma < n_pw; ++sigma) {
                    //compute the second momentum transfer vector mu - sigma
                    int mu_m_sigma_x = plane_waves(0, nu) - plane_waves(0, lambda);
                    int mu_m_sigma_y = plane_waves(1, nu) - plane_waves(1, lambda);
                    int mu_m_sigma_z = plane_waves(2, nu) - plane_waves(2, lambda);
                    //find the index of this momentum transfer vector
                    int index2 = -1;
                    for (size_t i = 0; i < n_mom; ++i) {
                        if (momentum_transfer_vectors(0, i) == mu_m_sigma_x && momentum_transfer_vectors(1, i) == mu_m_sigma_y && momentum_transfer_vectors(2, i) == mu_m_sigma_z) {
                            index2 = i;
                            break;
                        }
                    }
                    assert(index2 != -1);
                    hartree(mu, nu) += guess_density(lambda, sigma) * interaction(index);
                    exchange_matrix(mu, nu) += guess_density(lambda, sigma) * interaction(index2);
                }
            }

        }
    }
    // //I want you to print out the kinetic, normalized hartree and exchange matrices
    // cout << "The kinetic matrix is: " << endl;
    // cout << kinetic << endl;
    cout << "The hartree matrix is: " << endl;
    cout << hartree / volume << endl;
    cout << "The exchange matrix is: " <<endl;
    cout << exchange_matrix / volume << endl;
    
    return kinetic + (hartree - 0.5 * exchange_matrix) / volume;
}


double RHF::compute_energy(arma::mat &density_matrix, arma::mat &fock_matrix) {
    double energy = 0.0;
    for (size_t i = 0; i < n_pw; ++i) {
        for (size_t j = 0; j < n_pw; ++j) {
            energy += density_matrix(i, j) * (kinetic(i, j) + fock_matrix(i, j));
        }
    }
    return 0.5 * energy;
}

arma::mat RHF::generate_density_matrix(arma::mat &eigenvectors) {
    size_t n_occ = n_elec / 2;
    arma::mat occupied_eigenvectors = eigenvectors.cols(0, n_occ - 1);
    arma::mat density_matrix = 2 * (occupied_eigenvectors * occupied_eigenvectors.t());
    return density_matrix;
}