#include <iostream>
#include <armadillo>
#include <vector>
#include <cassert>
#include "basis.h"
#include "scf.h"
#include "matrix_utils.h"

using namespace std;

int main() {
    const int nelec = 4;
    const double ke_cutoff = 16;
    const double rs = 0.8;

    Basis pw_3d(ke_cutoff, rs, nelec);
    int n_pw = pw_3d.n_plane_waves();
    cout << "The number of plane waves within the kinetic energy cutoff is: " << n_pw << endl;
    assert(n_pw > 0 && n_pw < 250);

    arma::mat kinetic_integral_matrix = pw_3d.kinetic_integrals();
    arma::mat coulomb_integral_matrix = pw_3d.coulombIntegrals();

    // Generate initial guess for the density matrix
    arma::mat guess = pw_3d.generate_initial_guess();

    // Generate the SCF object
    Scf rhf(kinetic_integral_matrix, coulomb_integral_matrix, nelec);
    
    double previous_energy = 0.0;
    int iteration = 0;  // Initialize iteration counter
    const double density_threshold = 1e-6;
    const double energy_threshold = 1e-6;

    do {
        // Make the Fock matrix
        arma::mat fock_matrix = rhf.make_fock_matrix(guess);
        // Diagonalize the Fock matrix
        arma::vec eigenvalues;
        arma::mat eigenvectors;
        arma::eig_sym(eigenvalues, eigenvectors, fock_matrix);
        cout << "The lowest eigenvalue is " << eigenvalues(0) << endl;

        // Compute the RHF energy
        double rhf_energy = rhf.compute_rhf_energy(eigenvalues);

        // Construct the new density matrix
        arma::mat new_density = rhf.generate_density_matrix(eigenvectors);

        // Check if the new density matrix is the same as the previous one
        if (arma::approx_equal(new_density, guess, "absdiff", density_threshold)) {
            cout << "Density matrix has not changed significantly in iteration " << iteration << "." << endl;
            break; // Break the loop if the density matrix has not changed
        }

        // Find the difference in the trace norm of the new density versus the old density
        arma::mat density_difference_matrix = new_density - guess;
        arma::vec diff_eigenvalues;
        arma::mat diff_eigenvectors;
        arma::eig_sym(diff_eigenvalues, diff_eigenvectors, density_difference_matrix);

        cout << "SCF iteration number: " << iteration << endl;

        // Update the variables for the next iteration
        previous_energy = rhf_energy;
        guess = new_density;  // This updates the guess for the next iteration
        iteration++;

    } while (iteration < 40); // Convergence criteria or maximum iterations

    cout << "The SCF procedure has converged!" << endl;

    return 0;
}
