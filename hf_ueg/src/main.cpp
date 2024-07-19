#include <iostream>
#include <fstream>
#include <armadillo>
#include <vector>
#include <cassert>
#include "basis.h"
#include "scf.h"
#include "matrix_utils.h"

using namespace std;

int main() {
    const int nelec = 4;
    const double ke_cutoff = 50;
    const double rs = 0.5;

    Basis_3D pw_3d(ke_cutoff, rs, nelec);
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

    // Open a file to save the results
    ofstream results_file("hf_ueg/plt/scf_results_dense.txt");
    if (!results_file.is_open()) {
        cerr << "Failed to open the results file." << endl;
        return 1;
    }
    // Write the number of electrons to the file
    results_file << "Number of electrons: " << nelec << endl;
    // Write the number of plane waves to the file
    results_file << "Number of plane waves: " << n_pw << endl;

    do {
        // Make the Fock matrix
        arma::mat fock_matrix = rhf.make_fock_matrix(guess);
        // Diagonalize the Fock matrix
        arma::vec eigenvalues;
        arma::mat eigenvectors;
        arma::eig_sym(eigenvalues, eigenvectors, fock_matrix);

        // Construct the new density matrix
        arma::mat new_density = rhf.generate_density_matrix(eigenvectors);

        // Compute the RHF energy
        double rhf_energy = rhf.compute_rhf_energy(new_density, fock_matrix);
        cout << "The RHF energy is: " << rhf_energy << endl;

        // Save the iteration number and RHF energy per electron to the results file
        results_file << iteration << " " << rhf_energy / nelec << endl;

        // Check if the new density matrix is the same as the previous one and if the energy has converged
        if (arma::approx_equal(new_density, guess, "absdiff", density_threshold) && abs(rhf_energy - previous_energy) < energy_threshold) {
            break;
        }

        cout << "SCF iteration number: " << iteration << endl;

        // Update the variables for the next iteration
        previous_energy = rhf_energy;
        guess = new_density;  // This updates the guess for the next iteration
        iteration++;

    } while (iteration < 100); // Convergence criteria or maximum iterations

    cout << "The SCF procedure has converged!" << endl;

    // Close the results file
    results_file.close();

    return 0;
}
