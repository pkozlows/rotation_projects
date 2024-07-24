#include <iostream>
#include <fstream>
#include <armadillo>
#include <cassert>
#include "basis.h"
#include "scf.h"
#include "matrix_utils.h"

using namespace std;

void run_scf(Basis &basis, const int nelec, ofstream &results_file, bool track_iterations) {
    int n_pw = basis.n_plane_waves();
    cout << "Number of plane waves: " << n_pw << endl;
    results_file << "Number of plane waves: " << n_pw << endl;
    assert(n_pw > 12 && n_pw < 250);

    arma::mat kinetic_integral_matrix = basis.kinetic_integrals();
    arma::mat coulomb_integral_matrix = basis.coulombIntegrals();

    // Generate the SCF object
    Scf rhf(kinetic_integral_matrix, coulomb_integral_matrix, nelec, n_pw);

    // Generate initial guess for the density matrix
    arma::mat guess = rhf.zeros_guess();
    
    double previous_energy = 0.0;
    double rhf_energy = 0.0;
    int iteration = 0;
    const double density_threshold = 1e-6;
    const double energy_threshold = 1e-6;

    do {
        arma::mat fock_matrix = rhf.make_fock_matrix(guess);
        arma::vec eigenvalues;
        arma::mat eigenvectors;
        arma::eig_sym(eigenvalues, eigenvectors, fock_matrix);

        arma::mat new_density = rhf.generate_density_matrix(eigenvectors);
        rhf_energy = rhf.compute_rhf_energy(new_density, fock_matrix);

        if (track_iterations) {
            results_file << iteration << " " << rhf_energy / nelec << endl;
        }

        if (arma::approx_equal(new_density, guess, "absdiff", density_threshold) &&
            abs(rhf_energy - previous_energy) < energy_threshold) {
                cout << "The converged RHF energy is: " << rhf_energy / nelec << " after " << iteration << " iterations." << endl;
                break;
        }

        previous_energy = rhf_energy;
        guess = new_density;
        iteration++;

    } while (iteration < 100);

    // Final converged energy output
    if (!track_iterations) {
        results_file << rhf_energy / nelec << endl;
    }
}

int main() {
    const double ke_cutoff = 0.1;
    const double n_elec = 14;

    // Open a file to save the results for iteration tracking
    ofstream iter_results_file("hf_ueg/plt/scf_iterations.txt");

    // Open a file to save the results for final energies comparison
    ofstream final_results_file("hf_ueg/plt/scf_final_energies.txt");

    for (double r_s = 1.0; r_s <= 10.0; r_s += 0.5) {
        Basis_3D basis_3d(ke_cutoff, r_s, n_elec);
        run_scf(basis_3d, n_elec, iter_results_file, true);
    }


    final_results_file << endl;
    

    iter_results_file.close();
    final_results_file.close();

    return 0;
}
