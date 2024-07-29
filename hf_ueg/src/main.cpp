#include <iostream>
#include <fstream>
#include <armadillo>
#include <vector>
#include <cassert>
#include "basis.h"
#include "scf.h"
#include "matrix_utils.h"

using namespace std;

void run_scf(Basis &basis, const int nelec, ofstream &results_file) {
    auto [n_pw, sorted_plane_waves] = basis.n_plane_waves(); // Unpack the pair
    cout << "Number of plane waves: " << n_pw << endl;
    results_file << "Number of plane waves: " << n_pw << endl;
    // assert(n_pw > 12 && n_pw < 250);

    arma::mat kinetic_integral_matrix = basis.kinetic_integrals();
    arma::mat exchange_integral_matrix = basis.exchangeIntegrals();

    // Generate the SCF object
    Scf rhf(kinetic_integral_matrix, exchange_integral_matrix, nelec, n_pw, sorted_plane_waves);

    // Generate initial guess for the density matrix
    arma::mat guess = rhf.zeros_guess();
    // arma::mat guess = rhf.identity_guess();
    
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
        
        rhf_energy = rhf.compute_rhf_energy(guess, fock_matrix);

        results_file << iteration << " " << rhf_energy / nelec << endl;
        cout << "Energy: " << rhf_energy / nelec << " at iteration: " << iteration << endl;

        arma::mat new_density = rhf.generate_density_matrix(eigenvectors);
        //find the trace of the new density matrix
        double trace = arma::trace(new_density);
        cout << "Trace of the density matrix: " << trace << endl;
        // print_matrix(new_density);
        // Calculate the Frobenius norm of the difference between the new and old density matrices
        double density_diff = arma::norm(new_density - guess, "fro");
        cout << "Density difference: " << density_diff << endl;

        // Check for convergence
        if (density_diff < density_threshold && std::abs(rhf_energy - previous_energy) < energy_threshold) {
            std::cout << "The converged RHF energy is: " << rhf_energy / nelec << " after " << iteration << " iterations." << std::endl;
            break;
        }

        previous_energy = rhf_energy;
        guess = (new_density + guess) / 2;
        iteration++;

    } while (iteration < 100);
}

int main() {
    const double ke_cutoff = 1;
    const double rs = 4;
    // Open a file to save the results
    ofstream results_file("hf_ueg/plt/scf_ld.txt");

    for (int nelec = 14; nelec <= 14; nelec += 2) {
        cout << "Number of electrons: " << nelec << endl;
        results_file << "Number of electrons: " << nelec << endl;
        const int n_elec = nelec;
        Basis_3D basis_3d(ke_cutoff, rs, n_elec);
        run_scf(basis_3d, n_elec, results_file);

        // Basis_2D basis_2d(ke_cutoff, rs, n_elec);
        // run_scf(basis_2d, n_elec, results_file);
    }

    results_file.close();

    return 0;
}
