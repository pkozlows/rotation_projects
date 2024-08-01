#include <iostream>
#include <fstream>
#include <armadillo>
#include <vector>
#include <cassert>
#include <map>
#include "basis.h"
#include "uhf.h"
#include "matrix_utils.h"

using namespace std;

void run_scf(Basis_3D &basis, const int nelec, ofstream &results_file, double rs) {
    auto [n_pw, sorted_plane_waves] = basis.generate_plan_waves(); // Unpack the pair
    cout << "Number of plane waves: " << n_pw << endl;
    results_file << "Number of plane waves: " << n_pw << endl;

    arma::mat lookup_table = basis.make_lookup_table();

    arma::mat kinetic_integral_matrix = basis.kinetic_integrals();
    // Determine the kinetic energy of the HOMO
    double homo_e = kinetic_integral_matrix.diag()(nelec / 2);
    cout << "The HOMO energy is:" << homo_e << endl;
    // Now to determine the Fermi energy 
    double fermi_energy = basis.compute_fermi_energy();
    cout << "The Fermi energy is:" << fermi_energy << endl;
    arma::vec exchange_integral_matrix = basis.exchangeIntegrals();

    double madeleung_constant = basis.compute_madeleung_constant();

    // Generate the SCF object
    UHF uhf(kinetic_integral_matrix, exchange_integral_matrix, nelec, n_pw, sorted_plane_waves, lookup_table, madeleung_constant);

    // Generate initial guess for the density matrix
    pair<arma::mat, arma::mat> combination_guess = uhf.combination_guess();
    // arma::mat guess = uhf.identity_guess();
    
    double previous_energy = 0.0;
    double uhf_energy = 0.0;
    int iteration = 0;
    // const double density_threshold = 1e-6;
    const double energy_threshold = 1e-6;

    do {
        pair<arma::mat, arma::mat> fock_matrices = uhf.make_fock_matrices(combination_guess);
        // cout << "The fock matrices are:" << endl;
        // print_matrix(fock_matrix);
        arma::mat fock_alpha = fock_matrices.first;
        arma::vec eigenvalues_alpha;
        arma::mat eigenvectors_alpha;
        arma::eig_sym(eigenvalues_alpha, eigenvectors_alpha, fock_alpha);

        arma::mat fock_beta = fock_matrices.second;
        arma::vec eigenvalues_beta;
        arma::mat eigenvectors_beta;
        arma::eig_sym(eigenvalues_beta, eigenvectors_beta, fock_beta);

        pair<arma::mat, arma::mat> eigenvecs = make_pair(eigenvectors_alpha, eigenvectors_beta);
        
        // cout << "Eigenvalues: " << eigenvalues << endl;
        pair<arma::mat, arma::mat> new_density = uhf.generate_density_matrices(eigenvecs);
        combination_guess = new_density; 
        
        double uhf_energy = uhf.compute_uhf_energy(new_density, fock_matrices);


        

        results_file << iteration << " " << uhf_energy << endl;
        cout << "Energy: " << uhf_energy << " at iteration: " << iteration << endl;

        // // Calculate the Frobenius norm of the difference between the new and old density matrices
        // double density_diff = arma::norm(new_density - guess, "fro");
        // cout << "Density difference: " << density_diff << endl;

        // Check for convergence
        if (std::abs(uhf_energy - previous_energy) < energy_threshold) {
            std::cout << "The converged RHF energy is: " << uhf_energy << " after " << iteration << " iterations." << std::endl;
            break;
        }

        previous_energy = uhf_energy;
        iteration++;

    } while (iteration < 100);

    // Arrays holding the r_s values and corresponding RHF values
    double rs_values[] = {0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0};
    double rhf_values[] = {58.592675, 13.603557, 5.581754, 2.878884, 1.675156, 1.047235, 0.684122, 0.456478, 0.310868, 0.209867, 0.138911, 0.090870, 0.050008, 0.012190, -0.037320, -0.050570, -0.065675, -0.082398, -0.046037, -0.051994};

    // Create a map to hold the r_s to RHF mapping
    std::map<double, double> rs_to_rhf;

    // Populate the map
    for (int i = 0; i < 20; ++i) {
        rs_to_rhf[rs_values[i]] = rhf_values[i];
    }

    // Compare the computed RHF energy with the table value for the given r_s
    if (rs_to_rhf.find(rs) != rs_to_rhf.end()) {
        double table_rhf_energy = rs_to_rhf[rs];
        std::cout << "Given r_s: " << rs << ", Table RHF energy: " << table_rhf_energy << ", Computed RHF energy: " << uhf_energy << std::endl;
    } else {
        std::cout << "r_s value not found in the table." << std::endl;
    }
}

// int main() {
//     const double ke_cutoff = 1;
//     const double rs = 2; // Change this value to compare with other r_s values
//     // Open a file to save the results
//     ofstream results_file("hf_ueg/plt/scf_uhf.txt");

//     size_t nelec = 4;
//     cout << "Number of electrons: " << nelec << endl;
//     results_file << "Number of electrons: " << nelec << endl;
//     cout << "Wigner-Seitz radius: " << rs << endl;
//     results_file << "Wigner-Seitz radius: " << rs << endl;
//     Basis_3D basis_3d(ke_cutoff, rs, nelec);
//     run_scf(basis_3d, nelec, results_file, rs);

//     results_file.close();

//     return 0;
// }
