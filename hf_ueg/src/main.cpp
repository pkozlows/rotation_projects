#include <iostream>
#include <fstream>
#include <armadillo>
#include <vector>
#include <cassert>
#include <map>
#include "basis.h"
#include "scf.h"
#include "matrix_utils.h"

using namespace std;

// Function to run SCF and return the converged energy
double run_scf(Basis_3D &basis, const int nelec, ofstream &results_file, double rs, bool use_rhf) {
    auto [n_pw, sorted_plane_waves] = basis.generate_plan_waves();
    // cout << "Number of plane waves: " << n_pw << endl;
    // results_file << "Number of plane waves: " << n_pw << endl;

    arma::mat lookup_table = basis.make_lookup_table();
    arma::mat kinetic_integral_matrix = basis.kinetic_integrals();
    double homo_e = kinetic_integral_matrix.diag()(nelec / 2);
    cout << "The HOMO energy is: " << homo_e << endl;
    arma::vec exchange_integral_matrix = basis.exchangeIntegrals();
    double madeleung_constant = basis.compute_madeleung_constant();

    double previous_energy = 0.0;
    double energy = 0.0;
    int iteration = 0;
    const double density_threshold = 1e-6;
    const double energy_threshold = 1e-6;

    if (use_rhf) {
        RHF rhf(kinetic_integral_matrix, exchange_integral_matrix, nelec, n_pw, sorted_plane_waves, lookup_table, madeleung_constant);
        arma::mat rhf_guess = rhf.guess_rhf("identity");

        do {
            arma::mat fock_matrix = rhf.make_fock_matrix(rhf_guess);
            arma::vec eigenvalues;
            arma::mat eigenvectors;
            arma::eig_sym(eigenvalues, eigenvectors, fock_matrix);
            arma::mat new_density = rhf.generate_density_matrix(eigenvectors);
            rhf_guess = (new_density + rhf_guess) / 2;

            energy = rhf.compute_energy(rhf_guess, fock_matrix);

            if (abs(energy - previous_energy) < energy_threshold) {
                // cout << "The converged RHF energy is: " << energy << " after " << iteration << " iterations." << endl;
                break;
            }

            previous_energy = energy;
            iteration++;

        } while (iteration < 100);
    } else {
        UHF uhf(kinetic_integral_matrix, exchange_integral_matrix, nelec, n_pw, sorted_plane_waves, lookup_table, madeleung_constant);
        pair<arma::mat, arma::mat> uhf_guess = uhf.guess_uhf();

        do {
            pair<arma::mat, arma::mat> fock_matrices = uhf.make_uhf_fock_matrix(uhf_guess);
            arma::mat fock_alpha = fock_matrices.first;
            arma::vec eigenvalues_alpha;
            arma::mat eigenvectors_alpha;
            arma::eig_sym(eigenvalues_alpha, eigenvectors_alpha, fock_alpha);

            arma::mat fock_beta = fock_matrices.second;
            arma::vec eigenvalues_beta;
            arma::mat eigenvectors_beta;
            arma::eig_sym(eigenvalues_beta, eigenvectors_beta, fock_beta);

            pair<arma::mat, arma::mat> eigenvecs = make_pair(eigenvectors_alpha, eigenvectors_beta);
            pair<arma::mat, arma::mat> new_density = uhf.generate_uhf_density_matrix(eigenvecs);

            // //get the trace of both density matrices
            // double trace_alpha = arma::trace(new_density.first);
            // double trace_beta = arma::trace(new_density.second);
            // cout << "The trace of the alpha density matrix is: " << trace_alpha << endl;
            // cout << "The trace of the beta density matrix is: " << trace_beta << endl;

            uhf_guess = new_density;

            energy = uhf.compute_uhf_energy(new_density, fock_matrices);

            if (abs(energy - previous_energy) < energy_threshold) {
                // cout << "The converged UHF energy is: " << energy << " after " << iteration << " iterations." << endl;
                break;
            }

            previous_energy = energy;
            iteration++;

        } while (iteration < 100);
    }

    return energy;
}

int main() {
    ofstream results_file("hf_ueg/plt/scf_id.txt");

    int nelec = 14;
    double rs = 5.0; // Set your desired value for rs

    // Reference RHF and UHF energies
    double rs_values[] = {0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0};
    double rhf_values[] = {58.592675, 13.603557, 5.581754, 2.878584, 1.675156, 1.047235, 0.684122, 0.458493, 0.310680, 0.209867, 0.138911, 0.087707, 0.050008, 0.021800, 0.000420, -0.015953, -0.028590, -0.038398, -0.046037, -0.051994};
    double uhf_values_m179[] = {58.592675, 13.603557, 5.581754, 2.878584, 1.675156, 1.047235, 0.684122, 0.456724, 0.301432, 0.191606, 0.112045, 0.053268, 0.009146, -0.024404, -0.050177, -0.070130, -0.085664, -0.097798, -0.107288, -0.114699};

    map<double, double> rs_to_rhf;
    map<double, double> rs_to_uhf_m179;

    for (int i = 0; i < 20; ++i) {
        rs_to_rhf[rs_values[i]] = rhf_values[i];
        rs_to_uhf_m179[rs_values[i]] = uhf_values_m179[i];
    }

    // find out the carver spawning table rhf value
    double carver_rhf = rs_to_rhf[rs];
    cout << "The carver RHF value is: " << carver_rhf << endl;
    cout << "___________________ " << endl;


    // Test constants
    vector<double> constants = {5.25, 5.375, 5.5, 7.596333120576};

    for (double constant : constants) {
        const double ke_cutoff = 15 / pow(rs, 2);
        cout << "Testing constant: " << constant << endl;
        results_file << "Constant: " << constant << endl;

        Basis_3D basis_3d(ke_cutoff, rs, nelec, constant);
        double fermi_energy = basis_3d.compute_fermi_energy();
        cout << "Computed Fermi energy: " << fermi_energy << endl;

        // Run both RHF and UHF
        double rhf_energy = run_scf(basis_3d, nelec, results_file, rs, true);
        // cout << "Starting unrestricted "  << endl;
        // double uhf_energy = run_scf(basis_3d, nelec, results_file, rs, false);

        // Output results
        cout << "Computed RHF: " << rhf_energy << endl;
        // cout << "Computed UHF: " << uhf_energy << endl;

        results_file << "Computed RHF: " << rhf_energy << endl;
        // results_file << "Computed UHF: " << uhf_energy << endl;
    }

    results_file.close();
    return 0;
}

