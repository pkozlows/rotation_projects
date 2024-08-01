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

void run_scf(Basis_3D &basis, const int nelec, ofstream &results_file, double rs, bool use_rhf) {
    auto [n_pw, sorted_plane_waves] = basis.generate_plan_waves();
    cout << "Number of plane waves: " << n_pw << endl;
    results_file << "Number of plane waves: " << n_pw << endl;

    arma::mat lookup_table = basis.make_lookup_table();
    arma::mat kinetic_integral_matrix = basis.kinetic_integrals();
    double homo_e = kinetic_integral_matrix.diag()(nelec / 2);
    cout << "The HOMO energy is: " << homo_e << endl;
    double fermi_energy = basis.compute_fermi_energy();
    cout << "The Fermi energy is: " << fermi_energy << endl;
    arma::vec exchange_integral_matrix = basis.exchangeIntegrals();
    double madeleung_constant = basis.compute_madeleung_constant();

    double previous_energy = 0.0;
    double energy = 0.0;
    int iteration = 0;
    const double density_threshold = 1e-6;
    const double energy_threshold = 1e-6;

    // Variables for RHF/UHF computation
    RHF rhf(kinetic_integral_matrix, exchange_integral_matrix, nelec, n_pw, sorted_plane_waves, lookup_table, madeleung_constant);
    UHF uhf(kinetic_integral_matrix, exchange_integral_matrix, nelec, n_pw, sorted_plane_waves, lookup_table, madeleung_constant);
    
    // Initial guess
    arma::mat rhf_guess;
    pair<arma::mat, arma::mat> uhf_guess;

    if (use_rhf) {
        rhf_guess = rhf.guess_rhf("identity");
    } else {
        uhf_guess = uhf.guess_uhf();
    }

    do {
        if (use_rhf) {
            // RHF iteration
            arma::mat fock_matrix = rhf.make_fock_matrix(rhf_guess);
            arma::vec eigenvalues;
            arma::mat eigenvectors;
            arma::eig_sym(eigenvalues, eigenvectors, fock_matrix);
            arma::mat new_density = rhf.generate_density_matrix(eigenvectors);
            rhf_guess = (new_density + rhf_guess) / 2;

            energy = rhf.compute_energy(rhf_guess, fock_matrix);
        } else {
            // UHF iteration
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
            uhf_guess = new_density;

            energy = uhf.compute_uhf_energy(new_density, fock_matrices);
        }

        results_file << iteration << " " << energy / nelec << endl;
        cout << "Energy: " << energy << " at iteration: " << iteration << endl;

        if (use_rhf) {
            double density_diff = arma::norm(rhf.guess_rhf("identity") - rhf_guess, "fro");
            if (density_diff < density_threshold && std::abs(energy - previous_energy) < energy_threshold) {
                cout << "The converged RHF energy is: " << energy << " after " << iteration << " iterations." << endl;
                break;
            }
        } else {
            if (std::abs(energy - previous_energy) < energy_threshold) {
                cout << "The converged UHF energy is: " << energy << " after " << iteration << " iterations." << endl;
                break;
            }
        }

        previous_energy = energy;
        iteration++;

    } while (iteration < 100);

    // Compare with reference RHF energies
    double rs_values[] = {0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0};
    double rhf_values[] = {58.592675, 13.603557, 5.581754, 2.878884, 1.675156, 1.047235, 0.684122, 0.456478, 0.310868, 0.209867, 0.138911, 0.090870, 0.050008, 0.012190, -0.037320, -0.050570, -0.065675, -0.082398, -0.046037, -0.051994};

    map<double, double> rs_to_rhf;
    for (int i = 0; i < 20; ++i) {
        rs_to_rhf[rs_values[i]] = rhf_values[i];
    }

    if (rs_to_rhf.find(rs) != rs_to_rhf.end()) {
        double table_rhf_energy = rs_to_rhf[rs];
        cout << "r_s: " << rs << ", Table RHF: " << table_rhf_energy << ", Computed " << (use_rhf ? "RHF" : "UHF") << " energy: " << energy << endl;
    } else {
        cout << "r_s value not found in the table." << endl;
    }
}

int main() {
    ofstream results_file("hf_ueg/plt/scf_id.txt");

    int nelec = 14;
    cout << "Number of electrons: " << nelec << endl;
    results_file << "Number of electrons: " << nelec << endl;
    for (double rs = 4; rs <= 5.0; rs += 0.5) {
        const double ke_cutoff = 10 / pow(rs, 2);
        cout << "Wigner-Seitz radius: " << rs << endl;
        results_file << "Wigner-Seitz radius: " << rs << endl;
        Basis_3D basis_3d(ke_cutoff, rs, nelec);

        // Specify whether to run RHF (true) or UHF (false)
        run_scf(basis_3d, nelec, results_file, rs, true);  // Run RHF
        // run_scf(basis_3d, nelec, results_file, rs, false);  // Uncomment to run UHF
    }

    results_file.close();

    return 0;
}
