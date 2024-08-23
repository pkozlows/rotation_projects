#include <iostream>
#include <fstream>
#include <armadillo>
#include <cassert>
#include <map>
#include "basis.h"
#include "rhf.h"
#include "uhf.h"
#include "matrix_utils.h"
#include <cmath>

using namespace std;

// Function to run SCF and return the converged energy
double run_scf(Basis_3D &basis, const size_t &n_elec, const float &rs, ofstream &results_file, bool use_rhf, float spin_polarisation = 0.0) {

    auto [n_pw, plane_waves] = basis.generate_plan_waves();
    auto [n_mom, momentum_transfer_vectors] = basis.generate_momentum_transfer_vectors();
    

    pair<arma::Mat<int>, arma::Mat<int>> lookup_tables = make_pair(basis.generate_momentum_lookup_table(), basis.generate_pw_lookup_table());
    const arma::mat kinetic = basis.kinetic_integrals();

    const arma::vec integrals = basis.interaction_integrals();
    const double madeleung_constant = basis.compute_madeleung_constant();

    double previous_energy = 0.0;
    double energy = 0.0;
    int iteration = 0;
    const double density_threshold = 1e-6;
    const double energy_threshold = 1e-9;
    const double volume = 4.0 * n_elec / 3.0 * M_PI * pow(rs, 3);
    if (use_rhf) {
        RHF rhf(kinetic, integrals, n_elec, n_pw, n_mom, plane_waves, momentum_transfer_vectors, lookup_tables, volume);
        arma::mat rhf_guess = rhf.guess_rhf("random");

        do {
            arma::mat fock_matrix = rhf.make_fock_matrix(previous_guess);
            cout << "The fock matrix is: " << endl;
            cout << fock_matrix << endl;
            arma::vec eigenvalues;
            arma::mat eigenvectors;
            arma::eig_sym(eigenvalues, eigenvectors, fock_matrix);
            arma::mat new_density = rhf.generate_density_matrix(eigenvectors);

            energy = rhf.compute_energy(new_density, fock_matrix);

            double density_difference = arma::norm(new_density - rhf_guess, "fro");

            if (density_difference < density_threshold && abs(energy - previous_energy) < energy_threshold) {
                cout << "SCF Converged in " << iteration << " iterations." << endl;
                cout << "The diagonal of the final density matrix is: " << endl;
                cout << new_density.diag() << endl;
                break;
            }

            rhf_guess += new_density;
            rhf_guess /= 2;
            previous_energy = energy;
            iteration++;

        } while (iteration < 500);

        // Check if the SCF didn't converge
        if (iteration >= 500) {
            cout << "SCF did not converge after X iterations." << endl;
            cout << "The diagonal of the last density matrix is: " << endl;
            cout << rhf_guess.diag() << endl;
        }


    } else {
        UHF uhf(kinetic, integrals, n_elec, n_pw, n_mom, plane_waves, momentum_transfer_vectors, lookup_tables, volume, spin_polarisation);
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
            pair<arma::mat, arma::mat> eigenvectors;

            eigenvectors = eigenvecs;
            uhf_guess = new_density;

            energy = uhf.compute_uhf_energy(new_density, fock_matrices);
            // cout << "The energy is " << energy << " after " << iteration << " iterations." << endl;

            if (abs(energy - previous_energy) < energy_threshold) {
                // cout << "The converged UHF energy is: " << energy << " after " << iteration << " iterations." << " and density matrix is: " << endl;
                // cout << uhf_guess.first << endl;
                // cout << uhf_guess.second << endl;
                break;
            }

            previous_energy = energy;
            iteration++;

        } while (iteration < 200);
    }

    
    return energy;
}

int main() {
    ofstream results_file("hf_ueg/plt/scf_id.txt");

    size_t n_elec = 14;

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
    for (float rs = 0.5; rs <= 4; rs += 0.5) {
        
        cout << "--------------------------------" << endl;
        cout << "Starting rs = " << rs << endl;
        cout << "--------------------------------" << endl;

        
        Basis_3D basis_3d(rs, n_elec);

        double rhf_energy = run_scf(basis_3d, n_elec, rs, results_file, true);
        // double uhf_energy = run_scf(basis_3d, n_elec, rs, results_file, false, 0);

        cout << "Computed RHF: " << rhf_energy << endl;
        // cout << "Computed UHF: " << uhf_energy << endl;

        results_file << "Computed RHF: " << rhf_energy << endl;
        cout << "--------------------------------" << endl;

        cout << "Reference RHF: " << rs_to_rhf[rs] << endl;
        // cout << "Reference UHF: " << rs_to_uhf_m179[rs] << endl;

        results_file << "Reference RHF: " << rs_to_rhf[rs] << endl;
    }
    return 0;
}