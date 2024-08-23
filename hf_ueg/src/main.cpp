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
template <typename SCFType>
double run_scf(SCFType &scf_procedure, ofstream &results_file) {
    double previous_energy = 0.0;
    double energy = 0.0;
    int iteration = 0;
    const double density_threshold = 1e-6;
    const double energy_threshold = 1e-9;

    // Guess initialization for RHF or UHF
    auto previous_guess = scf_procedure.guess("random");

    do {
        // Fock matrix creation
        auto fock_matrix = scf_procedure.make_fock_matrix(previous_guess);

        // Generate density matrix
        auto new_density = scf_procedure.generate_density_matrix(fock_matrix);

        // Compute energy
        energy = scf_procedure.compute_energy(new_density, fock_matrix);

        // Calculate density difference
        double density_difference = scf_procedure.calculate_density_difference(new_density, previous_guess);

        if (density_difference < density_threshold && abs(energy - previous_energy) < energy_threshold) {
            cout << "SCF Converged in " << iteration << " iterations." << endl;
            scf_procedure.print_density_matrix(new_density);
            break;
        }

        scf_procedure.update_density_matrix(previous_guess, new_density);

        previous_energy = energy;
        iteration++;

    } while (iteration < 500);

    if (iteration >= 500) {
        cout << "SCF did not converge after " << iteration << " iterations." << endl;
        scf_procedure.print_density_matrix(previous_guess);
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

    for (float rs = 2; rs <= 5; rs += 0.5) {
        cout << "--------------------------------" << endl;
        cout << "Starting rs = " << rs << endl;
        cout << "--------------------------------" << endl;

        Basis basis(rs, n_elec);  // Initialize once

        // RHF rhf(basis);  // Use the initialized basis
        // double rhf_energy = run_scf(rhf, results_file);
        // cout << "Computed RHF: " << rhf_energy << endl;
        cout << "Reference RHF: " << rs_to_rhf[rs] << endl;
        // results_file << "Computed RHF: " << rhf_energy << endl;

        UHF uhf(basis, 0.0);  // Use the same basis
        double uhf_energy = run_scf(uhf, results_file);
        cout << "Computed UHF: " << uhf_energy << endl;
        cout << "Reference UHF: " << rs_to_uhf_m179[rs] << endl;
        // results_file << "Computed UHF: " << uhf_energy << endl;

        cout << "--------------------------------" << endl;
    }
    return 0;
}

