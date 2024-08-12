#include <iostream>
#include <fstream>
#include <armadillo>
#include <vector>
#include <cassert>
#include <map>
#include "basis.h"
#include "rhf.h"
#include "matrix_utils.h"
#include <cmath>

using namespace std;

// Function to run SCF and return the converged energy
double run_scf(Basis_3D &basis, const int n_elec, ofstream &results_file, const double rs) {

    auto [n_pw, plane_waves] = basis.generate_plan_waves();
    const size_t n_mom = basis.generate_momentum_transfer_vectors();
    

    const arma::mat lookup_table = basis.make_lookup_table();
    const arma::mat kinetic = basis.kinetic_integrals();
    const arma::vec exchange = basis.exchangeIntegrals();
    const double madeleung_constant = basis.compute_madeleung_constant();
    cout << "madeleung_constant is: " << madeleung_constant << endl;

    cout << "-----------------------------------" << endl;
    double previous_energy = 0.0;
    double energy = 0.0;
    int iteration = 0;
    const double density_threshold = 1e-6;
    const double energy_threshold = 1e-6;
    const double volume = pow(pow(4.0 * M_PI * n_elec / 3.0, 1.0 / 3.0) * rs, 3);

    RHF rhf(kinetic, exchange, n_elec, n_pw, n_mom, plane_waves, lookup_table, volume);
    arma::mat rhf_guess = rhf.guess_rhf("zeros");
    // arma::mat rhf_guess = rhf.guess_rhf("identity");

    do {
        arma::mat fock_matrix = rhf.make_fock_matrix(rhf_guess);
        arma::vec eigenvalues;
        arma::mat eigenvectors;
        //print out the diagonal of the fock matrix

        arma::eig_sym(eigenvalues, eigenvectors, fock_matrix);
        cout << "The eigenvalues are: " << endl;
        for (int i = 0; i < n_pw; ++i) {
            cout << eigenvalues(i) << endl;
        }
        arma::mat new_density = rhf.generate_density_matrix(eigenvectors);
        print_matrix(new_density);
        //find out the sum of the eigenvalues
        double sum = 0.0;
        for (int i = 0; i < n_pw; ++i) {
            sum += eigenvalues(i);
        }
        cout << "The sum of the eigenvalues is: " << sum << endl;
        rhf_guess = new_density;

        energy = rhf.compute_energy(rhf_guess, fock_matrix);

        if (abs(energy - previous_energy) < energy_threshold) {
            cout << "It took this many iterations to converge RHF: " << iteration << endl;
            break;
        }

        previous_energy = energy;
        iteration++;

    } while (iteration < 100);
    return energy;
}

int main() {
    ofstream results_file("hf_ueg/plt/scf_id.txt");

    int n_elec = 14;

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
    for (double rs = 3.5; rs <= 3.5; rs += 0.5) {
        
        cout << "--------------------------------" << endl;
        cout << "Starting rs = " << rs << endl;
        cout << "--------------------------------" << endl;

        
        Basis_3D basis_3d(rs, n_elec);

        double rhf_energy = run_scf(basis_3d, n_elec, results_file, rs);

        cout << "Computed RHF: " << rhf_energy << endl;

        results_file << "Computed RHF: " << rhf_energy << endl;
        cout << "--------------------------------" << endl;

        cout << "Reference RHF: " << rs_to_rhf[rs] << endl;

        results_file << "Reference RHF: " << rs_to_rhf[rs] << endl;
    }
    return 0;
}
