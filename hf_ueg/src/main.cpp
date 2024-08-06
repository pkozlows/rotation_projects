#include <iostream>
#include <fstream>
#include <armadillo>
#include <map>
#include "basis.h"
#include "scf.h"

using namespace std;

// Function to run SCF and return the converged energy
double run_scf(Basis_3D &basis, const int nelec, double rs) {
    auto [n_pw, sorted_plane_waves] = basis.generate_plane_waves();
    cout << "Number of plane waves: " << n_pw << endl;
    arma::mat kinetic_integral_matrix = basis.kinetic_integrals();
    double homo_energy = kinetic_integral_matrix.diag()(nelec / 2);
    double fermi_energy = basis.compute_fermi_energy();

    return abs(homo_energy - fermi_energy);
}

int main() {
    ofstream results_file("affirm_me_comparison_results.txt");

    // Define ranges for the number of electrons and rs values
    int nelec_values[] = {100, 200, 500, 1000, 2000, 3000, 4000, 5000, 10000, 20000}; // Adjust as needed
    double rs_values[] = {3.0, 3.5, 4.0, 4.5, 5.0}; // Values of rs to compute

    // Loop over number of electrons
    for (double rs : rs_values) {
        cout << "--------------------------------" << endl;
        cout << "Starting rs = " << rs << endl;
        results_file << "Starting rs = " << rs << endl;
        cout << "--------------------------------" << endl;

        // Loop over rs values
        for (int nelec : nelec_values) {
            cout << "--------------------------------" << endl;
            cout << "Starting calculations for nelec = " << nelec << endl;
            results_file << "Starting calculations for nelec = " << nelec << endl;
            cout << "--------------------------------" << endl;
            

            // Initialize Basis_3D object with current rs and nelec
            Basis_3D basis_3d(rs, nelec);

            // Compute the difference between the Fermi energy and the ke of HOMO
            double diff = run_scf(basis_3d, nelec, rs);
            results_file << "Difference between Fermi and HOMO energies: " << diff << endl;

        }
    }

    results_file.close();
    return 0;
}
