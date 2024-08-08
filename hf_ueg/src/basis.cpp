#include <iostream>
#include <armadillo>
#include <vector>
#include "basis.h"
#include "matrix_utils.h"

using namespace std;

// Base class constructor
Basis_3D::Basis_3D(const double &rs, const int &n_elec)
    : rs(rs), n_elec(n_elec) {
}


// Function to determine the number of plane waves within the kinetic energy cutoff and compute the kinetic energy integral matrix
pair<int, vector<tuple<int, int, int>>> Basis_3D::generate_plan_waves() {
    int n_pw = 0;
    vector<pair<tuple<int, int, int>, double>> plane_wave_kinetic_pairs; // Pair of plane wave and kinetic energy

    //compute the kinetic Autry cutoff based off of the number of electrons and the Wigner-Seitz radius
    double ke_cutoff = 15*pow(rs, -2) * pow(n_elec, -2.0 / 3.0);
    double length = pow(4.0 * M_PI * n_elec / 3.0, 1.0 / 3.0) * rs;
    double constant = pow(2 * M_PI / length, 2) / 2;

    // Define the maximum value that nx, ny, nz can take
    int max_n = static_cast<int>(floor(sqrt(ke_cutoff / constant)));
    // cout << "Max n: " << max_n << endl;
    // cout << "ke_cutoff: " << ke_cutoff << endl;
    this->max_n = max_n;

    for (int nx = -max_n; nx <= max_n; nx++) {
        int nx2 = nx * nx;
        double ke_nx = constant * nx2;
        // cout << "ke_nx: " << ke_nx << endl;



        int max_ny = static_cast<int>(floor(sqrt((ke_cutoff - ke_nx) / constant)));
        for (int ny = -max_ny; ny <= max_ny; ny++) {
            int ny2 = ny * ny;
            double ke_nx_ny = ke_nx + constant * ny2;
            // cout << "ke_nx_ny: " << ke_nx_ny << endl;
            if (ke_nx_ny > ke_cutoff) continue;

            int max_nz = static_cast<int>(floor(sqrt((ke_cutoff - ke_nx_ny) / constant)));
            for (int nz = -max_nz; nz <= max_nz; nz++) {
                int nz2 = nz * nz;
                double ke = ke_nx_ny + constant * nz2;
                if (ke <= ke_cutoff) {
                    plane_wave_kinetic_pairs.emplace_back(make_tuple(nx, ny, nz), ke);
                    //find out this information
                    // cout << "nx: " << nx << " ny: " << ny << " nz: " << nz << " ke: " << ke << endl;
                    n_pw++;
                }
            }
        }
    }
    cout << "Number of plain waves " << n_pw << endl;
    // cout << "---------------------" << endl;
    // cout << "Number before sorting: " << n_pw << endl;
    // cout << "---------------------" << endl;

    // Sort the plane waves based on kinetic energy
    sort(plane_wave_kinetic_pairs.begin(), plane_wave_kinetic_pairs.end(),
              [](const pair<tuple<int, int, int>, double>& a,
                 const pair<tuple<int, int, int>, double>& b) {
                  return a.second < b.second;
              });


    // Separate the sorted plane waves and kinetic energies
    vector<tuple<int, int, int>> sorted_plane_waves;
    vector<double> sorted_kinetic_energies;
    for (const auto& pair : plane_wave_kinetic_pairs) {
        sorted_plane_waves.push_back(pair.first);
        sorted_kinetic_energies.push_back(pair.second);
    }
    this->n_pw = n_pw;
    this->kinetic_energies = sorted_kinetic_energies;
    this->plane_waves = sorted_plane_waves;

    return {n_pw, sorted_plane_waves};
}

//function to generate the list of momentum transfer effectors
void Basis_3D::generate_momentum_transfer_vectors() {
    vector<tuple<int, int, int>> momentum_transfer_vectors;
    size_t n_mom = 0;
    for (int i = -2 * max_n; i <= 2 * max_n; i++) {
        for (int j = -2 * max_n; j <= 2 * max_n; j++) {
            for (int k = -2 * max_n; k <= 2 * max_n; k++) {
                momentum_transfer_vectors.push_back(make_tuple(i, j, k));
                n_mom++;
            }
        }
    }
    this->n_mom = n_mom;
    this->momentum_transfer_vectors = momentum_transfer_vectors;
    return;
}

arma::mat Basis_3D::make_lookup_table() {
    arma::mat lookup_table(n_pw, n_mom, arma::fill::zeros);

    for (int p = 0; p < n_pw; p++) {
        auto [px, py, pz] = plane_waves[p];

        for (int Q = 0; Q < n_mom; Q++) {
            auto [qx, qy, qz] = momentum_transfer_vectors[Q];
            int p_minus_q_x = px - qx;
            int p_minus_q_y = py - qy;
            int p_minus_q_z = pz - qz;

            // Create tuple p_minus_q
            tuple<int, int, int> p_minus_q = make_tuple(p_minus_q_x, p_minus_q_y, p_minus_q_z);

            // Find p_minus_q in plane_waves
            auto it = find(plane_waves.begin(), plane_waves.end(), p_minus_q);

            if (it != plane_waves.end()) {
                // Calculate index of p_minus_q in plane_waves
                int index = distance(plane_waves.begin(), it);
                lookup_table(p, Q) = index;
            } else {
                lookup_table(p, Q) = -1;
            }
        }
    }
    print_matrix(lookup_table);

    return lookup_table;
}


// Function to generate the kinetic integral matrix
arma::mat Basis_3D::kinetic_integrals() {
    // Create a diagonal matrix directly from the std::vector<double>
    arma::mat kinetic_integral_matrix = arma::diagmat(arma::vec(kinetic_energies));

    return kinetic_integral_matrix;
}

// Function to generate the Coulomb integral matrix
arma::vec Basis_3D::exchangeIntegrals() {
    arma::vec exchange(2*n_pw, arma::fill::zeros);
    double length = pow(4.0 * M_PI * n_elec / 3.0, 1.0 / 3.0) * rs;
    double factor = ((4 * M_PI) / pow(length, 3));

    for (int Q = 0; Q < 2*n_pw; Q++) {
        auto [qx, qy, qz] = plane_waves[Q%n_pw];
        //check if the value of Q is > n_pw - 1 to determine if it is a double momentum transfer and then we multiply by 2
        if (Q >= n_pw) {
            qx *= 2;
            qy *= 2;
            qz *= 2;
        }
        double q2 = qx * qx + qy * qy + qz * qz;
        if (q2 > 1e-8) {
            exchange[Q] = factor / q2;
        }
        else {
            exchange[Q] = 0.0;
        }
    }
    return exchange;
}

double Basis_3D::compute_madeleung_constant() {
    // E_M \approx-2.837297 \times\left(\frac{3}{4 \pi}\right)^{1 / 3} N^{2 / 3} r_\pi^{-1}
    double madeleung_constant = -2.837297 * pow(3.0 / (4.0 * M_PI), 1.0 / 3.0) * pow(n_elec, 2.0 / 3.0) * pow(rs, -1.0);
    return madeleung_constant;
}

double Basis_3D::compute_fermi_energy() {
    // # Express the electron density n in terms of the Wigner-Seitz radius r_s: n_expr = 3 / (4 * sp.pi * r_s**3)
    double n = 3.0 / (4.0 * M_PI * pow(rs, 3));
    double fermi_energy = 0.5 * pow(3.0 * pow(M_PI, 2) * n, 2.0 / 3.0);
    return fermi_energy;
}
