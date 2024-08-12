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
    cout << "max_n " << max_n << endl;

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

//function to generate the list of momentum transfer vectors
pair<size_t, vector<tuple<int, int, int>>> Basis_3D::generate_momentum_transfer_vectors() {
    vector<tuple<int, int, int>> momentum_transfer_vectors;
    size_t n_mom = 0;
    //this goes the same as when we construct the plane waves but now we try up to |2 * max_n|
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
    return {n_mom, momentum_transfer_vectors};
}

arma::Mat<int> Basis_3D::make_lookup_table() {
    arma::Mat<int> lookup_table(n_pw, n_mom, arma::fill::zeros);

    for (int p = 0; p < n_pw; p++) {
        auto [px, py, pz] = plane_waves[p];

        for (int Q = 0; Q < n_mom; Q++) {
            auto [qx, qy, qz] = momentum_transfer_vectors[Q];

            // we subtract the momentum transfer vector
            int px_m_qx = px - qx;
            int py_m_qy = py - qy;
            int pz_m_qz = pz - qz;

            // Create tuple p_pm_Q
            tuple<int, int, int> p_m_Q = make_tuple(px_m_qx, py_m_qy, pz_m_qz);

            // search for p_pm_Q in plane_waves
            auto it = find(plane_waves.begin(), plane_waves.end(), p_m_Q);

            int index;
            if (it != plane_waves.end()) {
                // if found, get the index
                index = distance(plane_waves.begin(), it);
            } else {
                // if not found, set index to -1
                index = -1;
            }
            lookup_table(p, Q) = index;
        }
    }

    return lookup_table;
}


// Function to generate the kinetic integral matrix
arma::mat Basis_3D::kinetic_integrals() {
    // Create a diagonal matrix directly from the std::vector<double>
    arma::mat kinetic_integral_matrix = arma::diagmat(arma::vec(kinetic_energies));

    return kinetic_integral_matrix;
}

// Function to generate the exchange integrals; note that we just need one entry per momentum transfer vector
arma::vec Basis_3D::exchangeIntegrals() {
    arma::vec exchange(n_mom, arma::fill::zeros);

    for (int Q = 0; Q < n_mom; Q++) {
        auto [qx, qy, qz] = momentum_transfer_vectors[Q];
        double q2 = qx * qx + qy * qy + qz * qz;
        if (q2 > 1e-8) {
            exchange[Q] = (4 * M_PI) / q2;
        }
        // treat case where Q = [0, 0, 0]
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
