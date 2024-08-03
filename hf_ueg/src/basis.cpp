#include <iostream>
#include <armadillo>
#include <vector>
#include "basis.h"

using namespace std;

// Base class constructor
Basis_3D::Basis_3D(const double &ke_cutoff, const double &rs, const int &n_elec)
    : ke_cutoff(ke_cutoff), rs(rs), n_elec(n_elec) {
}


// Function to determine the number of plane waves within the kinetic energy cutoff and compute the kinetic energy integral matrix
pair<int, vector<tuple<int, int, int>>> Basis_3D::generate_plan_waves() {
    int n_pw = 0;
    vector<pair<tuple<int, int, int>, double>> plane_wave_kinetic_pairs; // Pair of plane wave and kinetic energy

    // Define the numerical factor used to compute the kinetic energy
    double ke_factor = 7.59633120576 * pow(n_elec, -2.0 / 3.0) * pow(rs, -2.0);
    // cout << "ke_factor: " << ke_factor << endl;

    // Define the maximum value that nx, ny, nz can take
    int max_n = static_cast<int>(floor(sqrt(ke_cutoff / ke_factor)));
    // cout << "Max n: " << max_n << endl;
    // cout << "ke_cutoff: " << ke_cutoff << endl;

    for (int nx = -max_n; nx <= max_n; nx++) {
        int nx2 = nx * nx;
        double ke_nx = ke_factor * nx2;
        // cout << "ke_nx: " << ke_nx << endl;



        int max_ny = static_cast<int>(floor(sqrt((ke_cutoff - ke_nx) / ke_factor)));
        for (int ny = -max_ny; ny <= max_ny; ny++) {
            int ny2 = ny * ny;
            double ke_nx_ny = ke_nx + ke_factor * ny2;
            // cout << "ke_nx_ny: " << ke_nx_ny << endl;
            if (ke_nx_ny > ke_cutoff) continue;

            int max_nz = static_cast<int>(floor(sqrt((ke_cutoff - ke_nx_ny) / ke_factor)));
            for (int nz = -max_nz; nz <= max_nz; nz++) {
                int nz2 = nz * nz;
                double ke = ke_nx_ny + ke_factor * nz2;
                if (ke <= ke_cutoff) {
                    plane_wave_kinetic_pairs.emplace_back(make_tuple(nx, ny, nz), ke);
                    //find out this information
                    // cout << "nx: " << nx << " ny: " << ny << " nz: " << nz << " ke: " << ke << endl;
                    n_pw++;
                }
            }
        }
    }
    // cout << "---------------------" << endl;
    // cout << "Number before sorting: " << n_pw << endl;
    // cout << "---------------------" << endl;

    // Sort the plane waves based on kinetic energy
    sort(plane_wave_kinetic_pairs.begin(), plane_wave_kinetic_pairs.end(),
              [](const pair<tuple<int, int, int>, double>& a,
                 const pair<tuple<int, int, int>, double>& b) {
                  return a.second < b.second;
              });
    int n = 0;
    //find out this same information as before
    for (const auto& pair : plane_wave_kinetic_pairs) {
        auto [nx, ny, nz] = pair.first;
        double ke = pair.second;
        cout << "nx: " << nx << " ny: " << ny << " nz: " << nz << " ke: " << ke << endl;
    }
    // cout << "---------------------" << endl;
    // cout << "Number after sorting: " << n << endl;


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

arma::mat Basis_3D::make_lookup_table() {
    arma::mat lookup_table(n_pw, n_pw);
    for (int i = 0; i < n_pw; i++) {
        for (int j = 0; j < n_pw; j++) {
            auto [ix, iy, iz] = plane_waves[i];
            auto [jx, jy, jz] = plane_waves[j];
            //compute the momentum transfer vector between these waves
            int qx = jx - ix;
            int qy = jy - iy;
            int qz = jz - iz;
            tuple<int, int, int> Q = make_tuple(qx, qy, qz);
            //chuck it this new back door q is within the original list
            // Find the index of the vector Q in the list of plane waves
            auto it = std::find(plane_waves.begin(), plane_waves.end(), Q);
            if (it != plane_waves.end()) {
                // Get the index of the found vector Q
                int q_index = std::distance(plane_waves.begin(), it);

                // Set the lookup table entry to the index of Q
                lookup_table(i, j) = q_index;
            } else {
                // If not found, set a default value (e.g., -1 or some other indicator)
                lookup_table(i, j) = -1;
            }
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

// Function to generate the Coulomb integral matrix
arma::vec Basis_3D::exchangeIntegrals() {
    arma::vec exchange(n_pw, arma::fill::zeros);

    double length = pow(4.0 * M_PI * n_elec / 3.0, 1.0 / 3.0) * rs;
    double factor = ((4 * M_PI) / pow(length, 3));

    for (int Q = 0; Q < n_pw; Q++) {
        auto [qx, qy, qz] = plane_waves[Q];
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
    double fermi_energy = 0.5 * pow(3.0 * M_PI * M_PI * n, 2.0 / 3.0);
    return fermi_energy;
}
