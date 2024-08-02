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
#include <armadillo>
#include <vector>
#include <tuple>
#include <cmath>
#include <algorithm>

void generate_plane_waves(
    double ke_factor,
    double ke_cutoff,
    arma::Mat<int>& plane_waves, 
    arma::vec& kinetic_energies
) {
    int n_pw = 0;
    
    // Determine the maximum value for nx, ny, nz
    int max_n = static_cast<int>(std::ceil(std::sqrt(ke_cutoff / ke_factor)));
    
    // Vectors to store plane waves and their corresponding kinetic energies
    std::vector<arma::Col<int>> pw_vectors;
    std::vector<double> pw_energies;

    // Loop over possible values of nx, ny, nz
    for (int nx = -max_n; nx <= max_n; ++nx) {
        double ke_nx = ke_factor * nx * nx;
        int max_ny = static_cast<int>(std::ceil(std::sqrt((ke_cutoff - ke_nx) / ke_factor)));
        for (int ny = -max_ny; ny <= max_ny; ++ny) {
            double ke_nx_ny = ke_nx + ke_factor * ny * ny;
            if (ke_nx_ny > ke_cutoff) continue;
            int max_nz = static_cast<int>(std::ceil(std::sqrt((ke_cutoff - ke_nx_ny) / ke_factor)));
            for (int nz = -max_nz; nz <= max_nz; ++nz) {
                double ke = ke_nx_ny + ke_factor * nz * nz;
                if (ke <= ke_cutoff) {
                    arma::Col<int> pw(3);
                    pw(0) = nx;
                    pw(1) = ny;
                    pw(2) = nz;
                    pw_vectors.push_back(pw);
                    pw_energies.push_back(ke);
                    n_pw++;
                }
            }
        }
    }

    // Convert vectors to Armadillo types
    plane_waves.set_size(3, n_pw);
    kinetic_energies.set_size(n_pw);

    // Copy data into Armadillo matrices
    #pragma omp parallel for
    for (size_t k = 0; k < n_pw; ++k) {
        plane_waves.col(k) = pw_vectors[k];
        kinetic_energies(k) = pw_energies[k];
    }

    // Sort based on kinetic energy
    arma::uvec indices = arma::stable_sort_index(kinetic_energies, "ascend");
    kinetic_energies = kinetic_energies(indices);
    plane_waves = plane_waves.cols(indices);
}
1

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
    // compute energy in terms of the Wigner-Seitz radius r_s: 1.84158427617643/r_s**2
    double fermi_energy = 1.84158427617643 / pow(rs, 2.0);
    return fermi_energy;
}
