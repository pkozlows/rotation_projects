#include <iostream>
#include <armadillo>
#include <vector>
#include <cassert>
#include "basis.h"
#include "matrix_utils.h"
#include <utility>

using namespace std;

Basis::Basis(const float &rs, const size_t &n_elec)
    : rs(rs), n_elec(n_elec) {
    generate_basis(); // Generate the basis on initialization
}

void Basis::generate_basis() {
    generate_plan_waves();
    generate_momentum_transfer_vectors();
    generate_momentum_lookup_table();
    generate_pw_lookup_table();
    compute_kinetic_integrals();
    compute_interaction_integrals();
    compute_madeleung_constant();
}

void Basis::generate_plan_waves() {
    size_t n_pw = 0;
    vector<arma::Col<int>> plane_waves;
    vector<double> kinetic_energies;
    double length = pow(4.0 * M_PI * n_elec / 3.0, 1.0 / 3.0) * rs;
    this->volume = pow(length, 3);
    double unit = pow(2 * M_PI / length, 2) / 2;

    double ke_cutoff = 2 * unit;
    int max_n = static_cast<int>(floor(sqrt(ke_cutoff / unit)));
    this->max_n = max_n;

    for (int nx = -max_n; nx <= max_n; nx++) {
        size_t nx2 = nx * nx;
        double ke_nx = unit * nx2;
        int max_ny = static_cast<int>(floor(sqrt((ke_cutoff - ke_nx) / unit)));
        for (int ny = -max_ny; ny <= max_ny; ny++) {
            size_t ny2 = ny * ny;
            double ke_nx_ny = ke_nx + unit * ny2;
            if (ke_nx_ny > ke_cutoff) continue;
            int max_nz = static_cast<int>(floor(sqrt((ke_cutoff - ke_nx_ny) / unit)));
            for (int nz = -max_nz; nz <= max_nz; nz++) {
                size_t nz2 = nz * nz;
                double pw_ke = ke_nx_ny + unit * nz2;
                if (pw_ke <= ke_cutoff) {
                    plane_waves.push_back(arma::Col<int>{nx, ny, nz});
                    kinetic_energies.push_back(pw_ke);
                    n_pw++;
                }
            }
        }
    }

    arma::Mat<int> basis(3, n_pw);
    arma::vec eigval(n_pw);

    #pragma omp parallel for
    for (size_t k = 0; k < n_pw; k++) {
        basis.col(k) = plane_waves[k];
        eigval(k) = kinetic_energies[k];
    }

    arma::uvec indices = arma::stable_sort_index(eigval, "ascend");
    basis = basis.cols(indices);
    eigval = eigval(indices);

    this->n_pw = n_pw;
    this->kinetic_energies = eigval;
    this->plane_waves = basis;
}

void Basis::generate_momentum_transfer_vectors() {
    arma::Col<int> max_pw = plane_waves.col(n_pw - 1);
    double fermi_radius_sq = arma::dot(max_pw, max_pw);

    vector<arma::Col<int>> momentum_transfer_vectors;
    size_t n_mom = 0;

    for (int i = -2 * max_n; i <= 2 * max_n; ++i) {
        int i2 = i * i;
        for (int j = -2 * max_n; j <= 2 * max_n; ++j) {
            int j2 = j * j;
            for (int k = -2 * max_n; k <= 2 * max_n; ++k) {
                int k2 = k * k;
                if (i2 + j2 + k2 <= 4 * fermi_radius_sq) {
                    momentum_transfer_vectors.push_back(arma::Col<int>{i, j, k});
                    ++n_mom;
                }
            }
        }
    }

    arma::Mat<int> momentum_transfer_vectors_mat(3, n_mom);
    #pragma omp parallel for
    for (size_t k = 0; k < n_mom; ++k) {
        momentum_transfer_vectors_mat.col(k) = momentum_transfer_vectors[k];
    }

    this->n_mom = n_mom;
    this->momentum_transfer_vectors = momentum_transfer_vectors_mat;
}

void Basis::generate_momentum_lookup_table() {
    arma::Mat<int> momentum_lookup_table(n_pw, n_mom, arma::fill::zeros);

    for (size_t p = 0; p < n_pw; p++) {
        int px = plane_waves(0, p);
        int py = plane_waves(1, p);
        int pz = plane_waves(2, p);

        for (size_t Q = 0; Q < n_mom; Q++) {
            int Qx = momentum_transfer_vectors(0, Q);
            int Qy = momentum_transfer_vectors(1, Q);
            int Qz = momentum_transfer_vectors(2, Q);

            int index = -1;

            for (int i = 0; i < n_pw; ++i) {
                if ((plane_waves(0, i) == (px - Qx)) && (plane_waves(1, i) == (py - Qy)) && (plane_waves(2, i) == (pz - Qz))) {
                    index = i;
                    break;
                }
            }

            momentum_lookup_table(p, Q) = index;
        }
    }

    this->momentum_lookup_table = momentum_lookup_table;
}



void Basis::generate_pw_lookup_table() {
    arma::Mat<int> pw_lookup_table(n_pw, n_pw, arma::fill::zeros);

    for (size_t p = 0; p < n_pw; p++) {
        int px = plane_waves(0, p);
        int py = plane_waves(1, p);
        int pz = plane_waves(2, p);

        for (size_t q = 0; q < n_pw; q++) {
            int qx = plane_waves(0, q);
            int qy = plane_waves(1, q);
            int qz = plane_waves(2, q);

            int index = -1;  // Default to -1 (not found)

            #pragma omp parallel for
            for (int i = 0; i < n_mom; ++i) {
                if (momentum_transfer_vectors(0, i) == px - qx && momentum_transfer_vectors(1, i) == py - qy && momentum_transfer_vectors(2, i) == pz - qz) {
                    index = i;
                    break;
                }
            }

            assert(index != -1);  // Ensure something was found
            pw_lookup_table(p, q) = index;
        }
    }

    this->pw_lookup_table = pw_lookup_table;
}







// Function to generate the kinetic integral matrix
void Basis::compute_kinetic_integrals() {
    // Create a diagonal matrix directly from the std::vector<double>
    arma::mat kinetic_integral_matrix = arma::diagmat(kinetic_energies);

    this->kinetic_integrals = kinetic_integral_matrix;
}

// Function to generate the exchange integrals; note that we just need one entry per momentum transfer vector
void Basis::compute_interaction_integrals() {
    arma::vec exchange(n_mom, arma::fill::zeros);
    double length = pow(4.0 * M_PI * n_elec / 3.0, 1.0 / 3.0) * rs;
    double constant = pow(2 * M_PI / length, 2);

    for (size_t Q = 0; Q < n_mom; Q++) {
        
        // Extract momentum transfer vector components
        int qx = momentum_transfer_vectors(0, Q);
        int qy = momentum_transfer_vectors(1, Q);
        int qz = momentum_transfer_vectors(2, Q);

        double q2 = qx * qx + qy * qy + qz * qz;
        q2 *= constant;
        if (q2 > 1e-8) {
            exchange[Q] = (4 * M_PI) / q2;
        }
        // treat case where Q = [0, 0, 0]
        else {
            exchange[Q] = 0.0;
        }
    }
    this->interaction_integrals = exchange;
}



void Basis::compute_madeleung_constant() {
    // E_M \approx-2.837297 \times\left(\frac{3}{4 \pi}\right)^{1 / 3} N^{2 / 3} r_\pi^{-1}
    double madeleung_constant = -2.837297 * pow(3.0 / (4.0 * M_PI), 1.0 / 3.0) * pow(n_elec, 2.0 / 3.0) * pow(rs, -1.0);
    this->madeleung_constant = madeleung_constant;
}


