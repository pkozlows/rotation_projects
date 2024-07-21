#include <iostream>
#include <armadillo>
#include <vector>
#include "basis.h"

using namespace std;

// Base class constructor
Basis::Basis(const double &ke_cutoff, const double &rs, const int &n_elec)
    : ke_cutoff(ke_cutoff), rs(rs), n_elec(n_elec) {
        this->ke_cutoff = ke_cutoff;
        this->rs = rs;
        this->n_elec = n_elec;
    }

// 3D Basis implementation
Basis_3D::Basis_3D(const double &ke_cutoff, const double &rs, const int &n_elec)
    : Basis(ke_cutoff, rs, n_elec) {
}


// Function to determine the number of plane waves within the kinetic energy cutoff and compute the kinetic energy integral matrix
int Basis_3D::n_plane_waves() {
    int n_pw = 0;
    std::vector<std::tuple<int, int, int>> plane_waves;
    std::vector<double> kinetic_energies; // To store kinetic energies

    // Define the numerical factor used to compute the kinetic energy
    double ke_factor = 7.5963 * pow(n_elec, -2.0/3.0) * pow(rs, -2.0);
    
    // Define the maximum value that nx can take
    int max_nx = static_cast<int>(std::floor(std::sqrt(ke_cutoff / ke_factor)));

    for (int nx = -max_nx; nx <= max_nx; nx++) {
        int nx2 = nx * nx;
        double ke_nx = ke_factor * nx2;
        if (ke_nx > ke_cutoff) {
            continue; // Skip if nx alone exceeds the cutoff
        }
        int max_ny = static_cast<int>(std::floor(std::sqrt((ke_cutoff - ke_nx) / ke_factor)));
        for (int ny = -max_ny; ny <= max_ny; ny++) {
            int ny2 = ny * ny;
            double ke_nx_ny = ke_nx + ke_factor * ny2;
            if (ke_nx_ny > ke_cutoff) {
                continue; // Skip further processing if the energy is already above the cutoff
            }
            int max_nz = static_cast<int>(std::floor(std::sqrt((ke_cutoff - ke_nx_ny) / ke_factor)));
            for (int nz = -max_nz; nz <= max_nz; nz++) {
                int nz2 = nz * nz;
                double ke = ke_nx_ny + ke_factor * nz2;
                //check if we have a valid plane wave
                if (ke <= ke_cutoff) {
                    plane_waves.emplace_back(nx, ny, nz);
                    kinetic_energies.push_back(ke); // Store the kinetic energy
                    n_pw++;
                }
            }
        }
    }

    this->n_pw = n_pw;
    this->plane_waves = plane_waves;
    this->kinetic_energies = kinetic_energies;
    return n_pw;
}

// Function to generate the kinetic integral matrix
arma::mat Basis_3D::kinetic_integrals() {
    // Create a diagonal matrix directly from the std::vector<double>
    arma::mat kinetic_integral_matrix = arma::diagmat(arma::vec(kinetic_energies));

    return kinetic_integral_matrix;
}

// Function to generate the Coulomb integral matrix
arma::mat Basis_3D::coulombIntegrals() {
    int pair_product = n_pw * n_pw;
    // Initialize a partially flattened 4-tensor of 0s to store the matrix elements
    arma::mat coulomb_integral(pair_product, pair_product, arma::fill::zeros);

    // Make the necessary loops
    for (int i = 0; i < n_pw; i++) {
        for (int j = 0; j < n_pw; j++) {
            for (int k = 0; k < n_pw; k++) {
                for (int l = 0; l < n_pw; l++) {
                    // Get momentum vectors
                    auto [qxi, qyi, qzi] = plane_waves[i];
                    auto [qxj, qyj, qzj] = plane_waves[j];
                    auto [qxk, qyk, qzk] = plane_waves[k];
                    auto [qxl, qyl, qzl] = plane_waves[l];

                    // Compute the momentum transfer vectors
                    int qx1 = qxi - qxj;
                    int qy1 = qyi - qyj;
                    int qz1 = qzi - qzj;

                    int qx2 = qxk - qxl;
                    int qy2 = qyk - qyl;
                    int qz2 = qzk - qzl;
                    
                    // Check if the momentum transfer differences are equal
                    if (qx1 == qx2 && qy1 == qy2 && qz1 == qz2) {
                        // Calculate the squared momentum transfer
                        double q_squared = pow(sqrt(qx1 * qx1 + qy1 * qy1 + qz1 * qz1), 2);

                        if (q_squared != 0) {
                            // Compute the Coulomb integral
                            // L = \left( \frac{4\pi N}{3} \right)^{1/3} r_s
                            double length = pow(4.0 * M_PI * n_elec / 3.0, 1.0 / 3.0) * rs;
                            // double factor = ((4 * M_PI) / pow(length, 3));
                            // = \frac{32\pi^4}{L^6}
                            double factor = 32 * pow(M_PI, 4) / pow(length, 6);
                            double term = factor / q_squared;


                            // Assign the computed Coulomb integral to the matrix elements
                            coulomb_integral(i * n_pw + j, k * n_pw + l) = term;
                        }
                    }
                }
            }
        }
    }

    return coulomb_integral;
}

// 2D Basis implementation
Basis_2D::Basis_2D(const double &ke_cutoff, const double &rs, const int &n_elec)
    : Basis(ke_cutoff, rs, n_elec) {
}



arma::mat Basis_2D::kinetic_integrals() {
    // Create a diagonal matrix directly from the std::vector<double>
    arma::mat kinetic_integral_matrix = arma::diagmat(arma::vec(kinetic_energies));

    return kinetic_integral_matrix;
}

int Basis_2D::n_plane_waves() {
    int n_pw = 0;
    std::vector<std::tuple<int, int>> plane_waves;
    std::vector<double> kinetic_energies; // To store kinetic energies

    // Define the numerical factor used to compute the kinetic energy = 2\pi N^{-1} r_s^{-2} \left(n_x^2 + n_y^2\right)
    double ke_factor = 2 * M_PI * pow(n_elec, -1.0) * pow(rs, -2.0);
    
    // Define the maximum value that nx can take
    int max_nx = static_cast<int>(std::floor(std::sqrt(ke_cutoff / ke_factor)));

    for (int nx = -max_nx; nx <= max_nx; nx++) {
        int nx2 = nx * nx;
        double ke_nx = ke_factor * nx2;
        if (ke_nx > ke_cutoff) {
            continue; // Skip if nx alone exceeds the cutoff
        }
        int max_ny = static_cast<int>(std::floor(std::sqrt((ke_cutoff - ke_nx) / ke_factor)));
        for (int ny = -max_ny; ny <= max_ny; ny++) {
            int ny2 = ny * ny;
            double ke_nx_ny = ke_nx + ke_factor * ny2;
            if (ke_nx_ny > ke_cutoff) {
                continue; // Skip further processing if the energy is already above the cutoff
            }
            //check if we have a valid plane wave
            if (ke_nx_ny <= ke_cutoff) {
                plane_waves.emplace_back(nx, ny);
                kinetic_energies.push_back(ke_nx_ny); // Store the kinetic energy
                n_pw++;
            }
        }
    }

    this->n_pw = n_pw;
    this->plane_waves = plane_waves;
    this->kinetic_energies = kinetic_energies;
    return n_pw;
}

arma::mat Basis_2D::coulombIntegrals() {
    int pair_product = n_pw * n_pw;
    // Initialize a partially flattened 4-tensor of 0s to store the matrix elements
    arma::mat coulomb_integral(pair_product, pair_product, arma::fill::zeros);

    // Make the necessary loops
    for (int i = 0; i < n_pw; i++) {
        for (int j = 0; j < n_pw; j++) {
            for (int k = 0; k < n_pw; k++) {
                for (int l = 0; l < n_pw; l++) {
                    // Get momentum vectors
                    auto [qxi, qyi] = plane_waves[i];
                    auto [qxj, qyj] = plane_waves[j];
                    auto [qxk, qyk] = plane_waves[k];
                    auto [qxl, qyl] = plane_waves[l];

                    // Compute the momentum transfer vectors
                    int qx1 = qxi - qxj;
                    int qy1 = qyi - qyj;

                    int qx2 = qxk - qxl;
                    int qy2 = qyk - qyl;
                    
                    // Check if the momentum transfer differences are equal
                    if (qx1 == qx2 && qy1 == qy2) {
                        // Calculate the momentum transfer
                        double q = sqrt(qx1 * qx1 + qy1 * qy1);

                        if (q != 0) {
                            // Compute the Coulomb integral
                            // L = \left( \frac{4\pi N}{3} \right)^{1/3} r_s
                            double length = pow(M_PI * n_elec, 1.0 / 2.0) * rs;
                            // double factor = ((2 * M_PI) / pow(length, 2));
                            // \frac{8\pi^3}{L^4}
                            double factor = 8 * pow(M_PI, 3) / pow(length, 4);
                            double term = factor / q;
                            // Assign the computed Coulomb integral to the matrix elements
                            coulomb_integral(i * n_pw + j, k * n_pw + l) = term;
                        }
                    }
                }
            }
        }
    }
    return coulomb_integral;
}
                        
