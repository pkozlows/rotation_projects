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
std::pair<int, arma::Mat<int>> Basis_3D::generate_plan_waves() {
    int n_pw = 0;
    arma::Mat<int> basis; // Changed to Mat<int> for integer storage
    arma::vec eigval;

    // Define the numerical factor used to compute the kinetic energy
    double L = rs * std::pow(4.0 * n_elec * M_PI / 3., 1. / 3.); // Box length
    double ke_factor = pow(2 * M_PI / L, 2);

    int max_n = int(std::ceil(std::sqrt(2.0 * ke_cutoff)));

    arma::vec ns = arma::linspace(-max_n, max_n, 2 * max_n + 1);
    arma::vec es = 0.5 * ns % ns;

    std::vector<arma::Col<int>> ksp;
    std::vector<double> spval;

    for (size_t i = 0; i < 2 * max_n + 1; i++) {
        for (size_t j = 0; j < 2 * max_n + 1; j++) {
            for (size_t k = 0; k < 2 * max_n + 1; k++) {
                double ek = es(i) + es(j) + es(k);
                if (ek <= ke_cutoff) {
                    arma::Col<int> ktmp(3);
                    ktmp(0) = int(ns(i));
                    ktmp(1) = int(ns(j));
                    ktmp(2) = int(ns(k));
                    ksp.push_back(ktmp);
                    spval.push_back(ke_factor * ke_factor * ek);
                    n_pw += 1;
                }
            }
        }
    }

    basis.set_size(3, n_pw);
    eigval.set_size(n_pw);

    for (size_t k = 0; k < n_pw; k++) {
        basis.col(k) = ksp[k];
        eigval(k) = spval[k];
    }

    arma::uvec indices = arma::sort_index(eigval);
    eigval = eigval(indices);
    basis = basis.cols(indices);

    this->n_pw = n_pw;
    this->kinetic_energies = eigval;
    this->plane_waves = basis;

    return {n_pw, basis};
}


arma::mat Basis_3D::make_lookup_table() {
    arma::mat lookup_table(n_pw, n_pw);

    for (int i = 0; i < n_pw; i++) {
        for (int j = 0; j < n_pw; j++) {
            // Get the indices of the plane waves
            arma::Col<int> pi = plane_waves.col(i); // i-th plane wave as Col<int>
            arma::Col<int> pj = plane_waves.col(j); // j-th plane wave as Col<int>
            
            // Compute the momentum transfer vector between these waves
            int qx = pi(0) - pj(0);
            int qy = pi(1) - pj(1);
            int qz = pi(2) - pj(2);
            arma::Col<int> Q = {qx, qy, qz}; // Define Q as Col<int>

            // Initialize a flag to indicate if Q was found
            bool found = false;
            int q_index = -1;

            // Search for Q in the list of plane waves
            for (size_t k = 0; k < plane_waves.n_cols; ++k) {
                arma::Col<int> pw = plane_waves.col(k);
                if (arma::all(pw == Q)) {
                    q_index = static_cast<int>(k); // Convert size_t to int
                    found = true;
                    break;
                }
            }

            // Set the lookup table entry to the index of Q if found, otherwise -1
            lookup_table(i, j) = found ? q_index : -1;
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
        arma::Col<int> q = plane_waves.col(Q);
        int qx = q(0);
        int qy = q(1);
        int qz = q(2);

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
