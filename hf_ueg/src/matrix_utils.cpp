#include "matrix_utils.h"
#include <iomanip>
#include <fstream>
#include <iostream>

void print_matrix(const arma::mat& matrix) {
    for (size_t i = 0; i < matrix.n_rows; ++i) {
        for (size_t j = 0; j < matrix.n_cols; ++j) {
            std::cout << std::setprecision(4) << matrix(i, j) << " ";
        }
        std::cout << std::endl;
    }
}

void save_matrix_to_file(const arma::mat& matrix, const std::string& filename) {
    std::ofstream file(filename);
    if (file.is_open()) {
        for (size_t i = 0; i < matrix.n_rows; ++i) {
            for (size_t j = 0; j < matrix.n_cols; ++j) {
                file << matrix(i, j) << " ";
            }
            file << std::endl;
        }
        file.close();
    } else {
        std::cerr << "Unable to open file: " << filename << std::endl;
    }
}

bool is_symmetric(const arma::mat& matrix, double tolerance) {
    return arma::approx_equal(matrix, matrix.t(), "absdiff", tolerance);
}
