#ifndef MATRIX_UTILS_H
#define MATRIX_UTILS_H

#include <armadillo>
#include <string>

// Function to print the matrix in a controlled format
void print_matrix(const arma::mat& matrix);

// Function to save the matrix to a file
void save_matrix_to_file(const arma::mat& matrix, const std::string& filename);

// Function to check if a matrix is symmetric
bool is_symmetric(const arma::mat& matrix, double tolerance = 1e-12);

#endif // MATRIX_UTILS_H
