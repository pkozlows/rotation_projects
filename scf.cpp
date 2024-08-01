#include <iostream>
#include <armadillo>
#include <vector>
#include <unordered_map>
#include <tuple>
#include <cassert>
#include "scf.h"
#include "matrix_utils.h"

using namespace std;


Scf::Scf(const arma::mat& kinetic_integral, const arma::vec& coulombIntegral, const int& nelec, const int& npws, const vector<tuple<int, int, int>>& plane_waves, const arma::mat& lookup_table, const double& madeleung_constant) {
    kinetic = kinetic_integral;
    exchange = coulombIntegral;

    this->nelec = nelec;
    this->n_pw = npws;
    this->plane_waves = plane_waves;
    this->lookup_table = lookup_table;
    this->madeleung_constant = madeleung_constant;

    
}


