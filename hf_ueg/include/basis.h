#ifndef BASIS_H
#define BASIS_H
#include <armadillo>
#include <vector>
#include <tuple>


using namespace std;

class Basis {
public:
    Basis(const double &ke_cutoff, const double &rs, const int &n_elec);
    virtual ~Basis() = default;
    virtual int n_plane_waves() = 0;
    virtual arma::mat kinetic_integrals() = 0;
    virtual arma::mat coulombIntegrals() = 0;

protected:
    double ke_cutoff;
    double rs;
    int n_elec;
    int n_pw;
};

class Basis_3D : public Basis {
public:
    Basis_3D(const double &ke_cutoff, const double &rs, const int &n_elec);
    int n_plane_waves() override;
    arma::mat kinetic_integrals() override;
    arma::mat coulombIntegrals() override;

protected:
    vector<tuple<int, int, int>> plane_waves;
    vector<double> kinetic_energies;
};

class Basis_2D : public Basis {
public:
    Basis_2D(const double &ke_cutoff, const double &rs, const int &n_elec);
    int n_plane_waves() override;
    arma::mat kinetic_integrals() override;
    arma::mat coulombIntegrals() override;

protected:
    vector<tuple<int, int>> plane_waves;
    vector<double> kinetic_energies;

};

#endif