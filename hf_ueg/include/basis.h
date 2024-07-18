#ifndef BASIS_H
#define BASIS_H
#include <armadillo>
#include <vector>
#include <tuple>


using namespace std;

class Basis {
    public:
        //make a constructor that takes in the kinetic energy cutoff the Wigner-Seitz radius, and the number of electrons
        Basis(const double &, const double &, const int &);
        //make a function that determines the number of plane waves within the kinetic industry cutoff
        int n_plane_waves();
        //make a function that generates an initial guess for the density matrix
        arma::mat generate_initial_guess();
        //make a function that generates the kinetic integral matrix
        arma::mat kinetic_integrals();
        //make a function that generates the coulomb integral matrix
        arma::mat coulombIntegrals();
    private:
        double ke_cutoff;
        double rs;
        int n_elec;
        int n_pw;
        vector<tuple<int, int, int>> plane_waves;
        vector<double> kinetic_energies;
};

#endif