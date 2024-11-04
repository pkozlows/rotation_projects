# uhf_loop.py
import numpy as np
from pyscf import gto, scf, ao2mo
import sys
import os
from tqdm import tqdm  # For progress bar
import argparse

# Ensure the ipie package path is correctly set
sys.path.append('../ipie')
from ipie.systems.generic import Generic
from ipie.legacy.hamiltonians.hubbard import Hubbard
from ipie.legacy.trial_wavefunction.hubbard_uhf import HubbardUHF
import matplotlib.pyplot as plt
np.set_printoptions(suppress=True)

def run_uhf_calculation(U, output_dir):
    # 1D Hubbard Model Parameters
    nx = 8  # Number of sites in the x-direction (adjust as needed)
    ny = 1  # Single site in the y-direction for 1D
    n = int(nx * ny)
    nup = int(n / 2)
    ndown = int(n / 2)
    
    # Hubbard Model Options for 1D
    options = {
        "name": "Hubbard",
        "nup": nup,
        "ndown": ndown,
        "nx": nx,
        "ny": ny,
        "U": U,
        "t": 1.0,
        "xpbc": True,   # Periodic boundary conditions in x-direction
        "ypbc": False,  # No periodicity in y-direction for 1D
        "ueff": U
    }
    
    # Initialize System and Hamiltonian
    system = Generic(nelec=(nup, ndown))  # No need to pass options here
    ham = Hubbard(options, verbose=True)
    uhf = HubbardUHF(system, ham, {"ueff": ham.U}, verbose=True)
    
    # Set up PySCF Molecule Object
    mol = gto.M()
    mol.nelectron = nup + ndown
    mol.verbose = 0
    mol.incore_anyway = True  # Use customized Hamiltonian
    
    # Define One- and Two-Electron Integrals
    nbasis = ham.nbasis
    h1 = ham.T[0].copy()
    eri = np.zeros((nbasis, nbasis, nbasis, nbasis))
    for i in range(nbasis):
        eri[i, i, i, i] = ham.U
    
    # Setup UHF Calculation
    mf = scf.UHF(mol)
    mf.max_cycle = 100
    mf.max_memory = 10000
    mf.get_hcore = lambda *args: h1
    mf.get_ovlp = lambda *args: np.eye(nbasis)
    
    # Symmetrize Two-Electron Integrals
    eri_symm = ao2mo.restore(8, eri, nbasis)
    mf._eri = eri_symm.copy()
    mf.init_guess = '1e'
    
    # Perform UHF Calculation
    mf.kernel(uhf.G)
    
    # Optional: Perform Stability Analysis and Density Matrix Manipulation
    mf.max_cycle = 1000
    mf = mf.newton().run()
    
    mo1 = mf.stability()[0]
    dm1 = mf.make_rdm1(mo1, mf.mo_occ)
    
    # Add Random Noise to Beta Density Matrix for Stability Analysis
    dmb = dm1[1].copy()
    dmb += np.random.randn(dmb.shape[0] * dmb.shape[1]).reshape(dmb.shape)
    dm1[1] = dmb.copy()
    
    # Rerun UHF with Modified Density Matrix
    mf = mf.run(dm1)
    mf.stability()
    
    # Obtain Converged Density Matrix
    dm_converged = mf.make_rdm1(mf.mo_coeff, mf.mo_occ)
    
    # Final UHF Run with Converged Density Matrix
    mf = mf.run(dm_converged)
    
    # Extract MO Energies and Coefficients
    mo_energies = mf.mo_energy  # MO energies
    mo_coefficients = mf.mo_coeff  # MO coefficients
    
    # Save MO Energies and Coefficients to Files
    energies_filename = os.path.join(output_dir, f'mo_energies_U{U:.3f}.npy')
    coeffs_filename = os.path.join(output_dir, f'mo_coefficients_U{U:.3f}.npy')
    
    np.save(energies_filename, mo_energies)
    np.save(coeffs_filename, mo_coefficients)
    
    # Optional: Print confirmation
    print(f"Saved MO energies to {energies_filename}")
    print(f"Saved MO coefficients to {coeffs_filename}")

def main():
    parser = argparse.ArgumentParser(description="Run UHF calculations for a range of Hubbard U parameters.")
    parser.add_argument('--U_start', type=float, default=0.0, help='Starting value of Hubbard U')
    parser.add_argument('--U_end', type=float, default=0.8, help='Ending value of Hubbard U')
    parser.add_argument('--resolution', type=int, default=100, help='Number of U steps')
    parser.add_argument('--output_base_dir', type=str, default='output', help='Base directory to save output files')
    args = parser.parse_args()
    
    U_start = args.U_start
    U_end = args.U_end
    resolution = args.resolution
    output_base_dir = args.output_base_dir
    
    # Create the base output directory
    os.makedirs(output_base_dir, exist_ok=True)
    
    # Generate U values
    hubb_us = np.linspace(U_start, U_end, resolution)
    
    # Loop over U values
    for U in tqdm(hubb_us, desc="Processing U values"):
        # Define output directory for current U
        output_dir = os.path.join(output_base_dir, f'U_{U:.3f}')
        os.makedirs(output_dir, exist_ok=True)
        
        # Run UHF calculation for current U
        try:
            run_uhf_calculation(U, output_dir)
        except Exception as e:
            print(f"Error processing U={U}: {e}")

if __name__ == "__main__":
    main()
