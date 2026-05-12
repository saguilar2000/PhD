"""
mri_analysis.py
---------------
Compute growth rates and eigenvectors for the 2-fluid MRI (2fmri) setup.

Usage:
    python mri_analysis.py --eigvec   # compute and save eigenvectors
    python mri_analysis.py --curve    # plot dispersion relation + measured rates
    python mri_analysis.py --eigvec --curve  # both
"""

import os
import struct
import argparse

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2
from numpy import polyfit, log


# ---------------------------------------------------------------------------
# Physical / global constants
# ---------------------------------------------------------------------------

R0            = 1.0
H0            = 0.05 * R0
ETA           = (H0 / R0) ** 2          # aspect-ratio squared
INV_SQRT_ETA  = 1.0 / np.sqrt(ETA)

# FARGO3D grid / time parameters (used in curve())
NY       = 64
NZ       = 64
NINTERM  = 2
DT       = 0.1
N_SNAPS  = 81   # number of snapshots per run

# Directory layout
DIR_SOLVER  = "setups/2fmri/condinit_nodrift/"
DIR_FARGO   = "setups/2fmri/condinit_nodrift/fargo/"
DIR_OUTPUTS = "outputs/2fmri/"
EIGVEC_FILE = "setups/2fmri/eigenvectors.dat"

# Zmax values that define each simulation run
Z_RUNS = [0.3974, 0.0911, 0.0274, 0.0223, 0.012, 0.0115]


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def project_eigenvector(hd_component: complex, kz_mode: float, dz: float) -> float:
    """Return the real-space projection of a complex eigenvector component.

    Evaluates  Re(hd) * cos(kz * z) - Im(hd) * sin(kz * z)  at z = dz.
    """
    return hd_component.real * np.cos(kz_mode * dz) - hd_component.imag * np.sin(kz_mode * dz)


def read_field(path: str, nz: int = NZ, ny: int = NY) -> np.ndarray:
    """Load a flat binary field file and reshape to (nz, ny)."""
    return np.fromfile(path).reshape(nz, ny)


def read_param(parfile_path: str, key: str) -> float:
    """Extract a float parameter value from a FARGO3D variables.par file."""
    with open(parfile_path) as f:
        for line in f:
            if key in line:
                return float(line.split()[1])
    raise KeyError(f"Parameter '{key}' not found in {parfile_path}")


def fit_growth_rate(time: list, amplitudes: list) -> float:
    """Fit exponential growth rate via linear regression on log(amplitude)."""
    slope, _ = polyfit(time, log(amplitudes), 1)
    return slope


# ---------------------------------------------------------------------------
# FARGO3D mode identification
# ---------------------------------------------------------------------------

def get_mode(outputdir: str) -> float:
    """
    Parse variables.par and return the normalised wavenumber k_z v_Az / Omega_0
    for the dominant mode in that run.

    Assumes mu0 = rho0 = 1.
    """
    par   = outputdir + "variables.par"
    cs    = read_param(par, "SOUNDSPEED")
    betaz = read_param(par, "BETAZ")
    om0   = read_param(par, "OMEGAFRAME")
    zmin = read_param(par, "ZMIN")
    zmax = read_param(par, "ZMAX")

    kmode = 2 * np.pi / (zmax - zmin) * cs * np.sqrt(2 / betaz) / om0
    print(f"  cs={cs}, betaz={betaz}, zmin={zmin}, zmax={zmax}  =>  k_mode = {kmode:.4f}")
    return kmode


# ---------------------------------------------------------------------------
# Eigenvector computation
# ---------------------------------------------------------------------------

def load_eigenvector_file(path: str) -> tuple:
    """
    Load a GRW_*.npz file and return relevant physical quantities.
    Returns (kz_mode, dz, kzvA_omega, omega, eigvecs_normalised, metadata_dict).
    """
    data    = np.load(path)
    betaz   = data['betaz']
    chi     = data['chi']
    mi      = data['mi']
    mn      = data['mn']
    eps     = chi * (mi / mn)
    omega   = data['data'][0][0]
    kz_tilde = data['kz']

    kz_mode    = kz_tilde / (ETA * R0)
    dz         = 2 * np.pi / kz_mode
    kzvA_omega = kz_tilde * INV_SQRT_ETA * np.sqrt(2.0 / betaz) * np.sqrt((1.0 + chi) / (1.0 + eps))

    # Load and phase-normalise eigenvectors
    raw      = np.load(path.replace("GRW", "VEC"))['data'][0, 0, :]
    eigvecs  = raw[:, 0] + 1j * raw[:, 1]
    i_max    = np.argmax(np.abs(eigvecs))
    eigvecs *= np.exp(-1j * np.angle(eigvecs[i_max])) # phase so that largest component is real and positive

    meta = dict(betaz=betaz, betay=data['betay'], Am=data['Am'],
                chi=chi, eps=eps, omega=omega,
                kz_mode=kz_mode, dz=dz, kzvA_omega=kzvA_omega)
    return eigvecs, meta


def compute_real_perturbations(eigvecs: np.ndarray, kz_mode: float, dz: float) -> dict:
    """
    Project all complex eigenvector components onto real space at z = dz.
    Returns a dict mapping component name -> real value.
    """
    names = ['dvnx', 'dvny', 'dvnz', 'dvix', 'dviy', 'dviz', 'dbx', 'dby', 'dpn', 'dpi']
    return {
        name: project_eigenvector(eigvecs[i], kz_mode, dz)
        for i, name in enumerate(names)
    }


def get_eigvec() -> None:
    """
    Read all GRW_*.npz eigenvector files, compute real-space perturbations,
    and write them to a binary file (eigenvectors.dat).
    """
    files = sorted(f for f in os.listdir(DIR_FARGO) if f.startswith("GRW"))

    if os.path.exists(EIGVEC_FILE):
        os.remove(EIGVEC_FILE)
        print(f"Removed existing {EIGVEC_FILE}")

    for fname in files:
        zmax    = float(fname.split("_kz_")[1].replace(".npz", ""))
        eigvecs, meta = load_eigenvector_file(DIR_FARGO + fname)
        pert    = compute_real_perturbations(eigvecs, meta['kz_mode'], meta['dz'])

        # Write binary row: zmax, dvnx, dvny, dvix, dviy, dbx, dby
        row = struct.pack('ddddddd',
                          zmax,
                          pert['dvnx'], pert['dvny'],
                          pert['dvix'], pert['dviy'],
                          pert['dbx'],  pert['dby'])
        with open(EIGVEC_FILE, "ab") as f:
            f.write(row)

        # Summary printout
        print(f"\nzmax={zmax:.4f}, kz_vA/Omega={meta['kzvA_omega']:.4f}, "
              f"growth_rate={meta['omega']:.4f}")
        for name, val in pert.items():
            hd = eigvecs[list(pert.keys()).index(name)]
            print(f"  {name}: hd={hd},  real={val:.4e}")

    print(f"\nWrote {len(files)} rows to {EIGVEC_FILE}")


# ---------------------------------------------------------------------------
# Growth-rate measurement from FARGO3D outputs
# ---------------------------------------------------------------------------

def measure_growth_rates(outputdir: str) -> dict:
    """
    For each snapshot in outputdir, compute the growth of By via:
      - max value  (simple proxy)
      - Fourier amplitude of the dominant mode  (cleaner)

    Returns dict with keys 'time', 'max_rate', 'fourier_rate'.
    """
    maxv   = []
    amp_k  = []
    time   = []

    for i in range(N_SNAPS):
        by        = read_field(outputdir + f"by{i}.dat") # check why other fields fail to match the curve...
        by_fft    = np.abs(fft2(by))

        time.append(i * DT * NINTERM)
        maxv.append(by.max())
        amp_k.append(by_fft.max())

    return {
        'time':         time,
        'max_rate':     fit_growth_rate(time, maxv),
        'fourier_rate': fit_growth_rate(time, amp_k),
    }


# ---------------------------------------------------------------------------
# Dispersion-relation solver data loader
# ---------------------------------------------------------------------------

def load_solver_curve() -> tuple:
    """
    Load the theoretical dispersion relation from the pre-computed solver files.
    Returns (kz_normalised, omega).
    """
    data        = np.load(DIR_SOLVER + "GRW_0_kz_0.0000.npz")
    kz_tilde    = np.load(DIR_SOLVER + "KZ_0.npz")['arr_0']
    omega       = data['data']
    betaz       = data['betaz']
    chi         = data['chi']
    eps         = chi   # simplified: eps = chi when mi/mn = 1

    kz_norm = kz_tilde * INV_SQRT_ETA * np.sqrt(2.0 / betaz) * np.sqrt((1.0 + chi) / (1.0 + eps))
    return kz_norm, omega


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_dispersion_curve(ax: plt.Axes, kz: np.ndarray, omega: np.ndarray) -> None:
    """Plot the theoretical dispersion relation on ax."""
    ax.plot(kz, omega, 'k-', label='Solver', linewidth=1.5)


def plot_measured_rates(ax: plt.Axes, runs: list) -> None:
    """
    For each run in the list, measure growth rates and overplot them
    on the dispersion relation.
    """
    for j, _ in enumerate(runs):
        outputdir = DIR_OUTPUTS + f"output{j}/"
        mode      = get_mode(outputdir)
        rates     = measure_growth_rates(outputdir)

        ax.plot(mode, rates['max_rate'],     'ko', label='Max-value method'    if j == 0 else "")
        ax.plot(mode, rates['fourier_rate'], 'ro', label='Fourier-space method' if j == 0 else "")
        print(f"Run {j}: mode={mode:.4f}, "
              f"max_rate={rates['max_rate']:.4f}, "
              f"fourier_rate={rates['fourier_rate']:.4f}")


def curve() -> None:
    """Load solver data, measure simulation growth rates, and plot everything."""
    kz_norm, omega = load_solver_curve()

    fig, ax = plt.subplots(figsize=(7, 5))
    plot_dispersion_curve(ax, kz_norm, omega)
    plot_measured_rates(ax, Z_RUNS)

    ax.set_xlim(0, 4.0)
    ax.set_ylim(0, 0.9)
    ax.set_xlabel(r"$k_z v_{Az} / \Omega_0$")
    ax.set_ylabel(r"$\mathrm{Im}(\omega / \Omega_0)$")
    ax.legend()
    plt.tight_layout()
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute growth rates and eigenvectors for the 2fmri setup."
    )
    parser.add_argument("-e", "--eigvec", action="store_true",
                        help="Compute eigenvectors and save to binary file.")
    parser.add_argument("-c", "--curve",  action="store_true",
                        help="Plot the dispersion relation with measured growth rates.")
    args = parser.parse_args()

    if not args.eigvec and not args.curve:
        parser.print_help()
        return

    if args.eigvec:
        get_eigvec()
    if args.curve:
        curve()


if __name__ == "__main__":
    main()