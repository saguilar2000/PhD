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


# Directory layout
DIR_SOLVER  = "setups/2fmri/condinit/"
DIR_FARGO   = "setups/2fmri/condinit/fargo/"
DIR_OUTPUTS = "outputs/2fmri/"
DIR_BACK    = "outputs/back_2fmri/"
DIR_MRI     = "outputs/mri/"
EIGVEC_FILE = "setups/2fmri/eigenvectors.dat"

# Zmax values that define each simulation run
Z_RUNS = [0.3974, 0.0911, 0.0274, 0.0223, 0.012, 0.0115]

# These are populated lazily by load_global_params()
NY = NZ = NINTERM = NTOT = N_SNAPS = None
DT = None

def load_global_params(output_dir) -> None:
    """Read grid/time parameters from variables.par (only needed for --curve)."""
    global NY, NZ, NINTERM, DT, NTOT, N_SNAPS
    with open(output_dir + "output0/variables.par") as f:
        for line in f:
            if "NY" in line:
                NY = int(line.split()[1])
            elif "NZ" in line:
                NZ = int(line.split()[1])
            elif "NINTERM" in line:
                NINTERM = int(line.split()[1])
            elif "DT" in line:
                DT = float(line.split()[1])
            elif "NTOT" in line:
                NTOT = int(line.split()[1])
    N_SNAPS = (NTOT // NINTERM) + 1
    # N_SNAPS = 80

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def eigvect(hd_component: complex, kz_mode: float, z: float) -> float:
    """Return the real-space projection of a complex eigenvector component.

    Evaluates  Re(hd) * cos(kz * z) - Im(hd) * sin(kz * z)  at z = dz.
    """
    # return hd_component.real * np.cos(kz_mode * z) - hd_component.imag * np.sin(kz_mode * z)
    return hd_component.real, hd_component.imag


def read_field(path: str, nz: int = None, ny: int = None) -> np.ndarray:
    """Load a flat binary field file and reshape to (nz, ny)."""
    nz = nz if nz is not None else NZ
    ny = ny if ny is not None else NY
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
    # select last 10 points
    slope, intercept = polyfit(time[-50:-10], log(amplitudes[-50:-10]), 1) # arreglar esto para solo los puntos crecientes, no el régimen no lineal
    # slope, _ = polyfit(time, log(amplitudes), 1) # arreglar esto para solo los puntos crecientes, no el régimen no lineal
    return slope, intercept


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
    # check if the par is BETAZ or BETA
    if "BETAZ" in open(par).read():
        betaz = read_param(par, "BETAZ")
    else:
        beta  = read_param(par, "BETA")
        betaz = beta  # if only BETA is given, assume it's the vertical one
    om0   = read_param(par, "OMEGAFRAME")
    zmin = read_param(par, "ZMIN")
    zmax = read_param(par, "ZMAX")

    kmode = 2 * np.pi / (zmax - zmin) * cs * np.sqrt(2 / betaz) / om0
    print(f"  cs={cs}, betaz={betaz}, zmin={zmin}, zmax={zmax}  =>  k_mode = {kmode:.4f}")
    return kmode


# ---------------------------------------------------------------------------
# Eigenvector computation
# ---------------------------------------------------------------------------

def load_eigvector_file(path: str) -> tuple:
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


def compute_perturbations(eigvecs: np.ndarray, kz_mode: float, dz: float) -> dict:
    """
    Project all complex eigenvector components onto real space at z = dz.
    Returns a dict mapping component name -> real value.
    """
    names = ['dvnx', 'dvny', 'dvnz', 'dvix', 'dviy', 'dviz', 'dbx', 'dby', 'dpn', 'dpi']
    return {
        name: eigvect(eigvecs[i], kz_mode, dz)
        for i, name in enumerate(names)
    }


def get_eigvect() -> None:
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
        eigvecs, meta = load_eigvector_file(DIR_FARGO + fname)
        pert    = compute_perturbations(eigvecs, meta['kz_mode'], meta['dz'])

        # Write binary row: zmax, dvnx, dvny, dvix, dviy, dbx, dby
        row = struct.pack('ddddddddddddddddddddd',
                          zmax,
                          pert['dvnx'][0], pert['dvnx'][1],
                          pert['dvny'][0], pert['dvny'][1],
                          pert['dvnz'][0], pert['dvnz'][1],
                          pert['dvix'][0], pert['dvix'][1],
                          pert['dviy'][0], pert['dviy'][1],
                          pert['dviz'][0], pert['dviz'][1],
                          pert['dbx'][0], pert['dbx'][1],
                          pert['dby'][0], pert['dby'][1],
                          pert['dpn'][0], pert['dpn'][1],
                          pert['dpi'][0], pert['dpi'][1]
                          )
        with open(EIGVEC_FILE, "ab") as f:
            f.write(row)

        # Summary printout
        print(f"\nzmax={zmax:.4f}, kz_vA/Omega={meta['kzvA_omega']:.4f}, "
              f"growth_rate={meta['omega']:.4f}")
        for name, val in pert.items():
            
            hd = eigvecs[list(pert.keys()).index(name)]
            print(f"  {name}: hd={hd}")

    print(f"\nWrote {len(files)} rows to {EIGVEC_FILE}")


# ---------------------------------------------------------------------------
# Growth-rate measurement from FARGO3D outputs
# ---------------------------------------------------------------------------

def measure_growth_rates(outputdir: str, field: str) -> dict:
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
        perturbation = read_field(outputdir + f"{field}{i}.dat")
        background   = perturbation.mean(axis=0)
        field_data   = perturbation - background
        field_fft    = np.abs(fft2(field_data))

        time.append(i * DT * NINTERM)
        maxv.append(field_data.max())
        amp_k.append(field_fft.max())

        # if i==0:
        #     fig_aux = plt.figure()
        #     ax_aux = fig_aux.add_subplot(111)
        #     ax_aux.imshow(field_data, origin='lower', aspect='auto')
        #     ax_aux.set_xlabel("Y")
        #     ax_aux.set_ylabel("Z")
        #     ind = np.argwhere(field_data == field_data.max())
        #     # print(ind)
        #     ax_aux.plot(ind[1], ind[0], 'ro', label='Dominant Fourier mode')
        #     ax_aux.legend()
        #     plt.savefig(f"setups/2fmri/{outputdir.split("/")[-2]}_{field}_{i}.png", dpi=300)
        #     plt.close(fig_aux)
    
    # max_rate_slope, max_rate_intercept = fit_growth_rate(time, maxv)
    # t = np.linspace(time[0], time[-1], 100)
    # ax_aux.plot(t, np.exp(max_rate_slope * t + max_rate_intercept), 'k--')
    

    return {
        'time':         time,
        'max_rate':     fit_growth_rate(time, maxv)[0],
        'fourier_rate': fit_growth_rate(time, amp_k)[0],
    }


# ---------------------------------------------------------------------------
# Dispersion-relation solver data loader
# ---------------------------------------------------------------------------

def load_solver_curve() -> tuple:
    """
    Load the theoretical dispersion relation from the pre-computed solver files.
    Returns (kz_normalised, omega).
    """
    data     = np.load(DIR_SOLVER + "GRW_0_kz_0.0000.npz")
    kz_tilde = np.load(DIR_SOLVER + "KZ_0.npz")['arr_0']
    omega    = data['data']
    betaz    = data['betaz']
    chi      = data['chi']
    eps      = chi   # simplified: eps = chi when mi/mn = 1

    kz_norm = kz_tilde * INV_SQRT_ETA * np.sqrt(2.0 / betaz) * np.sqrt((1.0 + chi) / (1.0 + eps))
    return kz_norm, omega


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_dispersion_curve(ax: plt.Axes, kz: np.ndarray, omega: np.ndarray) -> None:
    """Plot the theoretical dispersion relation on ax."""
    ax.plot(kz, omega, 'k-', label='Solver', linewidth=1.5)


def plot_measured_rates(ax: plt.Axes, runs: list, mode: int) -> None:
    """
    For each run in the list, measure growth rates and overplot them
    on the dispersion relation.
    """
    for j, _ in enumerate(runs):
        tfmri_outputdir = DIR_OUTPUTS + f"output{j}/"
        mri_outputdir   = DIR_MRI     + f"output{j}/"

        field           = "by"  # check why other fields fail to match the curve...
        tfmri_mode      = get_mode(tfmri_outputdir)
        tfmri_rates     = measure_growth_rates(tfmri_outputdir, field)
        # mri_mode        = get_mode(mri_outputdir)
        # mri_rates       = measure_growth_rates(mri_outputdir, field)

        ax.plot(tfmri_mode, tfmri_rates['max_rate'],     'o', c='tab:blue', fillstyle='full', label=f'2 fluid mri Max-value (field: {field})'    if j == 0 else "")
        # ax.plot(mri_mode, mri_rates['max_rate'], 'o', c='tab:orange', fillstyle='none', label=f'1 fluid mri Max-value (field: {field})' if j == 0 else "")

        if j==mode:
            # highlight the point corresponding to the eigenvector we computed
            ax.axvline(tfmri_mode - 0.05, color='red', linestyle='--', alpha=0.5)
            ax.axvline(tfmri_mode + 0.05, color='red', linestyle='--', alpha=0.5)
        ax.plot(tfmri_mode, tfmri_rates['fourier_rate'], 'o', c='tab:orange', fillstyle='none', label=f'Fourier-space method (field: {field})' if j == 0 else "")
        print(f"Run {j}: mode={tfmri_mode:.4f}, "
              f"max_rate={tfmri_rates['max_rate']:.4f}, "
              f"fourier_rate={tfmri_rates['fourier_rate']:.4f}, "
            #  f"mri_mode={mri_mode:.4f}, "
            #  f"mri_max_rate={mri_rates['max_rate']:.4f}"
              )

def plot_eigvect(ax: plt.Axes, mode: int) -> None:
    """
    For each run in the list, load the eigenvector from the binary file and
    plot the real-space perturbations of dvnx, dvny, dvix, dviy, dbx, dby.
    """
    fields = ["bx", "by", "vr"]
    tfmri_outputdir = DIR_OUTPUTS + f"output{mode}/"
    for field in fields:
        time = []
        maxv = []
        for j in range(N_SNAPS):
            if field == "vr":
                vx = "gasvx"
                vy = "gasvy"

                eigvec_vx = read_field(tfmri_outputdir + f"{vx}{j}.dat")
                eigvec_vy = read_field(tfmri_outputdir + f"{vy}{j}.dat")
                eigvec = np.sqrt(eigvec_vx**2 + eigvec_vy**2)
            else:
                eigvec = np.fromfile(tfmri_outputdir+f"{field}{j}.dat").reshape(NZ, NY)
            background = eigvec.mean(axis=0)
            # background = np.fromfile(tfmri_outputdir+f"{field}0.dat").reshape(NZ, NY).mean(axis=0)
            time.append(j * DT * NINTERM)
            maxv.append((eigvec - background).max())
        t = np.linspace(time[0], time[-1], 100)
        print(field)
        slope, intercept = fit_growth_rate(time, maxv)
        ax.plot(time, maxv, '-', label=f'{field}')
        color = ax.get_lines()[-1].get_color()
        ax.plot(t, np.exp(slope * t + intercept), '--', c=color)
    ax.set_xlabel("Time")
    ax.set_yscale('log')
    ax.set_ylabel("Max value")
    ax.legend()

def curve() -> None:
    """Load solver data, measure simulation growth rates, and plot everything."""
    kz_norm, omega = load_solver_curve()

    for mode, _ in enumerate(Z_RUNS):
        fig = plt.figure(figsize=(10, 5))
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)


        plot_dispersion_curve(ax1, kz_norm, omega)
        plot_measured_rates(ax1, Z_RUNS, mode)
        plot_eigvect(ax2, mode)

        ax1.set_xlim(0, 4.0)
        ax1.set_ylim(0, 0.9)
        ax1.set_xlabel(r"$k_z v_{Az} / \Omega_0$")
        ax1.set_ylabel(r"$\mathrm{Im}(\omega / \Omega_0)$")
        ax1.legend()
        fig.suptitle(f"Growth rates for 2-fluid MRI (mode {mode})")
        plt.tight_layout()
        plt.savefig(f"setups/2fmri/2fmri_growth_rates_mode{mode}.png", dpi=300)
    # plt.show()

def check_modes() -> None:
    """Utility to print the normalised kz mode for each run."""
    kz_norm, omega = load_solver_curve()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plot_dispersion_curve(ax, kz_norm, omega)

    files = sorted(f for f in os.listdir(DIR_FARGO) if f.startswith("GRW"))
    for i, fname in enumerate(files):
        zmax    = float(fname.split("_kz_")[1].replace(".npz", ""))
        eigvecs, meta = load_eigvector_file(DIR_FARGO + fname)
        if i==0:
            ax.plot(meta['kzvA_omega'], meta['omega'], 'o', c = 'tab:blue', fillstyle='none', label='Expected')
        else:
            ax.plot(meta['kzvA_omega'], meta['omega'], 'o', c = 'tab:blue', fillstyle='none')

    ax.set_xlim(0, 4.0)
    ax.set_ylim(0, 0.9)
    ax.set_xlabel(r"$k_z v_{Az} / \Omega_0$")
    ax.set_ylabel(r"$\mathrm{Im}(\omega / \Omega_0)$")
    ax.legend()

    plt.show()

def check_background(ax: plt.Axes, mode: int) -> None:
    """
    For each run in the list, load the eigenvector from the binary file and
    plot the real-space perturbations of dvnx, dvny, dvix, dviy, dbx, dby.
    """
    fields = ["bx", "by", "bz"]
    tfmri_outputdir = DIR_BACK + f"output{mode}/"

    bz = np.fromfile(tfmri_outputdir+f"bz0.dat").reshape(NZ, NY)

    # plt.imshow(bz)
    # colorbar = plt.colorbar()

    for field in fields:
        time = []
        maxv = []
        for j in range(N_SNAPS):
            if field == "vr":
                vx = "gasvx"
                vy = "gasvy"

                eigvec_vx = read_field(tfmri_outputdir + f"{vx}{j}.dat")
                eigvec_vy = read_field(tfmri_outputdir + f"{vy}{j}.dat")
                eigvec = np.sqrt(eigvec_vx**2 + eigvec_vy**2)
            else:
                eigvec = np.fromfile(tfmri_outputdir+f"{field}{j}.dat").reshape(NZ, NY)
            background = eigvec.mean(axis=0)
            # background = np.fromfile(tfmri_outputdir+f"{field}0.dat").reshape(NZ, NY).mean(axis=0)
            time.append(j * DT * NINTERM)
            maxv.append((eigvec).max())
        print(maxv[0])
        ax.plot(time, maxv, '-', label=f'{field}')
    ax.set_xlabel("Time")
    ax.set_yscale('log')
    ax.set_ylabel("Max value")
    ax.legend()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute growth rates and eigenvectors for the 2fmri setup."
    )
    parser.add_argument("-e", "--eigvec", action="store_true",
                        help="Compute eigenvectors and save to binary file.")
    parser.add_argument("-c", "--curve",  action="store_true",
                        help="Plot the dispersion relation with measured growth rates.")
    parser.add_argument("-k", "--check",  action="store_true",
                        help="Check the modes.")
    args = parser.parse_args()

    if not args.eigvec and not args.curve and not args.check:
        parser.print_help()
        return

    if args.eigvec:
        get_eigvect()
    if args.curve:
        load_global_params(DIR_OUTPUTS)  # needed for eigenvector plotting
        curve()
    if args.check:
        # check_modes()
        load_global_params(DIR_BACK)
        check_background(plt.figure().add_subplot(111), mode=0)
        plt.show()


if __name__ == "__main__":
    main()