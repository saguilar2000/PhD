#!/usr/bin/env python3
import os
import numpy as np
import multiprocessing
import solver as gs
import json
import sys

# ============================================================
# Parallel worker
# ============================================================
def parallel_process(proc_num, Kkx, Kkz, Am, chi,
                     fx, fy, fz, betay_i, betaz_i,
                     mn, mi, q, r0, h0,
                     vec, vec_size,
                     return_grw, return_freq, return_vec=None):

    sizex = len(Kkx)
    sizez = len(Kkz)

    GRW  = np.zeros((sizez, sizex))
    OSC  = np.zeros((sizez, sizex))
    VEC = np.zeros((sizez, sizex, vec_size), dtype='complex128')

    for j, Kz in enumerate(Kkz):
        for i, Kx in enumerate(Kkx):
            w, v = gs.compute_eigenvalues(
                Kx, Kz, Am, chi,
                fx, fy, fz,
                betay_i, betaz_i,
                mn, mi, q, r0, h0, vec
            )

            idx = np.argmax(w.real)
            GRW[j, i] = w.real[idx]
            OSC[j, i] = -w.imag[idx]

            if vec:
                VEC[j, i, :] = v[:,idx]

    # --------------------------------------------------------
    # Write to shared memory
    # --------------------------------------------------------
    n_local = sizez * sizex

    start = proc_num * n_local
    end   = start + n_local

    return_grw [start:end] = GRW.ravel()
    return_freq[start:end] = OSC.ravel()

    if return_vec is not None:
        vec_flat = VEC.view(np.float64).ravel()
        start_v = start * vec_size * 2
        end_v   = start_v + vec_flat.size

        if end_v > len(return_vec):
            raise IndexError(f"Proc {proc_num}: Index out of bounds of return_vec")

        return_vec[start_v:end_v] = vec_flat

# ============================================================
# Main
# ============================================================
def main():
    # --------------------------------------------------------
    # Input parameters
    # --------------------------------------------------------
    par = json.loads(sys.argv[1])

    # Disk parameters
    q   = par["q"]
    r0  = par["r0"]
    h0  = par["h0"]
    z   = par["z"]
    Q   = par["Q"]
    W   = par["W"]
    n   = par["n"]
    mn  = par["mn"]
    mi  = par["mi"]
    eta = h0**2 / r0**2

    # Wavenumber domain
    kxmin = par["KXmin"]
    kxmax = par["KXmax"]
    NX    = int(par["NX"])
    kzmin = par["KZmin"]
    kzmax = par["KZmax"]
    NZ    = int(par["NZ"])
    drift = par["drift"]

    # Parallel parameters
    N_nodes       = par["N_nodes"]
    node          = par["node"]
    task_per_node = par["task_per_node"]

    # --------------------------------------------------------
    # Build k-space
    # --------------------------------------------------------
    if kxmin == kxmax:
        Kkx = np.array([kxmax])
        NX = 1
    else:
        Kkx = np.logspace(np.log10(kxmin), np.log10(kxmax), NX)

    if kzmin == kzmax:
        Kkz = np.array([kzmax])
        NZ = 1
    else:
        Kkz = np.logspace(np.log10(kzmin), np.log10(kzmax), NZ)

    # --------------------------------------------------------
    # Eigenvector setup
    # --------------------------------------------------------
    vec = True
    vec_size = gs.MATRIX_ORDER

    # --------------------------------------------------------
    # Shared memory
    # --------------------------------------------------------
    GRW  = multiprocessing.Array('d', NZ * NX)
    OSC  = multiprocessing.Array('d', NZ * NX)
    VEC  = multiprocessing.Array('d', NZ * NX * vec_size * 2)  # Complex numbers stored as pairs of doubles

    # --------------------------------------------------------
    # Domain decomposition
    # --------------------------------------------------------
    actual_tasks = min(NZ, task_per_node)
    division_kz = max(1, NZ//actual_tasks)

    if NZ>1:
        assert NZ % task_per_node == 0, "NZ must be divisible by task_per_node"

    # --------------------------------------------------------
    # Parameter loops
    # --------------------------------------------------------
    # chi   = 1.0e-12
    chi   = 1.0
    betaz = 500.0
    betay = 500.0e+09

    eps = chi * (mi / mn)
    # Am_ = gs.Am(r0, W, z, eps)
    Am_ = 1.0e+10

    processes = []
    
    for iz in range(actual_tasks):

        if drift == "nodrift":
            fx, fy, fz = 0.0, 0.0, 0.0
        else:
            fx, fy, fz = gs.compute_drift(
                h0, r0, z, Am_, n, betay, betaz, chi
            )
        
        start_idx = iz * division_kz
        end_idx = (iz + 1) * division_kz

        if iz == actual_tasks - 1:
            end_idx = NZ
        
        kz_node = Kkz[start_idx : end_idx]
        
        if len(kz_node) == 0:
            continue
        args = (
            iz, Kkx, kz_node,
            Am_, chi,
            fx, fy, fz,
            betay, betaz,
            mn, mi, q, r0, h0,
            vec, vec_size,
            GRW, OSC, VEC
        )

        proc = multiprocessing.Process(
            target=parallel_process,
            args=args
        )

        processes.append(proc)
        proc.start()

    for p in processes:
        p.join()

    # ------------------------------------------------
    # Reshape output
    # ------------------------------------------------
    GRW_save  = np.array(GRW[:]).reshape(NZ, NX)
    OSC_save  = np.array(OSC[:]).reshape(NZ, NX)
    VEC_save = np.array(VEC[:]).reshape(NZ, NX, vec_size, 2)

    # ------------------------------------------------
    # Save
    # ------------------------------------------------
    directory = par["Output"] + "/"
    os.makedirs(directory, exist_ok=True)

    chi_str   = f"{chi:.0e}"
    betay_str = f"{betay:.0e}"
    betaz_str = f"{betaz:.0e}"

    zmax = 2*np.pi / (kzmax / (eta * r0))

    grw_fname = f"GRW_{node}_kz_{zmax:.4f}"
    osc_fname = f"OSC_{node}_kz_{zmax:.4f}"
    vec_fname = f"VEC_{node}_kz_{zmax:.4f}"

    np.savez(directory + grw_fname,
                data=GRW_save,
                drift=np.array([fx, fy, fz]),
                mi=mi,
                mn=mn,
                Am=Am_,
                chi=chi,
                betay=betay,
                betaz=betaz,
                kz=Kkz[0],
                kx=Kkx[0]
                )
    
    np.savez(directory + osc_fname,
                data=OSC_save,
                drift=np.array([fx, fy, fz]),
                mi=mi,
                mn=mn,
                Am=Am_,
                chi=chi,
                betay=betay,
                betaz=betaz,
                kz=Kkz[0],
                kx=Kkx[0]
                )
    
    np.savez(directory + vec_fname,
                data=VEC_save,
                drift=np.array([fx, fy, fz]),
                mi=mi,
                mn=mn,
                Am=Am_,
                chi=chi,
                betay=betay,
                betaz=betaz,
                kz=Kkz[0],
                kx=Kkx[0]
                )
    
    np.savez(directory + f"KX_{node}", Kkx)
    np.savez(directory + f"KZ_{node}", Kkz)

    print("\nDone!")
    print("Maps saved in:", directory)

    gs.quote()

if __name__ == "__main__":
    main()