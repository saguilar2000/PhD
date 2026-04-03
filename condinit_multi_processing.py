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
                     return_map, return_freq, return_mapv=None, return_mapv_freq=None):

    sizex = len(Kkx)
    sizez = len(Kkz)

    MAP  = np.zeros((sizez, sizex))
    OSC  = np.zeros((sizez, sizex))
    MAPV = np.zeros((sizez, sizex, vec_size))
    OSCV = np.zeros((sizez, sizex, vec_size))

    for j, Kz in enumerate(Kkz):
        for i, Kx in enumerate(Kkx):
            w, v = gs.compute_eigenvalues(
                Kx, Kz, Am, chi,
                fx, fy, fz,
                betay_i, betaz_i,
                mn, mi, q, r0, h0, vec
            )

            idx = np.argmax(w.real)
            MAP[j, i] = w.real[idx]
            OSC[j, i] = w.imag[idx]

            if vec:
                MAPV[j, i, :] = v[idx].real
                OSCV[j, i, :] = v[idx].imag

    # --------------------------------------------------------
    # Write to shared memory
    # --------------------------------------------------------
    n_local = sizez * sizex

    start = proc_num * n_local
    end   = start + n_local

    return_map [start:end] = MAP.ravel()
    return_freq[start:end] = OSC.ravel()

    if return_mapv is not None:
        start_v = start * vec_size
        end_v   = end * vec_size

        assert end_v - start_v == MAPV.size, \
            "Shared-memory slice size mismatch for MAPV"

        return_mapv[start_v:end_v] = MAPV.ravel()
        return_mapv_freq[start_v:end_v] = OSCV.ravel()

# ============================================================
# Elsasser number ambipolar diffusion term
# ============================================================
def Am(r, W, z, chi):
    r0 = 1.0 # AU
    h0 = 0.05 * r
    rho_n0 = 1.0e-10
    gamma = 3.5e13  # cm^3 g^-1 s^-1
    G = 6.67430e-8  # cm^3 g^-1 s^-2
    M_sun = 1.989e33  # g
    r_cm = r * 1.496e13  # Convert AU to cm

    rho_n = rho_n0 * (r/r0)**(W) * np.exp(-z**2/(2*h0**2))
    Omega_K = np.sqrt(G*M_sun/r_cm**3)
    alpha = gamma * rho_n / Omega_K

    return alpha * chi

# ============================================================
# Main
# ============================================================
if __name__ == "__main__":

    par = json.loads(sys.argv[1])

    # Disk parameters
    q  = par["q"]
    r0 = par["r0"]
    h0 = par["h0"]
    z  = par["z"]
    Q  = par["Q"]
    W  = par["W"]
    n  = par["n"]
    mn = par["mn"]
    mi = par["mi"]

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
    if kxmin == 0 and kxmax == 0:
        Kkx = np.array([0.0])
        NX = 1
    else:
        Kkx = np.logspace(np.log10(kxmin), np.log10(kxmax), NX)

    if kzmin == 0 and kzmax == 0:
        Kkz = np.array([0.0])
        NZ = 1
    else:
        Kkz = np.logspace(np.log10(kzmin), np.log10(kzmax), NZ)

    # --------------------------------------------------------
    # Eigenvector setup
    # --------------------------------------------------------
    vec = True
    vec_size = 10

    # --------------------------------------------------------
    # Shared memory
    # --------------------------------------------------------
    MAP  = multiprocessing.Array('d', NZ * NX)
    OSC  = multiprocessing.Array('d', NZ * NX)
    MAPV = multiprocessing.Array('d', NZ * NX * vec_size)
    OSCV = multiprocessing.Array('d', NZ * NX * vec_size)

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
    chi_list     = [1.0e-08, 5.0e-09, 1.0e-09, 5.0e-10, 1.0e-10, 5.0e-11, 1.0e-11]
    # chi_list     = [1.0]
    # ratio_list     = [10, 1, 0.1, 0.03, 0.02, 0.01, 6e-4, 0]
    # betaz_list   = [1.0e+01, 1.0e+02, 1.0e+03, 1.0e+04, 1.0e+05, 1.0e+06]
    betaz_list   = [800.0]
    vAp = 0.1072
    vAz = 0.05

    for chi in chi_list:
        Am_ = Am(r0, W, z, chi)
        # for ratio in ratio_list:
        #         f = 1 + chi
        #         cs2 = ratio * (vAp**2 + vAz**2)
        #         betay = 2 * f * cs2 / vAp**2
        #         betaz = 2 * f * cs2 / vAz**2
        #         if cs2 == 0.0:
        #             betay = 1e-10
        #             betaz = 1e-10
        for betaz in betaz_list:
            # betay_list = [0.1*betaz, 1*betaz, 10*betaz, 100*betaz]
            betay_list = [50.0, 100.0, 500.0, 1000.0, 5000.0]
            for betay in betay_list:

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
                        MAP, OSC, MAPV, OSCV
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
                MAP_save  = np.array(MAP[:]).reshape(NZ, NX)
                OSC_save  = np.array(OSC[:]).reshape(NZ, NX)
                MAPV_save = np.array(MAPV[:]).reshape(NZ, NX, vec_size)
                OSCV_save = np.array(OSCV[:]).reshape(NZ, NX, vec_size)

                # ------------------------------------------------
                # Save
                # ------------------------------------------------
                directory = par["Output"] + "/"
                os.makedirs(directory, exist_ok=True)

                chi_str   = f"{chi:.0e}"
                # ratio_str   = f"{ratio:.0e}"
                betay_str = f"{betay:.0e}"
                betaz_str = f"{betaz:.0e}"

                # map_fname = f"MAP_{node}_cs2vA2_{ratio_str}_betay_{betay_str}_betaz_{betaz_str}"
                # osc_fname = f"OSC_{node}_cs2vA2_{ratio_str}_betay_{betay_str}_betaz_{betaz_str}"
                # mapv_fname = f"MAPV_{node}_cs2vA2_{ratio_str}_betay_{betay_str}_betaz_{betaz_str}"
                # oscv_fname = f"OSCV_{node}_cs2vA2_{ratio_str}_betay_{betay_str}_betaz_{betaz_str}"

                map_fname = f"MAP_{node}_chi_{chi_str}_betay_{betay_str}_betaz_{betaz_str}"
                osc_fname = f"OSC_{node}_chi_{chi_str}_betay_{betay_str}_betaz_{betaz_str}"
                mapv_fname = f"MAPV_{node}_chi_{chi_str}_betay_{betay_str}_betaz_{betaz_str}"
                oscv_fname = f"OSCV_{node}_chi_{chi_str}_betay_{betay_str}_betaz_{betaz_str}"

                np.savez(directory + map_fname,
                            data=MAP_save,
                            drift=np.array([fx, fy, fz]),
                            Am=Am_,
                            chi=chi,
                            betay=betay,
                            betaz=betaz,
                            # cs2=cs2
                            )
                
                np.savez(directory + osc_fname,
                            data=OSC_save,
                            drift=np.array([fx, fy, fz]),
                            Am=Am_,
                            chi=chi,
                            betay=betay,
                            betaz=betaz,
                            # cs2=cs2
                            )
                
                np.savez(directory + mapv_fname,
                            data=MAPV_save,
                            drift=np.array([fx, fy, fz]),
                            Am=Am_,
                            chi=chi,
                            betay=betay,
                            betaz=betaz,
                            # cs2=cs2
                            )
                
                np.savez(directory + oscv_fname,
                            data=OSCV_save,
                            drift=np.array([fx, fy, fz]),
                            Am=Am_,
                            chi=chi,
                            betay=betay,
                            betaz=betaz,
                            # cs2=cs2
                            )

                if f"KX_{node}.npz" not in os.listdir(directory):
                    np.savez(directory + f"KX_{node}", Kkx)
                    np.savez(directory + f"KZ_{node}", Kkz)

    print("\nDone!")
    print("Maps saved in:", directory)