"""_summary_
"""
import sys
import os
import numpy as np
import json

# args
N_nodes = int(sys.argv[1])   # Number of nodes
node    = int(sys.argv[2])   # Specific node number (0 to N_nodes-1)
drift   = str(sys.argv[3])
z_scale = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0

# Disk parameters
q  = 1.5
r0 = 1.0
h0 = 0.05 * r0
z  = z_scale * h0
Q  = -0.375  # -3/8
W  = -0.75   # -3/4
n  = 2 * Q + W
mn = 1
mi = 1
eta = h0**2 / r0**2

# task_per_node (usually the number of processors available in the selected node)
task_per_node = 32
zmin = 0.0

def build_and_run(kzmin, kzmax,outputdir):
    """Set up parameters and launch the solver for a given kz interval."""
    min_res = 500  # Minimum resolution for the solver
    kxmin   = 0.0
    kxmax   = 0.0

    if kxmin == kxmax:
        NX = 1
    else:
        NX = ((min_res + task_per_node - 1) // task_per_node) * task_per_node

    if kzmin == kzmax:
        NZ = 1
    else:
        NZ = ((min_res + task_per_node - 1) // task_per_node) * task_per_node

    # Domain decomposition for KX
    if kxmin == kxmax:
        kxmin_local = kxmin
        kxmax_local = kxmax
        nx_local = 1
    else:
        Kx = np.logspace(np.log10(kxmin), np.log10(kxmax), NX)
        nx_local = NX / N_nodes
        kxmin_local = Kx[int(nx_local * node)]
        kxmax_local = Kx[int(nx_local * (node + 1)) - 1]

    # Input parameters dictionary
    input_parameters = {
        "KXmin"  : kxmin_local, "KXmax" : kxmax_local, "KZmin"  : kzmin  , "KZmax" : kzmax ,
        "NX"     : nx_local   , "NZ"    : NZ          , "q"      : q      , "r0"    : r0    ,
        "h0"     : h0         , "z"     : z           , "Q"      : Q      , "W"     : W     ,
        "mn"     : mn         , "mi"    : mi          , "n"      : n      , "Output": outputdir,
        "N_nodes": N_nodes    , "node"  : node        , "drift"  : drift  , "task_per_node": task_per_node
    }

    dump = json.dumps(input_parameters)
    print(f"--- Running {drift} at z={z_scale}H | kzmin={kzmin:.4e}, kzmax={kzmax:.4e} | Res: {NX}x{NZ} ---")
    os.system("OMP_NUM_THREADS=1 python3 multi_processing.py '{:s}'".format(dump))


def do_list():
    # --- List mode: loop over zmax values, one kz mode per entry ---
    print("Mode: LIST — looping over individual kz modes derived from zmax values.")

    # Directory for data
    outputdir = f"condinit/fargo"
    os.makedirs(outputdir, exist_ok=True)

    zmax_ = [0.3974, 0.0911, 0.0274, 0.0223, 0.012, 0.0115]

    for i, zmax in enumerate(zmax_):
        print(f"\n[{i+1}/{len(zmax_)}] zmax = {zmax:.4f}H")
        kz       = 2.0 * np.pi / (zmax - zmin)
        kzetar0  = kz * eta * r0
        kzmin    = kzetar0   # single mode: kzmin == kzmax
        kzmax    = kzetar0
        build_and_run(kzmin, kzmax, outputdir)

def do_nolist():
    # --- Range mode: single run over the full fixed kz range ---
    print("Mode: RANGE — running over fixed kz range [1e-2, 1e+4].")

    # Directory for data
    outputdir = f"condinit"
    os.makedirs(outputdir, exist_ok=True)

    kzmin = 1.0e-02
    kzmax = 1.0e+04
    build_and_run(kzmin, kzmax, outputdir)

def main():
    # Check input arguments
    if len(sys.argv) < 4:
        print(" ")
        print("Run the script as:")
        print("python3 parallel_solver.py <# nodes> <specific node (int [0, #nodes-1])> <drift (str 'drift' or 'nodrift')> <z scale (float, optional, default=0.0)>")
        print("========================================================================")
        print(" ")
        exit()

    do_list()
    do_nolist()

if __name__ == "__main__":
    main()