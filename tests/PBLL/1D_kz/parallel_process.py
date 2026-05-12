"""_summary_
"""

import sys
import os
import numpy as np
import json

# Check input arguments
if len(sys.argv) < 5:
    print(" ")
    print("Run the script as:")
    print("python3 parallel_solver.py <# nodes> <specific node (int [0, #nodes-1])> <drift (str 'drift' or 'nodrift')> <z scale (float, optional, default=0.0)>")
    print("========================================================================")
    print(" ")
    exit()

# args
N_nodes = int(sys.argv[1]) # Number of nodes
node    = int(sys.argv[2]) # Specific node number (0 to N_nodes-1)
drift   = str(sys.argv[3])
z_scale = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0

# Disk parameters
q  = 1.5
r0 = 1.0
h0 = 0.05 * r0
z  = z_scale * h0
Q  = -0.375 # -3/8
W  = -0.75 # -3/4
n  = 2 * Q + W
mn = 1
mi = 1
eta = h0**2 / r0**2

# Directory for data
output_dir = f"condinit_{drift}"
os.makedirs(output_dir, exist_ok=True)

# task_per_node (usually the number of processors avaiable in the selected node)
task_per_node = 32

zmin = 0.0
zmax = 0.0205
kz = 2.0 * np.pi / (zmax - zmin)
kzetar0 = kz * eta * r0

# Setup parameters
min_res = 500 # Minimum resolution for the solver
kxmin   = 0.0
kxmax   = 0.0
kzmin   = 1.0e-02
kzmax   = 1.0e+04
if kxmin == kxmax:
    NX = 1
else:
    NX      = ((min_res + task_per_node - 1) // task_per_node) * task_per_node # Ensure NX is multiple of task_per_node
if kzmin == kzmax:
    NZ = 1
else:
    NZ      = ((min_res + task_per_node - 1) // task_per_node) * task_per_node # Ensure NZ is multiple of task_per_node

# List of input parameters for the solver

## Domain decomposition for KX
if kxmin == kxmax:
    kxmin_local = kxmin
    kxmax_local = kxmax
    nx_local = 1
else:
    Kx = np.logspace(np.log10(kxmin), np.log10(kxmax), NX)
    nx_local = NX/N_nodes
    kxmin_local = Kx[int(nx_local*node)]
    kxmax_local = Kx[int(nx_local*(node+1))-1]

## Imput parameters dictionary
input_parameters = {
    "KXmin"  : kxmin_local, "KXmax" : kxmax_local, "KZmin"  : kzmin  , "KZmax" : kzmax ,
    "NX"     : nx_local   , "NZ"    : NZ         , "q"      : q      , "r0"    : r0    , 
    "h0"     : h0         , "z"     : z          , "Q"      : Q      , "W"     : W     ,
    "mn"     : mn         , "mi"    : mi         , "n"      : n      , "Output": output_dir, 
    "N_nodes": N_nodes    , "node"  : node       , "drift": drift    , "task_per_node": task_per_node
}

## Dictionary to str
dump = json.dumps(input_parameters)

print(f"--- Running {drift} at z={z_scale}H | Res: {NX}x{NZ} ---")

# Run the solver in the specific node
os.system("OMP_NUM_THREADS=1 python3 multi_processing.py '{:s}'".format(dump))


