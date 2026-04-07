#!/bin/bash
#SBATCH --partition=mapu
#SBATCH -N 1
#SBATCH --ntasks-per-node=32
#SBATCH --job-name=condinit
#SBATCH --error=condinit_%j.e
#SBATCH --output=condinit_%j.out

# args
# usage: sbatch make_condinit.sh [drift/nodrift] [0.0/0.5/1.0]
DRIFT=${1:-"drift"}
Z_SCALE=${2:-"0.0"}

module load Python/3.11.4

python3 parallel_solver.py 1 0 "$DRIFT" "$Z_SCALE"(base)