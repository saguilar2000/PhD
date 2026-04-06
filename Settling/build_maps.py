import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import random

def build_maps(directory):

    # Maps
    map_fname = [fname for fname in os.listdir(directory) if fname.startswith("GRW_0") and fname.endswith(".npz")]
    data = np.load(directory + map_fname[0])
    drift = data["drift"]
    Am = data["Am"]
    chi = data["chi"]
    betay = data["betay"]
    betaz = data["betaz"]
    mn = 2.34
    mi = 30
    eps = chi * (mi / mn)
    alpha = Am / eps
    print(f"wx: {drift[0]:.2e}", "\n", 
          f"wy: {drift[1]:.2e}", "\n", 
          f"wz: {drift[2]:.2e}", "\n", 
          f"Am: {Am:.2e}", "\n",
          f"alpha: {alpha:.2e}", "\n",
          f"chi: {chi:.2e}", "\n",
          f"eps: {eps:.2e}", "\n",
          f"betay: {betay:.2e}", "\n",
          f"betaz: {betaz:.2e}")

    # Load data
    kx  = np.load(directory+"KX_0.npz")
    kz  = np.load(directory+"KZ_0.npz")
    Kkx = kx["arr_0"]
    Kkz = kz["arr_0"]
    MAP = data["data"]

    # plotting
    fig = plt.figure(figsize=(6,6))
    MAP[MAP<=1e-3] = np.nan
    axes = plt.subplot(1,1,1)

    im1 = axes.pcolormesh(Kkx, Kkz, np.log10(MAP), shading='auto', cmap='plasma', vmin=-3, vmax=0)
    axes.set_xscale('log')
    axes.set_yscale('log')
    axes.set_xlabel(r'$K_x\eta r_0$')
    axes.set_ylabel(r'$K_z\eta r_0$')
    cbar = fig.colorbar(im1, ax=axes, extend='both')

    cax = cbar.ax
    cax.set_ylabel(r'$\log_{10}(\mathrm{Im}(\omega)/\Omega)$', rotation=270, labelpad=15)
    plt.show()

def main():
    if len(sys.argv) < 2:
        print(" ")
        print("Run the script as:")
        print("python3 build_map.py <directory>")
        print("=================================================")
        print(" ")
        exit()

    dir = sys.argv[1]

    build_maps(dir)

if __name__ == "__main__":
    main()