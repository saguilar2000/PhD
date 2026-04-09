import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
from matplotlib.lines import Line2D
import gc
import solver as gs

if len(sys.argv) < 2:
    print(" ")
    print("Run the script as:")
    print("python3 build_map.py <# nodes> <output directory>")
    print("=================================================")
    print(" ")
    exit()

# Inputs
ntot      = int(sys.argv[1])
directory = str(sys.argv[2])+"/"

# Group maps by (chi, betaz)
configs = {}

for fname in os.listdir(directory):
    if not fname.startswith("GRW_") or not fname.endswith(".npz"):
        continue

    parts = fname.replace(".npz", "").split("_")

    # wrap in try/except to handle malformed filenames safely
    try:
        chi   = float(parts[3])
        Am    = float(parts[5])
        betay = float(parts[7])
        betaz = float(parts[9])
        
        key = (chi, betaz)
        configs.setdefault(key, []).append((Am, betay, fname))
    except (IndexError, ValueError):
        continue

# Extract max growth + (kx(ind_max),kz(ind_max))
results = {}  # (chi, betaz) -> list of dicts

kx = np.load(directory + f"KX_0.npz")["arr_0"]
kz = np.load(directory + f"KZ_0.npz")["arr_0"]

for key, entries in configs.items():
    chi, betaz = key
    results[key] = []

    for Am, betay, fname in entries:
        
        archive = np.load(directory + fname)
        growth  = archive["data"]
        drift   = archive["drift"]

        # maximum growth rate
        max_growth = np.max(growth)

        # coordinates of the maximum
        idx = np.unravel_index(np.argmax(growth), growth.shape)
        kx_max = kx[idx[1]]
        kz_max = kz[idx[0]]

        results[key].append({
            "Am": Am,
            "betay": betay,
            "max_growth": max_growth,
            "kx_max": kx_max,
            "kz_max": kz_max,
            "fname": fname,
            "drift": drift
        })
        
        # Explicitly delete the large array
        del growth

# Force cleanup
gc.collect()

# Plotting: 1 figure per (chi, betaz)
for key, entries in results.items():
    
    chi, betaz = key
    
    Ams = sorted(list({e["Am"] for e in entries}))
    ratios  = sorted(list({e["betay"]/betaz for e in entries}))
    
    nL = len(Ams)
    nR = len(ratios)
    
    # allocate matrices
    heat  = np.zeros((nL, nR))
    kxmap = np.zeros((nL, nR))
    kzmap = np.zeros((nL, nR))
    
    # lookup helper to find filenames later
    entry_lookup = {} 
    
    # fill matrices
    for e in entries:
        # Check against zero division or small float issues
        try:
            betay = e["betay"]
            r_val = betay / betaz
            i = Ams.index(e["Am"])
            
            j = np.where(np.isclose(ratios, r_val))[0][0]
            
            heat[i, j]  = e["max_growth"]
            kxmap[i, j] = e["kx_max"]
            kzmap[i, j] = e["kz_max"]
            
            # Map grid coordinates to the specific entry (to get fname later)
            entry_lookup[(i, j)] = e
        except IndexError:
            continue

    # Figure
    fig, axes = plt.subplots(1, 5, figsize=(23, 5), constrained_layout=True)
    
    # 1st panel: Heatmap
    ax = axes[0]
    im = ax.imshow(
        np.log10(heat),
        origin="lower",
        aspect="equal",
        cmap="jet",
        interpolation="none",
        vmin=-3,
        vmax=0
    )
    
    Am_labels = [fr"$10^{{{int(np.log10(l))}}}$" for l in Ams]
    ratio_labels  = [fr"$10^{{{int(np.log10(r))}}}$" for r in ratios]
    
    ax.set_yticks(np.arange(nL))
    ax.set_yticklabels(Am_labels)
    
    ax.set_xticks(np.arange(nR))
    ax.set_xticklabels(ratio_labels, rotation=0)
    
    # grid lines
    ax.set_xticks(np.arange(-0.5, nR, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, nL, 1), minor=True)
    ax.grid(which='minor', color='black', linestyle='-', linewidth=0.5)
    ax.grid(which='major', linestyle='', linewidth=0)
    
    ax.set_xlabel(r"$\beta_{\phi}/\beta_z$")
    ax.set_ylabel(r"$A_m$")
    ax.set_title(fr"($\chi_i$={chi}, $\beta_z$={betaz})")
    
    # Annotate corner cells
    if nL < 2 or nR < 2:
        corners = [
            (0,      0,   "L(0,0)"),
            (1,      0,   "L(0,1)"),
            (2,      0,   "L(1,0)"),
            (3,      0,   "L(1,1)")
        ]
    else:
        corners = [
            (0,      0,      "L(0,0)"),
            (0,      nR-1,   "L(0,1)"),
            (nL-1,   0,      "L(1,0)"),
            (nL-1,   nR-1,   "L(1,1)")
        ]
    
    for (i, j, label) in corners:
        txt = ax.text(j, i, label, color="white", ha="center", va="center",
                fontsize=7, fontweight="bold")
        txt.set_path_effects([PathEffects.withStroke(linewidth=1, foreground='k')])

    # Corners subplots
    kx_full = np.load(directory + f"KX_0.npz")["arr_0"]
    kz_full = np.load(directory + f"KZ_0.npz")["arr_0"]

    #Reduce grid density for plotting.
    if len(kx_full) >= 5000 or len(kz_full) >= 5000:
        STEP = 10 
        # Slice coordinates immediately
        kx_plot = kx_full[::STEP]
        kz_plot = kz_full[::STEP]
    else:
        kx_plot = kx_full
        kz_plot = kz_full

    for k, (i, j, label) in enumerate(corners):
        axc = axes[k+1]
        
        if (i, j) in entry_lookup:
            e = entry_lookup[(i, j)]
            
            # Load the map
            full_map = np.load(directory + e["fname"])["data"]
            
            # Slice the map immediately to save RAM (if STEP in locals)
            if 'STEP' in locals():
                map_plot = full_map[::STEP, ::STEP]
            else:
                map_plot = full_map
            
            # Delete the huge array immediately
            del full_map 

            # Filter NaNs on the small map
            map_plot[map_plot <= 1e-3] = np.nan
            
            Am_c = e["Am"]
            ratio_c = e["betay"]/betaz

            im1 = axc.pcolormesh(
                kx_plot, kz_plot, np.log10(map_plot), 
                shading='auto', cmap='jet', vmin=-3,
                rasterized=True 
            )
            
            # Delete the small map too, we are done with it
            del map_plot

            # Resonant Modes
            if "_drift" in directory:
                betay_val = e["betay"]
                r0, h0 = 1.0, 0.05
                sqrt_eta = h0 / r0
                
                KX, KZ = np.meshgrid(kx_plot, kz_plot)
                fx, fy, fz = e["drift"]
                # print(fx, fy, fz)
                res_cond = KX * (1/sqrt_eta) * fx + KZ * (1/sqrt_eta) * fz - np.abs(KZ / np.sqrt(KX**2 + KZ**2))
                
                axc.contour(KX, KZ, res_cond, levels=[0], colors='k', linestyles='--', linewidths=1)
                axc.text(0.01, 0.99, rf"$f_x={fx:.2e}$" "\n"rf"$f_y={fy:.2e}$" "\n"rf"$f_z={fz:.2e}$"
                         ,transform=axc.transAxes, ha="left", va="top")
                
                # Clean up meshgrids
                del KX, KZ, res_cond

            axc.set_title(fr"{label}: $A_m$={Am_c:.0e}, $\beta_{{\varphi}}/\beta_z$={ratio_c:.0e}")
            
        else:
            axc.set_title(f"{label}: No Data")

        axc.set_xscale('log')
        axc.set_yscale('log')
        axc.set_xlabel(r"$K_x$")
        if k == 0:
            axc.set_ylabel(r"$K_z$")
    
    cbar = fig.colorbar(im1, ax=axes[1:], extend='both', label=r"$\log_{10}(\sigma/\Omega_0)$")
    if 'STEP' in locals():
        del STEP
    # Save Output
    drift_dir = directory.split("_")[-2]
    z_dir = directory.split("_")[-1].replace("/","")
    output = f"figures/{z_dir}/{drift_dir}"
    if not os.path.exists(output):
        os.makedirs(output, exist_ok=True)
        print(f"Created directory: {output}")
    
    plt.savefig(f"{output}/CASE_{drift_dir}_CHI_{chi:.0e}_BETAZ_{betaz:.0e}.png", dpi=300)
    plt.close('all') # Closes all figures
    gc.collect()     # Forces Python to release memory back to OS
