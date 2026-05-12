import numpy as np
import matplotlib.pyplot as plt
import os
import struct
import argparse
from scipy.fft import fft2, fftshift
from pylab import *

# Constants
r0 = 1.0
h0 = 0.05 * r0
eta = h0**2 / r0**2
invsqrteta = 1.0 / np.sqrt(eta)
zmin = 0.0

def zmed(dz):
    # return zmin + (k + 0.5) * (zmax - zmin) / 64.0
    return dz

def get_eigvec():

    dir_fargo = "setups/2fmri/condinit_nodrift/fargo/"

    files = [f for f in os.listdir(dir_fargo) if f.startswith("GRW")]
    # if eigenvectors.dat exists, remove it to avoid appending to old data
    if os.path.exists("eigenvectors.dat"):
        os.remove("eigenvectors.dat")
    for f in files:
        zmax = float(f.split("_kz_")[1].replace(".npz", ""))

        data = np.load(dir_fargo + f)
        kz_tilde = data['kz']
        betay = data['betay']
        betaz = data['betaz']
        Am_ = data['Am']
        chi = data['chi']
        mi = data['mi']
        mn = data['mn']
        eps = chi * (mi/mn)

        omega = data['data'][0][0]

        kz_mode  = kz_tilde / (eta * r0)
        dz = 2 * np.pi / kz_mode
        kzvA_omega = kz_tilde * invsqrteta * np.sqrt(2.0 / betaz) * np.sqrt((1.0 + chi)/(1.0 + eps))

        eigvecs_ = np.load(dir_fargo + f.replace("GRW", "VEC"))['data'][0,0,:]
        eigvecs = eigvecs_[:, 0] + 1j * eigvecs_[:, 1]
        i_max = np.argmax(np.abs(eigvecs))
        eigvecs *= np.exp(-1j * np.angle(eigvecs[i_max]))

        hdvnx = eigvecs[0]
        hdvny = eigvecs[1]
        hdvnz = eigvecs[2]
        hdvix = eigvecs[3]
        hdviy = eigvecs[4]
        hdviz = eigvecs[5]
        hdbx  = eigvecs[6]
        hdby  = eigvecs[7]
        hdpn  = eigvecs[8]
        hdpi  = eigvecs[9]

        AMPLITUDE = 1.0e-6

        dvnx = (hdvnx.real * np.cos(kz_mode * zmed(dz)) - hdvnx.imag * np.sin(kz_mode * zmed(dz)))
        dvny = (hdvny.real * np.cos(kz_mode * zmed(dz)) - hdvny.imag * np.sin(kz_mode * zmed(dz)))
        dvnz = (hdvnz.real * np.cos(kz_mode * zmed(dz)) - hdvnz.imag * np.sin(kz_mode * zmed(dz)))
        dvix = (hdvix.real * np.cos(kz_mode * zmed(dz)) - hdvix.imag * np.sin(kz_mode * zmed(dz)))
        dviy = (hdviy.real * np.cos(kz_mode * zmed(dz)) - hdviy.imag * np.sin(kz_mode * zmed(dz)))
        dviz = (hdviz.real * np.cos(kz_mode * zmed(dz)) - hdviz.imag * np.sin(kz_mode * zmed(dz)))
        dbx  = (hdbx.real * np.cos(kz_mode * zmed(dz)) - hdbx.imag * np.sin(kz_mode * zmed(dz)))
        dby  = (hdby.real * np.cos(kz_mode * zmed(dz)) - hdby.imag * np.sin(kz_mode * zmed(dz)))
        dpn  = (hdpn.real * np.cos(kz_mode * zmed(dz)) - hdpn.imag * np.sin(kz_mode * zmed(dz)))
        dpi  = (hdpi.real * np.cos(kz_mode * zmed(dz)) - hdpi.imag * np.sin(kz_mode * zmed(dz)))

        with open("eigenvectors.dat", "ab") as bin_file:
            # We pack: zmax (d), dvnx (d), dvny (d), dvix (d), dviy (d), dbx (d), dby (d)
            # 'd' stands for double (8 bytes), matching 'real' in FARGO3D
            data_row = struct.pack('ddddddd', zmax, dvnx, dvny, dvix, dviy, dbx, dby)
            bin_file.write(data_row)
            print("done .dat file")
        print(f"zmax={zmax:.4f}, kz_mode={kzvA_omega:.4f}, growth_rate={omega:.4f}")
        print(f"dvnx={hdvnx}={dvnx:.4e},\ndvny={hdvny}={dvny:.4e},\ndvnz={hdvnz}={dvnz:.4e},\ndvix={hdvix}={dvix:.4e},\ndviy={hdviy}={dviy:.4e},\ndviz={hdviz}={dviz:.4e},\ndbx={hdbx}={dbx:.4e},\ndby={hdby}={dby:.4e},\ndpn={hdpn}={dpn:.4e},\ndpi={hdpi}={dpi:.4e}\n")

def get_mode(outputdir):
    parfile = open(outputdir+"variables.par","r")
    for line in parfile:
        if line.rfind("SOUNDSPEED") >= 0:
            cs = float(line.split()[1])
        if line.rfind("BETA") >= 0:
            beta = float(line.split()[1])
        if line.rfind("OMEGAFRAME") >= 0:
            om0 = float(line.split()[1])
        if line.rfind("ZMIN") >= 0:
            zmin = float(line.split()[1])
        if line.rfind("ZMAX") >= 0:
            zmax = float(line.split()[1])
            
    #It only works if mu0 = rho0 = 1!
    print(beta)
    kmode = 2*np.pi/(zmax-zmin)*cs*np.sqrt(2/beta)/om0; 
    print( cs, beta, zmin, zmax, "==> ktilde-mode =", kmode)
    
    return kmode

def curve():
    # dir_fargo = "outputs/2fmri/"
    dir_solver = "setups/2fmri/condinit_nodrift/"

    z = [ 0.3974, 0.0911,
      0.0274 , 0.0223,
      0.012, 0.0115]
    
    ny = 64
    nz = 64
    ninterm = 2
    dt      = 0.1

    data_solver = np.load(dir_solver + f"GRW_0_kz_0.0000.npz")
    kz_tilde_solver = np.load(dir_solver + f"KZ_0.npz")['arr_0']
    
    omega_solver = data_solver['data']

    betaz_solver = data_solver['betaz']
    betay_solver = data_solver['betay']
    Am_solver = data_solver['Am']
    chi_solver = data_solver['chi']

    eps_solver = chi_solver

    kzvA_omega_solver = kz_tilde_solver * invsqrteta * np.sqrt(2.0 / betaz_solver) * np.sqrt((1.0 + chi_solver)/(1.0 + eps_solver))

    fig, ax = plt.subplots()
    ax.plot(kzvA_omega_solver, omega_solver, 'k-', label='Solver')

    for j in range(len(z)):
        outputdir = "outputs/2fmri/output{:d}/".format(j)
        mode = get_mode(outputdir)
        #We will use "By" to measure the growth rate.
        maxv = []
        amp_k = []
        time = []
        for i in range(20):
            by = np.fromfile(outputdir+"by{:d}.dat".format(i)).reshape(nz,ny)
            time.append(i*dt*ninterm)
            #Here, for simplicity, we use the evloution of the max value of
            #the magnetic field to measure the growth rate. Note that you can
            #also do it in the Fourier space.
            maxv.append(by.max())
            # fourier space
            by_fourier = abs(fft2(by))
            amp_k.append(by_fourier.max())

        #We get the growth rate with linear fit
        a,b = polyfit(time, log(maxv), 1)
        c,d = polyfit(time, log(amp_k), 1)
        #We plot the measured point over the dispertion relation
        ax.plot(mode,a,'ko') #This is the mode that we have simulated
        ax.plot(mode,c,'ro') #This is the mode that we have simulated, but with the Fourier space measurement of the growth rate
        # print(mode, a)

    ax.set_xlim(0, 4.0)
    ax.set_ylim(0, 0.9)
    ax.set_xlabel(r"$k_z v_{Az} / \Omega_0$")
    ax.set_ylabel(r"$Im(\omega / \Omega_0)$")
    ax.legend()
    plt.show()

def main(args):
    if args.eigvec:
        get_eigvec()
    if args.curve:
        curve()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute growth rates and eigenvectors for the 2fmri setup.")
    parser.add_argument("-e", "--eigvec", action="store_true", help="Compute eigenvectors and save to file.")
    parser.add_argument("-c", "--curve", action="store_true", help="Plot the growth rate curve.")
    args = parser.parse_args()
    main(args)