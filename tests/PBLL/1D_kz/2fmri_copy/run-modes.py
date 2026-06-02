from pylab import *
import os
from scipy.fft import fft2, fftshift
import numpy as np

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
    kmode = 2*pi/(zmax-zmin)*cs*sqrt(2/beta)/om0; 
    print( cs, beta, zmin, zmax, "==> ktilde-mode =", kmode)
    
    return kmode

#z = [ 0.3974, 0.0911,
#      0.0514, 0.0358,
#      0.0274 , 0.0223,
#      0.0188, 0.0162,
#      0.0142 , 0.0127,
#      0.012, 0.0115]

setup = "back_2fmri"
if setup == "back_2fmri":
    z = [0.03974]
    ntot = 40
else:
    z = [ 0.3974, 0.0911,
      0.0274 , 0.0223,
      0.012, 0.0115]
    ntot = 160

ny = 64
nz = 64
ninterm = 2
dt      = 0.1

os.system("rm -rf outputs/{:s}/*".format(setup))
os.system("python3 setups/2fmri/analyze.py -e".format(setup))

for i,z0 in enumerate(z):
    os.system("./fargo3d -o 'zmax={0:4.4f} outputdir=outputs/{2:s}/output{1:d} ny={3:d} nz={4:d} ninterm={5:d} dt={6:f} ntot={7:d}' setups/2fmri/2fmri.par".format(z0,i, setup, ny, nz, ninterm, dt, ntot))

if setup == "back_2fmri":
    os.system("python3 setups/2fmri/analyze.py -k")
else:
    os.system("python3 setups/2fmri/analyze.py -c")

# fig = figure(figsize=(5,5))
# ax1 = fig.add_subplot(111)

# #We start with a plot of the dispersion relation
# k = linspace(0,sqrt(3),1000) 
# ax1.plot(k,sqrt(-0.5*(1+2*k**2) + 0.5*sqrt(1+16*k**2)),'k-')
# ax1.set_xlabel("Mode")
# ax1.set_ylabel("Growth rate")

# #Now we put the simulations on top of it
# for j in range(len(z)):
#     outputdir = "outputs/2fmri/output{:d}/".format(j)
#     mode = get_mode(outputdir)
#     #We will use "By" to measure the growth rate.
#     maxv = []
#     amp_k = []
#     time = []
#     for i in range(20):
#         by = fromfile(outputdir+"by{:d}.dat".format(i)).reshape(nz,ny)
#         time.append(i*dt*ninterm)
#         #Here, for simplicity, we use the evloution of the max value of
#         #the magnetic field to measure the growth rate. Note that you can
#         #also do it in the Fourier space.
#         maxv.append(by.max())
#         # fourier space
#         by_fourier = abs(fft2(by))
#         amp_k.append(by_fourier.max())

#     #We get the growth rate with linear fit
#     a,b = polyfit(time, log(maxv), 1)
#     c,d = polyfit(time, log(amp_k), 1)
#     #We plot the measured point over the dispertion relation
#     ax1.plot(mode,a,'ko') #This is the mode that we have simulated
#     ax1.plot(mode,c,'ro') #This is the mode that we have simulated, but with the Fourier space measurement of the growth rate
#     print(mode, a)
# show()
