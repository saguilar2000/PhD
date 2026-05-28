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

z = [ 0.3974, 0.0911,
      0.0274 , 0.0223,
      0.012, 0.0115]

ny = 64
nz = 64
ninterm = 2
dt      = 0.1
os.system("rm -rf outputs/2fmri/background/*")
os.system("python3 setups/2fmri/analyze.py -e")

for i,z0 in enumerate(z):
    os.system("./fargo3d -o 'zmax={0:4.4f} outputdir=outputs/2fmri/background/output{1:d} ny={2:d} nz={3:d} ninterm={4:d} dt={5:f}' setups/2fmri/2fmri.par".format(z0,i, ny, nz, ninterm, dt))

# os.system("python3 setups/2fmri/analyze.py -c")
