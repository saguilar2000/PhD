import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.fft import fft2, fftshift

OUTPUTDIR = "outputs/2fmri/background/output0/"
# OUTPUT2fm = "outputs/2fmri/output0/"
with open(OUTPUTDIR+"variables.par","r") as parfile:
    for line in parfile:
        if line.rfind("DT") >= 0:
            DT = float(line.split()[1])
        if line.rfind("NINTERM") >= 0:
            NINTERM = int(line.split()[1])
        if line.rfind("NTOT") >= 0:
            NTOT = int(line.split()[1])
        if line.rfind("NY") >= 0:
            NY = int(line.split()[1])
        if line.rfind("NZ") >= 0:
            NZ = int(line.split()[1])

maxv = []
time = []
var = "gasvy"

for i in range(NTOT//NINTERM + 1):
    v = np.fromfile(OUTPUTDIR + "{:s}{:d}.dat".format(var,i), dtype=np.float64).reshape((NY,NZ))
    maxv.append((v - v.mean(axis=0)).max())
    # maxv.append(v.max())
    time.append(i * DT * NINTERM)

maxv2 = []
time2 = []

# for i in range(NTOT//NINTERM + 1):
#     v = np.fromfile(OUTPUT2fm + "{:s}{:d}.dat".format(var,i), dtype=np.float64).reshape((NY,NZ))
#     maxv2.append((v - v.mean(axis=0)).max())
#     # maxv.append(v.max())
#     time2.append(i * DT * NINTERM)

fig = plt.figure(figsize = (10,5))
ax = fig.add_subplot(111)
ax.plot(time, maxv, 'ro')
# ax.plot(time2, maxv2, 'bo')
ax.set_yscale("log")
ax.set_xlabel("Time")
ax.set_ylabel("Max({:s})".format(var))


plt.show()