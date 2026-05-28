from pylab import *

#Parameters-------------------------------------------------------------
outputdir = "../../outputs/mri-growth-sb/"   #From the .par file
Ninterm   = 2                                #From the .par file
ny        = 64                               #From the .par file
nz        = 64                               #From the .par file
DT        = 0.1                              #From the .par file
mode      = 0.969228                         #Given at runtime in the terminal
#------------------------------------------------------------------------

#For simplicity we measure the growth rate using use the
#evolution of the max value of the magnetic field. This is not
#precise when radial modes are present, so this should be done in
#the Fourier space.

maxv = []
time = []
for i in range(60):
    by = fromfile(outputdir+"by{:d}.dat".format(i)).reshape(nz,ny)
    time.append(i*DT*Ninterm)
    maxv.append(by.max()) 

#Now we plot the data
fig = figure(figsize=(15,5))
ax1 = fig.add_subplot(131)
ax2 = fig.add_subplot(132)
ax3 = fig.add_subplot(133)

#We get the growth rate with linear fit
a,b = polyfit(time, log(maxv), 1)
t   = linspace(time[0],time[-1],100)

#This is the simulations
ax1.plot(time,maxv,'ro')
#This is the fit
ax1.plot(t,exp(a*t+b),'k--',lw=3,label='Growth rate = {:4.4g}'.format(a))

ax1.legend(loc='best')
ax1.set_yscale('log')
ax1.set_xlabel("t")
ax1.set_ylabel("Amplitude")

#We plot the measured point over the dispertion relation
ax2.plot(mode,a,'ko') #This is the mode that we have simulated

#Now we plot the dispersion relation
k = linspace(0,sqrt(3),1000) 
ax2.plot(k,sqrt(-0.5*(1+2*k**2) + 0.5*sqrt(1+16*k**2)),'k-')

#We show the relative error to estimate how good is the measurement
exact_growth_rate =  sqrt(-0.5*(1+2*mode**2) + 0.5*sqrt(1+16*mode**2))
relative_error    = abs(a - exact_growth_rate)/exact_growth_rate
ax2.set_title("Relative error: {:g}".format(relative_error))
ax2.set_xlabel("Mode")
ax2.set_ylabel("Growth rate")

#Now we plot the colormap and check if everithing is ok
by = fromfile(outputdir+"by60.dat".format(i)).reshape(nz,ny)
ax3.imshow(by,origin='lower', aspect='auto')
ax3.set_xlabel("Y")
ax3.set_ylabel("Z")

show()
