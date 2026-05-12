# 2D set-up

Framework to analyze 2D growth rates

Changes to main set-up:
	- **Parameters**: 
	- **Magnetic fields**: 
	- **Drift**: 

Build:
	- solver.py: 10 x 10 matrix,2 fluid equations for compressible fluids, considering the solenoid condition, x1 and x2 variables to turn off ions compressibility and the Lorentz force (to replicate Krapp's figure I am using x1=x2=0, i. e., incompressible ions and no Lorentz forces).
	- condinit_parallel_process.py: Parameters and start-up of the parallelized solver.
	- condinit_multi_processing.py: Calls the solver, computes and saves the eigenvalues for the actual set-up.

Usage: Run <make_condinit.sh> to create the data for the growth rates and then run plot.sh to plot the figure using the data. The figure will be stored in the folder "./figures".