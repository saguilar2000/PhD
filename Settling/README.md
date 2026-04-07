# Settling set-up

Framework to replicate Figure 1 in Krapp et. al., 2020 (doi:10.1093/mnras/staa1854)

Changes to main set-up:
	- **Parameters**: Krapp uses Ts(\equiv 1/\alpha) = 1e-2 and \epsilon(\equiv \epsilon_i) = 1e-3.
	- **Magnetic fields**: magnetic fields should be turned off (not sure yet).
	- **Drift**: if there are no magnetic fields, the framework wont calculate any drift values, hence drift values must be included by-hand.

Build:
	- solver.py: 10 x 10 matrix,2 fluid equations for compressible fluids, considering the solenoid condition, x1 and x2 variables to turn off ions compressibility and the Lorentz force (to replicate Krapp's figure I am using x1=x2=0, i. e., incompressible ions and no Lorentz forces).
	- condinit_parallel_process.py: Parameters and start-up of the parallelized solver.
	- condinit_multi_processing.py: Calls the solver, computes and saves the eigenvalues for the actual set-up. Drift parameters set to fx = -0.001, fy = 0.0, fz = 0.0157, I still need to figure out how to obtain those values from input parameters.

Usage: Run <make_condinit.sh> to create the data for the growth rates and then run plot.sh to plot the figure using the data. The figure will be stored in the folder "./figures".