# PnP2005 set-up

Framework to replicate right panel in Figure 9 in Pessah & Psaltis 2005 (PnP2005 hereafter; doi:10.1086/430940)

Changes to main set-up:
	- **parameters**: PnP2005 uses the ratio cs2/vA2(\sim \beta/2; needs to account for each direction) = [10, 1, 0.1, 0.03, 0.02, 0.01, 6e-4].
	- **drift**: No drift included asn PnP2005 does not account for a drift effect on the results (fx = fy = fz = 0).
	- **ionization fraction**: As PnP2005 works with a fully ionized media, we use chi(\equiv \rho_i/\rho_n) = 1.0 (charge equilibrium).

Actual state: 
	- solver.py: 10 x 10 matrix,2 fluid equations for compressible fluids, considering the solenoid condition.
	- condinit_parallel_process.py: Contains parameters to start the parallelized solver.
	- condinit_multi_processing.py: Calls the solver, computes and saves the eigenvalues for the actual set-up.

Usage: Run <make_condinit.sh> to create the data for the growth rates and then run plot.sh to plot the figure using the data. The figure will be stored in the folder "./figures".