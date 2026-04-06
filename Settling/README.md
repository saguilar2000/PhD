# Settling set-up

Framework to replicate Figure 1 in Krapp et. al., 2020 (doi:10.1093/mnras/staa1854)

Changes:
	- **Parameters**: Krapp uses Ts(\equiv 1/\alpha) = 1e-2 and \epsilon(\equiv \epsilon_i) = 1e-3.
	- **Magnetic fields**: magnetic fields should be turned off (not sure yet).
	- **Drift**: if there are no magnetic fields, my framework wont calculate any drift, hence drift values must be included by-hand.

Build:
	- solver.py: 10 x 10 matrix,2 fluid equations for compressible fluids, considering the solenoid condition.
	- condinit_parallel_process.py: Parameters and start-up of the parallelized solver.
	- condinit_multi_processing.py: Calls the solver, computes and saves the eigenvalues for the actual set-up.
