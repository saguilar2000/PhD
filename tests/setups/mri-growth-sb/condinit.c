#include "fargo3d.h"

void Init() {

  int i,j,k;

  //These are global structures
  real *rho = Density->field_cpu; //Centered
  real *cs  = Energy->field_cpu;  //Centered
  real *vx  = Vy->field_cpu;      //Staggered in x
  real *vy  = Vx->field_cpu;      //Staggered in y
  real *vz  = Vz->field_cpu;      //Staggered in z
  real *bx  = By->field_cpu;      //Staggered in x
  real *by  = Bx->field_cpu;      //Staggered in y
  real *bz  = Bz->field_cpu;      //Staggered in z
  
  real b0     = SOUNDSPEED*sqrt(2.0*MU0/BETA);
  real va     = b0/sqrt(MU0);
  real kmode  = 2*M_PI/(ZMAX-ZMIN);
  real ktilde = kmode*va/OMEGAFRAME;
  real sigma  = sqrt(sqrt(16*pow(ktilde,2)+1) - 2*pow(ktilde,2)-1)/sqrt(2);
  
  //Loop over the mesh
  for (k=0; k<Nz+2*NGHZ; k++) {
    for (j=0; j<Ny+2*NGHY; j++) {
      for (i=0; i<Nx+2*NGHX; i++) {

	//Note: index l is the cell index
	
	rho[l] =  1.0;
	cs[l]  =  SOUNDSPEED;
	
	vx[l]  =  0.0;
	vy[l]  = -1.5*OMEGAFRAME*ymed(j);
	vz[l]  =  0.0;

	bx[l]  =  0.0;
	by[l]  =  0.0;
	bz[l]  =  b0;
  
  vx[l] += NOISE*SOUNDSPEED*(-1.0+2.0*drand48());
  vy[l] += NOISE*SOUNDSPEED*(-1.0+2.0*drand48());
  vz[l] += NOISE*SOUNDSPEED*(-1.0+2.0*drand48());

      }
    }
  }
}

void CondInit() {
   Fluids[0] = CreateFluid("gas",GAS);
   SelectFluid(0);
   Init();
}
