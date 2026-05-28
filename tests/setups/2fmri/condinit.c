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
  
  real b0z    = SOUNDSPEEDNEUTRAL*sqrt(2.0*MU0/BETAZ);
  real b0x    = SOUNDSPEEDNEUTRAL*sqrt(2.0*MU0/BETAP);
  real b0     = sqrt(b0x*b0x + b0z*b0z);
  real va     = b0/sqrt(MU0);
  real kmode  = 2*M_PI/(ZMAX-ZMIN);

  //df = dfr*cos(kz) - dfi*sin(kz)
  //Loop over the mesh
  for (k=0; k<Nz+2*NGHZ; k++) {
    for (j=0; j<Ny+2*NGHY; j++) {
      for (i=0; i<Nx+2*NGHX; i++) {

	// iones
	if(Fluidtype==GAS){
	  rho[l] =  1.0*EPSILON;
	  cs[l]  =  SOUNDSPEEDION;
	
	  vx[l]  =  0.0;
	  vy[l]  = -1.5*OMEGAFRAME*ymed(j);
	  vz[l]  =  0.0;

	  bx[l]  =  0.0;
	  by[l]  =  0.0;
	  bz[l]  =  b0z;

    vx[l] += NOISE*SOUNDSPEEDNEUTRAL*(-1.0+2.0*drand48());
    vy[l] += NOISE*SOUNDSPEEDNEUTRAL*(-1.0+2.0*drand48());
    vz[l] += NOISE*SOUNDSPEEDNEUTRAL*(-1.0+2.0*drand48());

	}
  // neutros
	if(Fluidtype==DUST){
	  
	  rho[l] =  1.0;
	  cs[l]  =  SOUNDSPEEDNEUTRAL;

	  vx[l]  =  0.0;
	  vy[l]  = -1.5*OMEGAFRAME*ymed(j);
	  vz[l]  =  0.0;

    vx[l] += NOISE*SOUNDSPEEDNEUTRAL*(-1.0+2.0*drand48());
    vy[l] += NOISE*SOUNDSPEEDNEUTRAL*(-1.0+2.0*drand48());
    vz[l] += NOISE*SOUNDSPEEDNEUTRAL*(-1.0+2.0*drand48());
	}
      }
    }
  }
}



void CondInit() {
   Fluids[0] = CreateFluid("gas",GAS);
   SelectFluid(0);
   Init();

   Fluids[1] = CreateFluid("dust",DUST);
   SelectFluid(1);
   Init();

   //
   ColRate(ALPHACOL, 0, 1, YES);

}
