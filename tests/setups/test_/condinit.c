#include "fargo3d.h"

void Init() {

  int i,j,k;

  //These are global structures
  real *rho = Density->field_cpu; //Centered
  real *cs  = Energy->field_cpu;  //Centered
  real *vx  = Vx->field_cpu;      //Staggered in x
  real *vy  = Vy->field_cpu;      //Staggered in y
  real *vz  = Vz->field_cpu;      //Staggered in z
  real *bx  = Bx->field_cpu;      //Staggered in x
  real *by  = By->field_cpu;      //Staggered in y
  real *bz  = Bz->field_cpu;      //Staggered in z
  
  real b0z    = SOUNDSPEEDNEUTRAL*sqrt(2.0*MU0/BETAZ);
  real b0x    = SOUNDSPEEDNEUTRAL*sqrt(2.0*MU0/BETAP);
  real b0     = sqrt(b0x*b0x + b0z*b0z);
  real va     = b0/sqrt(MU0);
  real kmode  = 2*M_PI/(ZMAX-ZMIN);
  
  //We now compute the amplitudes:
  real dvyi = 0.0;
  real dvxi = 0.0;
  real dvyn = 0.0;
  real dvxn = 0.0;
  real dby  = 0.0;
  real dbx  = 0.0;

  real f_zmax, f_dvnx, f_dvny, f_dvix, f_dviy, f_dbx, f_dby;
  int found = 0;
  FILE *fp = fopen("setups/2fmri/eigenvectors.dat", "rb");

  while (fread(&f_zmax, sizeof(real), 1, fp) == 1) {
    fread(&f_dvnx, sizeof(real), 1, fp);
    fread(&f_dvny, sizeof(real), 1, fp);
    fread(&f_dvix, sizeof(real), 1, fp);
    fread(&f_dviy, sizeof(real), 1, fp);
    fread(&f_dbx,  sizeof(real), 1, fp);
    fread(&f_dby,  sizeof(real), 1, fp);
    //printf("Reading eigenvector for ZMAX = %f\n", f_zmax);

    // Check if this row matches our current simulation box size
    if (fabs(f_zmax - ZMAX) < 1e-5) {
      dvxn = f_dvnx;
      dvyn = f_dvny;
      dvxi = f_dvix;
      dvyi = f_dviy;
      dbx  = f_dbx;
      dby  = f_dby;
      found = 1;
      //printf("Found eigenvector for ZMAX = %f, dvxn = %f, dvyn = %f, dvxi = %f, dvyi = %f, dbx = %f, dby = %f\n", ZMAX, dvxn, dvyn, dvxi, dvyi, dbx, dby);
      break; 
    }
  }
  fclose(fp);

  real norm  = sqrt(pow(dvxi,2) + pow(dvyi,2) + pow(dvxn,2) + pow(dvyn,2) + pow(dbx,2) + pow(dby,2));

  dvxi *= SOUNDSPEEDNEUTRAL*AMPLITUDE/norm;
  dvyi *= SOUNDSPEEDNEUTRAL*AMPLITUDE/norm;
  dvxn *= SOUNDSPEEDNEUTRAL*AMPLITUDE/norm;
  dvyn *= SOUNDSPEEDNEUTRAL*AMPLITUDE/norm;
  dbx *= b0*AMPLITUDE/norm;
  dby *= b0*AMPLITUDE/norm;


  //df = dfr*cos(kz) - dfi*sin(kz)
  //Loop over the mesh
  for (k=0; k<Nz+2*NGHZ; k++) {
    for (j=0; j<Ny+2*NGHY; j++) {
      for (i=0; i<Nx+2*NGHX; i++) {

		
	if(Fluidtype==GAS){
	  rho[l] =  1.0;
	  cs[l]  =  SOUNDSPEEDION;
	
	  vx[l]  = -1.5*OMEGAFRAME*ymed(j);
	  vy[l]  =  0.0;
	  vz[l]  =  0.0;
	  bx[l]  =  b0x;
	  by[l]  =  0.0;
	  bz[l]  =  b0z;

	  vx[l] += dvxi*sin(kmode*zmed(k));
	  vy[l] += dvyi*sin(kmode*zmed(k));
	  bx[l] += dbx*cos(kmode*zmed(k));
	  by[l]  = dby*cos(kmode*zmed(k));

	}
	if(Fluidtype==DUST){
	  
	  rho[l] =  1.0*EPSILON;
	  cs[l]  =  SOUNDSPEEDNEUTRAL;
	  vy[l]  =  0.0;
	  vz[l]  =  0.0;
	  vx[l]  = -1.5*OMEGAFRAME*ymed(j);
	  vx[l] += dvxn*sin(kmode*zmed(k));
	  vy[l] += dvyn*sin(kmode*zmed(k));
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
