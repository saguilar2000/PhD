//<FLAGS>
//#define __GPU
//#define __NOPROTO
//<\FLAGS>

//<INCLUDES>
#include "fargo3d.h"
//<\INCLUDES>

void _analytical_update(real dt, Field *V_n, Field *V_i) {
//<USER_DEFINED>
  INPUT(Density);
  INPUT(V_n);
  INPUT(V_i);
  OUTPUT(V_n);
  OUTPUT(V_i);
//<\USER_DEFINED>

//<EXTERNAL>
  real* rho_n = Fluids[1]->Density->field_cpu;
  real* rho_i = Fluids[0]->Density->field_cpu;
#ifdef X
  real* vi = V_i->field_cpu;
  real* vn = V_n->field_cpu;
#endif
  real alphacol = ALPHACOL;
  real eps_i    = EPSILON;
  real omega    = OMEGAFRAME;
  int size_x = XIP; 
  int size_y = Ny+2*NGHY-1;
  int size_z = Nz+2*NGHZ-1;
//<\EXTERNAL>

//<INTERNAL>
  int i;
  int j;
  int k;
  real exponential;
  real denom;
//<\INTERNAL>
 
//<MAIN_LOOP>

  i = j = k = 0;

  for (k=1; k<size_z; k++) {
    for (j=1; j<size_y; j++) {
      for (i=XIM; i<size_x; i++) {
//<#>
        denom = 1 / (1 + eps_i);
        exponential = exp(-alphacol * omega * (1 + eps_i) * dt);
        
        if(Fluidtype == GAS) {
          vi[l] = ((eps_i + exponential) * vi[l] + (1 - exponential) * vn[l]) * denom;
        }
        if(Fluidtype == DUST) {
          vn[l] = ((1 - exponential) * eps_i * vi[l] + (1 + eps_i * exponential) * vn[l]) * denom;
        }
//<\#>
      }
    }
  }
  printf("dt = %f, vi = %f, vn = %f \n",dt,vi,vn);
//<\MAIN_LOOP>
}

void Analytical_update(real dt, int option) {
  if (option == 0) {
    FARGO_SAFE(_analytical_update(dt, Fluids[1]->Vx_temp, Fluids[0]->Vx_temp));
  }
  if (option == 1) {
    FARGO_SAFE(_analytical_update(dt, Fluids[1]->Vy_temp, Fluids[0]->Vy_temp));
  }
  if (option == 2) {
    FARGO_SAFE(_analytical_update(dt, Fluids[1]->Vz_tempm Fluids[0]->Vz_temp))
  }
}
