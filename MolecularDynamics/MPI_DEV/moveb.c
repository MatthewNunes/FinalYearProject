#include "moldyn.h"

void moveb (kinetic, vx, vy, vz, natm, scratch, fparam)
float *kinetic;
float vx[];
float vy[];
float vz[];
int natm;
float *scratch;
float *fparam;
{
   float dt2;
   int i;
   float *fx, *fy, *fz;

   float dt = fparam[2];

   fx = scratch;
   fy = &scratch[natm];
   fz = &scratch[2*natm];

   dt2 = dt*0.5;
   *kinetic = 0.0;
   for(i=0;i<natm;i++){
      vx[i] = vx[i] + dt2*fx[i];
      vy[i] = vy[i] + dt2*fy[i];
      vz[i] = vz[i] + dt2*fz[i];
      *kinetic += vx[i]*vx[i] + vy[i]*vy[i] + vz[i]*vz[i];
   }
   *kinetic *= 0.5;
}
