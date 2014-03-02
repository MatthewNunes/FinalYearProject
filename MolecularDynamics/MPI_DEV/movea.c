#include "moldyn.h"

void movea (rx, ry, rz, vx, vy, vz, natm, scratch,fparam)
float rx[];
float ry[];
float rz[];
float vx[];
float vy[];
float vz[];
int natm;
float *scratch;
float *fparam;
{
   float dt2, dtsq2;
   int i;
   float *fx, *fy, *fz;

   float dt = fparam[2];

   fx = scratch;
   fy = &scratch[natm];
   fz = &scratch[2*natm];

   dt2 = dt*0.5;
   dtsq2 = dt*dt2;

   for(i=1;i<=natm;i++){
      rx[i-1] = rx[i-1] + dt*vx[i-1] + dtsq2*fx[i-1];
      ry[i-1] = ry[i-1] + dt*vy[i-1] + dtsq2*fy[i-1];
      rz[i-1] = rz[i-1] + dt*vz[i-1] + dtsq2*fz[i-1];
      vx[i-1] = vx[i-1] + dt2*fx[i-1];
      vy[i-1] = vy[i-1] + dt2*fy[i-1];
      vz[i-1] = vz[i-1] + dt2*fz[i-1];
   }
}
