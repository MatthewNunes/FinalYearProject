#include "moldyn.h"

void movea (float *rx, float *ry, float *rz, float *vx, float *vy, float *vz, float *fx, float *fy, float *fz, float dt, int natoms)
{
   float dt2, dtsq2;
   int i;

   dt2 = dt*0.5;
   dtsq2 = dt*dt2;

   for(i=0;i<natoms;i++){
      rx[i] = rx[i] + dt*vx[i] + dtsq2*fx[i];
      ry[i] = ry[i] + dt*vy[i] + dtsq2*fy[i];
      rz[i] = rz[i] + dt*vz[i] + dtsq2*fz[i];
      vx[i] = vx[i] + dt2*fx[i];
      vy[i] = vy[i] + dt2*fy[i];
      vz[i] = vz[i] + dt2*fz[i];
   }
}
