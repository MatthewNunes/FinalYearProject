#include "moldyn.h"

void moveb (float *kinetic, float *vx, float *vy, float *vz, float *fx, float *fy, float *fz, float dt, int natoms)
{
   float dt2;
   int i;

   dt2 = dt*0.5;
   *kinetic = 0.0;
   for(i=0;i<natoms;i++){
      vx[i] = vx[i] + dt2*fx[i];
      vy[i] = vy[i] + dt2*fy[i];
      vz[i] = vz[i] + dt2*fz[i];
      *kinetic += vx[i]*vx[i] + vy[i]*vy[i] + vz[i]*vz[i];
   }
   *kinetic *= 0.5;
}
