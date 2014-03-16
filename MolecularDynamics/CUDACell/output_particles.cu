#include <stdio.h>
#include "moldyn.h"

void output_particles(float *rx, float *ry, float *rz, float *vx, float *vy, float *vz, float *fx, float *fy, float *fz,  int natm)
{
   int i;

   for(i=0;i<natm;i++){
	   printf("\n%d: (rx,ry,rz) = (%f,%f,%f), (vx,vy,vz) = (%f,%f,%f), (fx,fy,fz) = (%f,%f,%f)",i,rx[i],ry[i],rz[i],vx[i],vy[i],vz[i],fx[i],fy[i],fz[i]);
   }
}
