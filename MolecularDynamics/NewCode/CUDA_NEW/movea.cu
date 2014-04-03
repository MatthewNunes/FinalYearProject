#include "moldyn.h"
#include <cuda.h>
__global__ void movea (float *rx, float *ry, float *rz, float *vx, float *vy, float *vz, float *fx, float *fy, float *fz, float dt, int natoms)
{
   float dt2, dtsq2;
   int i;
   int element = blockDim.x * blockIdx.x + threadIdx.x;
   dt2 = dt*0.5;
   dtsq2 = dt*dt2;

   if (element < natoms)
   {
      rx[element] = rx[element] + dt*vx[element] + dtsq2*fx[element];
      ry[element] = ry[element] + dt*vy[element] + dtsq2*fy[element];
      rz[element] = rz[element] + dt*vz[element] + dtsq2*fz[element];
      vx[element] = vx[element] + dt2*fx[element];
      vy[element] = vy[element] + dt2*fy[element];
      vz[element] = vz[element] + dt2*fz[element];
   }
}
