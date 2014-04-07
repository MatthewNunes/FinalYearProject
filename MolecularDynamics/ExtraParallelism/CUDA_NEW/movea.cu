#include "moldyn.h"
#include <cuda.h>
#define CHUNK_SIZE 16
__global__ void movea (float *rx, float *ry, float *rz, float *vx, float *vy, float *vz, float *fx, float *fy, float *fz, float dt, int natoms)
{
   float dt2, dtsq2;
   int element = blockDim.x * blockIdx.x + threadIdx.x;
   dt2 = dt*0.5;
   dtsq2 = dt*dt2;
   float fxElement;
   float fyElement;
   float fzElement;

   if (element < natoms)
   {
      fxElement = fx[element];
      fyElement = fy[element];
      fzElement = fz[element];
      rx[element] = rx[element] + dt*vx[element] + dtsq2*fxElement;
      ry[element] = ry[element] + dt*vy[element] + dtsq2*fyElement;
      rz[element] = rz[element] + dt*vz[element] + dtsq2*fzElement;
      vx[element] = vx[element] + dt2*fxElement;
      vy[element] = vy[element] + dt2*fyElement;
      vz[element] = vz[element] + dt2*fzElement;
   }
}
