#include "moldyn.h"
#include <cuda.h>

__global__ void moveb (float *kineticArray, float *vx, float *vy, float *vz, float *fx, float *fy, float *fz, float dt, int natoms)
{
   float dt2;
   int i;
   int element = blockDim.x * blockIdx.x + threadIdx.x;
   float privateKinetic = 0.0;
   dt2 = dt*0.5;
  // *kinetic = 0.0;
   __shared__ float kineticSum[BLOCK_WIDTH];
   kineticSum[threadIdx.x] = 0.0;
   if(element < natoms)
   {
      vx[element] = vx[element] + dt2*fx[element];
      vy[element] = vy[element] + dt2*fy[element];
      vz[element] = vz[element] + dt2*fz[element];
      kineticSum[threadIdx.x] += vx[element]*vx[element] + vy[element]*vy[element] + vz[element]*vz[element];
   }
   int stride;
   for (stride = blockDim.x/2; stride > 0; stride >>=1)
   {
      if (threadIdx.x < stride)
      {
         kineticSum[threadIdx.x] += kineticSum[threadIdx.x + stride];
      }
   }
   if (threadIdx.x == 0)
   {   
      kineticArray[blockIdx.x] = kineticSum[0];
   }
}


