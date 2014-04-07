#include "moldyn.h"
#include <cuda.h>

__global__ void moveb (float *kineticArray, float *vx, float *vy, float *vz, float *fx, float *fy, float *fz, float dt, int natoms)
{
   float dt2;
   int element = blockDim.x * blockIdx.x + threadIdx.x;
   dt2 = dt*0.5;
  // *kinetic = 0.0;
   __shared__ float kineticSum[BLOCK_WIDTH];
   kineticSum[threadIdx.x] = 0.0;
   float vxElement;
   float vyElement;
   float vzElement;
   if(element < natoms)
   {
      vxElement = vx[element];
      vyElement = vy[element];
      vzElement = vz[element];
      vxElement = vxElement + dt2*fx[element];
      vyElement = vyElement + dt2*fy[element];
      vzElement = vzElement + dt2*fz[element];
      kineticSum[threadIdx.x] += vxElement*vxElement + vyElement*vyElement + vzElement*vzElement;
      vx[element] = vxElement;
      vy[element] = vyElement;
      vz[element] = vzElement;
   }
   int stride;
   for (stride = blockDim.x/2; stride > 0; stride >>=1)
   {
      __syncthreads();
      if (threadIdx.x < stride)
      {
         kineticSum[threadIdx.x] += kineticSum[threadIdx.x + stride];
      }
   }
   __syncthreads();
   if (threadIdx.x == 0)
   {   
      kineticArray[blockIdx.x] = kineticSum[0];
   }
}


