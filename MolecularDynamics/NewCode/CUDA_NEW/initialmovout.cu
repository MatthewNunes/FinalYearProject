#include "moldyn.h"
#include <cuda.h> 
__global__ void initialMovout(float *rx, float *ry, float *rz, int natoms)
{
   int element = blockDim.x * blockIdx.x + threadIdx.x;
   if(element < natoms)
   {
      //printf("\ni = %d",i);
      if(rx[element] < -0.5){ rx[element] += 1.0;}
      if(rx[element] >  0.5){ rx[element] -= 1.0;}
      if(ry[element] < -0.5){ ry[element] += 1.0;}
      if(ry[element] >  0.5){ ry[element] -= 1.0;}
      if(rz[element] < -0.5){ rz[element] += 1.0;}
      if(rz[element] >  0.5){ rz[element] -= 1.0;}
   }
}