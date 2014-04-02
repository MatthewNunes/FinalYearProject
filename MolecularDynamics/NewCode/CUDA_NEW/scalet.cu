#include "moldyn.h"
#include <math.h>

__global__ void scalet ( float *vx, float *vy, float *vz, float kinetic, float eqtemp, float tmpx, int iscale, int natoms, int step)
{
   int i;
   int element = blockIdx.x * blockDim.x + threadIdx.x;
   float scalef;

   if (step%iscale==0) scalef = sqrt((double)(eqtemp/tmpx));
   else scalef = sqrt ((double)(eqtemp/(2.0*kinetic/(3.0*(float)(natoms-1)))));

   if(element < natoms){
      vx[element] *= scalef;
      vy[element] *= scalef;
      vz[element] *= scalef;
   }
}
