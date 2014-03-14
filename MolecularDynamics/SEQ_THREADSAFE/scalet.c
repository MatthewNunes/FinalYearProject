#include <math.h>

void scalet ( float *vx, float *vy, float *vz, float kinetic, float eqtemp, float tmpx, int iscale, int natoms, int step)
{
   int i;
   float scalef;

   if (step%iscale==0) scalef = sqrt((double)(eqtemp/tmpx));
   else scalef = sqrt ((double)(eqtemp/(2.0*kinetic/(3.0*(float)(natoms-1)))));

   for(i=0;i<natoms;i++){
      vx[i] *= scalef;
      vy[i] *= scalef;
      vz[i] *= scalef;
   }
}
