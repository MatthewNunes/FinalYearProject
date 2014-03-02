#include <math.h>

void scalet (kinetic, vx, vy, vz, natm, step, iparam, fparam)
float kinetic;
float vx[];
float vy[];
float vz[];
int natm;
int *iparam;
float *fparam;
{
   float scalef;
   int i;

   int iscale = iparam[6];

   float eqtemp = fparam[3];
   float tmpx   = fparam[22];

   if (step%iscale==0) scalef = sqrt((double)(eqtemp/tmpx));
   else scalef = sqrt ((double)(eqtemp/(2.0*kinetic/(3.0*(float)(natm-1)))));

   for(i=0;i<natm;i++){
      vx[i] *= scalef;
      vy[i] *= scalef;
      vz[i] *= scalef;
   }
}
