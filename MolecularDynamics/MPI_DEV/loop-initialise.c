#include "moldyn.h"
#include <stdio.h>

void loop_initialise (rank,iparam,fparam)
int rank;
int *iparam;
float *fparam;
{
   float ace, acv, ack, acp, acesq, acvsq, acksq, acpsq;

   float sigma = fparam[0];
   float rcut  = fparam[1];
   float dt    = fparam[2];

   int nstart = iparam[7];

   if (nstart <= 4) {
      ace = 0.0;
      acv = 0.0;
      ack = 0.0;
      acp = 0.0;
      acesq = 0.0;
      acvsq = 0.0;
      acksq = 0.0;
      acpsq = 0.0;
   }
   fparam[23] = ace;
   fparam[24] = acv;
   fparam[25] = ack;
   fparam[26] = acp;
   fparam[27] = acesq;
   fparam[28] = acvsq;
   fparam[29] = acksq;
   fparam[30] = acpsq;
   
   if (I_AM_HOST) {
      printf ("\n SIGMA/BOX              =  %10.4f",sigma);
      printf ("\n RCUT/BOX               =  %10.4f",rcut);
      printf ("\n DT                     =  %10.4f",dt);
      printf ("\n ** MOLECULAR DYNAMICS BEGINS ** \n\n\n");
      printf ("\n TIMESTEP  ..ENERGY..  ..KINETIC.  ..POTENT..  .PRESSURE.  ..TEMPER..  ");
   }
}
