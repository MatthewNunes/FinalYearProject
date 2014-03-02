#include "moldyn.h"
#include <stdio.h>

void loop_initialise(float *ace, float *acv, float *ack, float *acp, float *acesq, float *acvsq, float *acksq, float *acpsq, float sigma, float rcut, float dt)
{
      *ace = 0.0;
      *acv = 0.0;
      *ack = 0.0;
      *acp = 0.0;
      *acesq = 0.0;
      *acvsq = 0.0;
      *acksq = 0.0;
      *acpsq = 0.0;
   
      printf ("\n SIGMA/BOX              =  %10.4f",sigma);
      printf ("\n RCUT/BOX               =  %10.4f",rcut);
      printf ("\n DT                     =  %10.4f",dt);
      printf ("\n ** MOLECULAR DYNAMICS BEGINS ** \n\n\n");
      printf ("\n TIMESTEP  ..ENERGY..  ..KINETIC.  ..POTENT..  .PRESSURE.  ..TEMPER..  ");
}
