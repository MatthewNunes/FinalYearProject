#include <stdio.h>
#include "moldyn.h"

void initialise_particles (float rx[], float ry[], float rz[], float vx[], float vy[], float vz[], int nc)
{
   int is = 2215;
   int lz, ly, lx, natm;
   float rz1, rz2, rz3, rz4, ry1, ry2, ry3, ry4, rx1, vvx, vvy, vvz;

   natm = 0;
   for (lz=0;lz<nc;lz++){
      rz1 = -0.5 + (float)lz/(float)nc+ 0.000001;
      rz2 = -0.5 + ((float)lz+0.5)/(float)nc - 0.000001;
      rz3 = -0.5 + (float)lz/(float)nc+ 0.000001;
      rz4 = -0.5 + ((float)lz+0.5)/(float)nc - 0.000001;
      for(ly=0;ly<nc;ly++){
         ry1 = -0.5 + (float)ly/(float)nc + 0.000001;
         ry2 = -0.5 + (float)ly/(float)nc + 0.000001;
         ry3 = -0.5 + ((float)ly+0.5)/(float)nc - 0.000001;
         ry4 = -0.5 + ((float)ly+0.5)/(float)nc - 0.000001;
         for(lx=0;lx<nc;lx++){
            rx1 = -0.5 + (float)lx/(float)nc + 0.000001;
            pseudorand (&is, &vvx);
            pseudorand (&is, &vvy);
            pseudorand (&is, &vvz);
	    rx[natm] = rx1;
	    ry[natm] = ry1;
	    rz[natm] = rz1;
            vx[natm] = vvx;
	    vy[natm] = vvy;
	    vz[natm] = vvz;
	    natm++;
            rx1 = -0.5 + ((float)lx+0.5)/(float)nc - 0.000001;
	    rx[natm] = rx1;
	    ry[natm] = ry2;
	    rz[natm] = rz2;
            vx[natm] = -vvx;
	    vy[natm] = -vvy;
	    vz[natm] = -vvz;
	    natm++;
            rx1 = -0.5 + ((float)lx+0.5)/(float)nc - 0.000001;
            pseudorand (&is, &vvx);
            pseudorand (&is, &vvy);
            pseudorand (&is, &vvz);
	    rx[natm] = rx1;
	    ry[natm] = ry3;
	    rz[natm] = rz3;
            vx[natm] = vvx;
	    vy[natm] = vvy;
	    vz[natm] = vvz;
	    natm++;
            rx1 = -0.5 + (float)lx/(float)nc + 0.000001;
	    rx[natm] = rx1;
	    ry[natm] = ry4;
	    rz[natm] = rz4;
            vx[natm] = -vvx;
	    vy[natm] = -vvy;
	    vz[natm] = -vvz;
	    natm++;
         }
      }
   }

   if (natm!=4*nc*nc*nc) 
      printf ("\n Created %8d particles; should have %8d\n",natm, 4*nc*nc*nc);
}
