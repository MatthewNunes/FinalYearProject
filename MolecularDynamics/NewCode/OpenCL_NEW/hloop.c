#include "moldyn.h"
#include <stdio.h>

void hloop (float kinetic, int step, float vg, float wg, float kg, float freex, float dens, float sigma, float eqtemp, float *tmpx, float *ace, float *acv, float *ack, float *acp, float *acesq, float *acvsq, float *acksq, float *acpsq, float *vx, float *vy, float *vz, int iscale, int iprint, int nequil, int natoms)
{
   float e, en, vn, kn, pres;

   e = kg + vg;
   en = e/(float)natoms;
   vn = vg/(float)natoms;
   kn = kg/(float)natoms;
   *tmpx = 2.0*kg/freex;
   pres = dens*(*tmpx) + wg;
   pres = pres*sigma*sigma*sigma;
   
   if (step>nequil) {
      *ace += en;
      *acv += vn;
      *ack += kn;
      *acp += pres;
      *acesq += en*en;
      *acvsq += vn*vn;
      *acksq += kn*kn;
      *acpsq += pres*pres;
   }

/* If still equilibrating call subroutine to scale velocities */

   if (nequil > step) scalet (vx, vy, vz, kinetic, eqtemp, *tmpx, iscale, natoms, step);

/* Optionally print information */
      if (step%iprint == 0)
         printf("\n%8d%12.6f%12.6f%12.6f%12.6f%12.6f",step, en, kn, vn,
                 pres, *tmpx);
}
