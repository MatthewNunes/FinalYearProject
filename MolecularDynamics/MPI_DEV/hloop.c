#include "moldyn.h"
#include <stdio.h>

void hloop (kinetic, step, rank, vx, vy, vz, natm, iparam, fparam)
float kinetic;
int step;
int rank;
float vx[];
float vy[];
float vz[];
int natm;
int *iparam;
float *fparam;
{
   float tmpx;
   float e, en, vn, kn, pres;

   int nequil = iparam[4];
   int natoms = iparam[11];
   int iprint = iparam[12];

   float sigma = fparam[0];
   float dens  = fparam[4];
   float freex = fparam[5];
   float vg    = fparam[19];
   float wg    = fparam[20];
   float kg    = fparam[21];
   float ace   = fparam[23];
   float acv   = fparam[24];
   float ack   = fparam[25];
   float acp   = fparam[26];
   float acesq = fparam[27];
   float acvsq = fparam[28];
   float acksq = fparam[29];
   float acpsq = fparam[30];

   e = kg + vg;
   en = e/(float)natoms;
   vn = vg/(float)natoms;
   kn = kg/(float)natoms;
   tmpx = 2.0*kg/freex;
   pres = dens*tmpx + wg;
   pres = pres*sigma*sigma*sigma;
   
   if (step>nequil) {
      ace += en;
      acv += vn;
      ack += kn;
      acp += pres;
      acesq += en*en;
      acvsq += vn*vn;
      acksq += kn*kn;
      acpsq += pres*pres;
   }
   fparam[22] = tmpx;
   fparam[23] = ace;
   fparam[24] = acv;
   fparam[25] = ack;
   fparam[26] = acp;
   fparam[27] = acesq;
   fparam[28] = acvsq;
   fparam[29] = acksq;
   fparam[30] = acpsq;

/* If still equilibrating call subroutine to scale velocities */

   if (nequil > step) scalet (kinetic, vx, vy, vz, natm, step, iparam, fparam);

/* Optionally print information */
   if (I_AM_HOST) {
      if (step%iprint == 0)
         printf("\n%8d%12.6f%12.6f%12.6f%12.6f%12.6f",step, en, kn, vn,
                 pres, tmpx);
   }
}
