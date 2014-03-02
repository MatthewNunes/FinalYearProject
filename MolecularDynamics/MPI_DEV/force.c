#include "moldyn.h"
#include <math.h>

void force (potential, virial, rx, ry, rz, head, list, natm, map, scratch,iparam,fparam)
float *potential;
float *virial;
float rx[];
float ry[];
float rz[];
int head[];
int list[];
int natm;
int *map;
float *scratch;
int *iparam;
float *fparam;
{
   float sigsq, rcutsq;
   float rxi, ryi, rzi, fxi, fyi, fzi;
   float rxij, ryij, rzij, rijsq;
   float rij, sr2, sr6, vij, wij, fij, fxij, fyij, fzij;
   int i, icell, j, jcell0, jcell, nabor;
   int ix, jx, kx, xi, yi, zi, xcell, ycell, zcell, offset;
   float *fx, *fy, *fz;

   int mx = iparam[13];
   int my = iparam[14];
   int mz = iparam[15];

   float sigma = fparam[0];
   float rcut   = fparam[1];
   float vrcut  = fparam[16];
   float dvrcut = fparam[17];
   float dvrc12 = fparam[18];

   fx = scratch;
   fy = &scratch[natm];
   fz = &scratch[2*natm];

//   offset = xid+(yid+zid*my)*mx;
   offset = 0;
   sigsq  = sigma*sigma;
   rcutsq = rcut*rcut;

   for(i=0;i<natm;++i){
      *(fx+i) = 0.0;
      *(fy+i) = 0.0;
      *(fz+i) = 0.0;
   }

   *potential = 0.0;
   *virial    = 0.0;
   
   for(icell=1;icell<=mx*my*mz;icell++){
      i = head[icell-1];
      xi = (icell-1)%mx;
      yi = ((icell-1)/mx)%my;
      zi = ((icell-1)/(mx*my))%mz;
      while (i>0) {
         rxi = rx[i-1];
         ryi = ry[i-1];
         rzi = rz[i-1];
         if (i <= natm) {
            fxi = fx[i-1];
            fyi = fy[i-1];
            fzi = fz[i-1];
            j = list[i-1];
            while (j>0) {
               rxij = rxi - rx[j-1];
               ryij = ryi - ry[j-1];
               rzij = rzi - rz[j-1];
               rijsq = rxij*rxij + ryij*ryij + rzij*rzij;
               if (rijsq < rcutsq) {
                  rij = (float) sqrt ((double)rijsq);
                  sr2 = sigsq/rijsq;
                  sr6 = sr2*sr2*sr2;
                  vij = sr6*(sr6-1.0) - vrcut - dvrc12*(rij-rcut);
                  wij = sr6*(sr6-0.5) + dvrcut*rij;
                  *potential += vij;
                  *virial    += wij;
                  fij        = wij/rijsq;
                  fxij       = fij*rxij;
                  fyij       = fij*ryij;
                  fzij       = fij*rzij;
                  fxi       += fxij;
                  fyi       += fyij;
                  fzi       += fzij;
                  *(fx+j-1) -= fxij;
                  *(fy+j-1) -= fyij;
                  *(fz+j-1) -= fzij;
               }      
               j = list[j-1];
            }
         }
         for(ix=-1;ix<=1;ix++)
             for(jx=-1;jx<=1;jx++)
                 for(kx=-1;kx<=1;kx++){
                     jcell0 = ix+1+3*(jx+1+3*(kx+1));
                     if(map[jcell0]){
                         xcell = (ix+xi+mx)%(mx);
                         ycell = (jx+yi+my)%(my);
                         zcell = (kx+zi+mz)%(mz);
                         jcell = 1 + xcell + (mx)*(ycell+(my)*zcell);
                         j = head[jcell-1];
//    if(icell<=200&&rank==5) printf("rank = %d, icell = %d, jcell = %d, j = %d\n",rank,icell,jcell,j);
            if (i<=natm || j<=natm) {
               while (j!=0) {
                  rxij = rxi - rx[j-1];
                  ryij = ryi - ry[j-1];
                  rzij = rzi - rz[j-1];
                  rijsq = rxij*rxij + ryij*ryij + rzij*rzij;
                  if (rijsq < rcutsq) {
                     rij = (float) sqrt ((double)rijsq);
                     sr2 = sigsq/rijsq;
                     sr6 = sr2*sr2*sr2;
                     vij = (sr6*(sr6-1.0) - vrcut - dvrc12*(rij-rcut))*0.5;
                     wij = sr6*(sr6-0.5) + dvrcut*rij;
                     fij        = wij/rijsq;
                     wij *= 0.5;
                     fxij       = fij*rxij;
                     fyij       = fij*ryij;
                     fzij       = fij*rzij;
                     if (i<=natm) {
                        *potential += vij;
                        *virial    += wij;
                        fxi       += fxij;
                        fyi       += fyij;
                        fzi       += fzij;
                     }
                     if (j<=natm) {
                        *potential += vij;
                        *virial    += wij;
                        *(fx+j-1) -= fxij;
                        *(fy+j-1) -= fyij;
                        *(fz+j-1) -= fzij;
                     }
                  }
                  j = list[j-1];
               }
            }
            }
         }
         if (i<=natm) {
            *(fx+i-1) = fxi;
            *(fy+i-1) = fyi;
            *(fz+i-1) = fzi;
         }
         i = list[i-1];
      }
   }
   *potential *= 4.0;
   *virial    *= 48.0/3.0;

   for (i=1;i<=natm;++i) {
      *(fx+i-1) *= 48.0;
      *(fy+i-1) *= 48.0;
      *(fz+i-1) *= 48.0;
   }
}
