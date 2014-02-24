
#include <math.h>
#include <stdio.h>

#include "moldyn.h"

void force (float *pval, float *vval, float *rx, float *ry, float *rz, float *fx, float *fy, float *fz, float sigma, float rcut, float vrcut, float dvrc12, float dvrcut, int *head, int *list, int *map, int mx, int my, int mz, int natoms, int step)
{
   float sigsq, rcutsq;
   float rxi, ryi, rzi, fxi, fyi, fzi;
   float rxij, ryij, rzij, rijsq;
   float rij, sr2, sr6, vij, wij, fij, fxij, fyij, fzij;
   float potential, virial;
   int i, icell, j, jcell0, jcell, nabor;
   int xi, yi, zi, ix, jx, kx, icount, xcell, ycell, zcell;

   sigsq  = sigma*sigma;
   rcutsq = rcut*rcut;

   for(i=0;i<natoms;++i){
      *(fx+i) = 0.0;
      *(fy+i) = 0.0;
      *(fz+i) = 0.0;
   }

   potential = 0.0;
   virial    = 0.0;
   
   icount = 0;
   for(icell=0;icell<(mx+2)*(my+2)*(mz+2);icell++){
       xi = icell%(mx+2);
       yi = (icell/(mx+2))%(my+2);
       zi = icell/((mx+2)*(my+2));
//       if((xi>0 && xi <(mx+1))&&(yi>0 && yi<(my+1))&&(zi>0 && zi<(mz+1))){
           i = head[icell];
           while (i>=0) {
               rxi = rx[i];
               ryi = ry[i];
               rzi = rz[i];
               if (i < natoms) {
                   fxi = fx[i];
                   fyi = fy[i];
                   fzi = fz[i];
                   j = list[i];
                   while (j>=0) {
                       rxij = rxi - rx[j];
                       ryij = ryi - ry[j];
                       rzij = rzi - rz[j];
                       rijsq = rxij*rxij + ryij*ryij + rzij*rzij;
                       if (rijsq < rcutsq) {
	                   force_ij(rijsq, rxij, ryij, rzij, sigsq, vrcut, dvrc12, rcut, dvrcut, &vij, &wij, &fxij, &fyij, &fzij);
                           potential += vij;
                           virial    += wij;
                           fxi       += fxij;
                           fyi       += fyij;
                           fzi       += fzij;
                           *(fx+j) -= fxij;
                           *(fy+j) -= fyij;
                           *(fz+j) -= fzij;
                       }           
                       j = list[j];
                   }
		   icount++;
               }
//	       printf("\nCell %d at (%d,%d,%d) interacts with cells: ",icell,xi,yi,zi);
               for (ix=-1;ix<=1;ix++)
                   for (jx=-1;jx<=1;jx++)
                       for (kx=-1;kx<=1;kx++){
	                   jcell0 = ix+1+3*(jx+1+3*(kx+1));
	                   if(map[jcell0]){
                               xcell = (ix+xi+mx+2)%(mx+2);
                               ycell = (jx+yi+my+2)%(my+2);
                               zcell = (kx+zi+mz+2)%(mz+2);
                               jcell = xcell + (mx+2)*(ycell+(my+2)*zcell);
			//       printf("%d (%d,%d,%d); ",jcell,xcell,ycell,zcell);
			       if(icell==jcell) printf("\nShould not be processing this cell!");
                               j = head[jcell];
                               if (i<natoms || j<natoms) {
                                   while (j>=0) {
                                       rxij = rxi - rx[j];
                                       ryij = ryi - ry[j];
                                       rzij = rzi - rz[j];
                                       rijsq = rxij*rxij + ryij*ryij + rzij*rzij;
                                       if (rijsq < rcutsq) {
					   force_ij(rijsq, rxij, ryij, rzij, sigsq, vrcut, dvrc12, rcut, dvrcut, &vij, &wij, &fxij, &fyij, &fzij);
                                           wij *= 0.5;
                                           vij *= 0.5;
                                           if (i<natoms) {
                                               potential += vij;
                                               virial    += wij;
                                               fxi       += fxij;
                                               fyi       += fyij;
                                               fzi       += fzij;
                                           }
                                           if (j<natoms) {
                                               potential += vij;
                                               virial    += wij;
                                               *(fx+j) -= fxij;
                                               *(fy+j) -= fyij;
                                               *(fz+j) -= fzij;
                                           }
                                       }
                                       j = list[j];
                                   }
                               }
                           }
		       }
               if (i<natoms) {
                    *(fx+i) = fxi;
                    *(fy+i) = fyi;
                    *(fz+i) = fzi;
               }
               i = list[i];
	   }
       //}
   }
   if(icount!=natoms) printf("\nProcessed %d particles in force routine instead of %d",icount,natoms);
   potential *= 4.0;
   virial    *= 48.0/3.0;
   *pval = potential;
   *vval = virial;

   for (i=0;i<natoms;++i) {
      *(fx+i) *= 48.0;
      *(fy+i) *= 48.0;
      *(fz+i) *= 48.0;
   }
}
