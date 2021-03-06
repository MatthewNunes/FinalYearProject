
#include <math.h>
#include <stdio.h>

#include "moldyn.h"

#define BLOCK_SIZE 512

void force (float *pval, float *vval, float *rx, float *ry, float *rz, float *fx, float *fy, float *fz, float sigma, float rcut, float vrcut, float dvrc12, float dvrcut, int *head, int *list, int *map, int mx, int my, int mz, int natoms, int step)
{
   float sigsq, rcutsq;
   float rxi, ryi, rzi, fxi, fyi, fzi;
   float rxij, ryij, rzij, rijsq;
   float rij, sr2, sr6, vij, wij, fij, fxij, fyij, fzij;
   float potential, virial;
   int i, icell, j, jcell, nabor;
   int xi, yi, zi, ix, jx, kx, icount, xcell, ycell, zcell;
   float valv, valp;

   sigsq  = sigma*sigma;
   rcutsq = rcut*rcut;

//   for(i=0;i<natoms;++i){
//      *(fx+i) = 0.0;
//      *(fy+i) = 0.0;
//      *(fz+i) = 0.0;
//   }

   potential = 0.0;
   virial    = 0.0;
   valv = 0.0;
   valp = 0.0;
   
   icount = 0;
   for(icell=0;icell<(mx+2)*(my+2)*(mz+2);icell++){
       xi = icell%(mx+2);
       yi = (icell/(mx+2))%(my+2);
       zi = icell/((mx+2)*(my+2));
       if((xi>0 && xi <(mx+1))&&(yi>0 && yi<(my+1))&&(zi>0 && zi<(mz+1))){
           i = head[icell];
           while (i>=0) {
               rxi = rx[i];
               ryi = ry[i];
               rzi = rz[i];
//	 printf("Particle %5d, (xi,yi,zi) = %d,%d,%d, icel = %5d\n",i,xi,yi,zi,icell);
//               fxi = fx[i];
 //              fyi = fy[i];
  //             fzi = fz[i];
               fxi = fyi = fzi = 0.0;
               j = head[icell];
               while (j>=0) {
                   rxij = rxi - rx[j];
                   ryij = ryi - ry[j];
                   rzij = rzi - rz[j];
                   rijsq = rxij*rxij + ryij*ryij + rzij*rzij;
                   if (rijsq < rcutsq && j!=i) {
	               force_ij(rijsq, rxij, ryij, rzij, sigsq, vrcut, dvrc12, rcut, dvrcut, &vij, &wij, &fxij, &fyij, &fzij);
		       vij *= 0.5;
		       wij *= 0.5;
		       valp += vij;
		       valv += wij;
//                       potential += 0.5*vij;
//                       virial    += 0.5*wij;
                       fxi       += fxij;
                       fyi       += fyij;
                       fzi       += fzij;
                   }           
                   j = list[j];
               }
	       icount++;
//	       printf("\nCell %d at (%d,%d,%d) interacts with cells: ",icell,xi,yi,zi);
               for (ix=-1;ix<=1;ix++)
                   for (jx=-1;jx<=1;jx++)
                       for (kx=-1;kx<=1;kx++){
                           xcell = ix+xi;
                           ycell = jx+yi;
                           zcell = kx+zi;
                           jcell = xcell + (mx+2)*(ycell+(my+2)*zcell);
			//       printf("%d (%d,%d,%d); ",jcell,xcell,ycell,zcell);
		           if(icell!=jcell) {
                                printf("jcell: %d\n", jcell);
                               j = head[jcell];
                               while (j>=0) {
                                   rxij = rxi - rx[j];
                                   ryij = ryi - ry[j];
                                   rzij = rzi - rz[j];
                                   rijsq = rxij*rxij + ryij*ryij + rzij*rzij;
                                   if (rijsq < rcutsq) {
			               force_ij(rijsq, rxij, ryij, rzij, sigsq, vrcut, dvrc12, rcut, dvrcut, &vij, &wij, &fxij, &fyij, &fzij);
                                       wij *= 0.5;
                                       vij *= 0.5;
				       valp += vij;
				       valv += wij;
//                                       potential += vij;
 //                                      virial    += wij;
                                       fxi       += fxij;
                                       fyi       += fyij;
                                       fzi       += fzij;
                                   }
                                   j = list[j];
			      }
                           }
		       }
               *(fx+i) = 48.0*fxi;
               *(fy+i) = 48.0*fyi;
               *(fz+i) = 48.0*fzi;
               i = list[i];
               if((icount+1)%BLOCK_SIZE==0){
	           potential += valp;
	           virial += valv;
	           valp = valv = 0.0;
               }
	   }
       }
   }
   if(icount!=natoms) printf("\nProcessed %d particles in force routine instead of %d",icount,natoms);
   potential *= 4.0;
   virial    *= 48.0/3.0;
   *pval = potential;
   *vval = virial;

//   for (i=0;i<natoms;++i) {
//      *(fx+i) *= 48.0;
//      *(fy+i) *= 48.0;
//      *(fz+i) *= 48.0;
//   }
}
