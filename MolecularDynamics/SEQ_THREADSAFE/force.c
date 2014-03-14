
#include <math.h>
#include <stdio.h>

#include "moldyn.h"

#define BLOCK_SIZE 512

void force (float *pval, float *vval, float *rx, float *ry, float *rz, float *fx, float *fy, float *fz, float sigma, float rcut, float vrcut, float dvrc12, float dvrcut, int *head, int *list, int mx, int my, int mz, int natoms, int step, float sfx, float sfy, float sfz)
{
   float sigsq, rcutsq;
   float rxi, ryi, rzi, fxi, fyi, fzi;
   float rxij, ryij, rzij, rijsq;
   float rij, sr2, sr6, vij, wij, fij, fxij, fyij, fzij;
   float potential, virial;
   int i, icell, j, jcell, nabor;
   int xi, yi, zi, ix, jx, kx, xcell, ycell, zcell;
   float valp, valv;

   sigsq  = sigma*sigma;
   rcutsq = rcut*rcut;

   potential = 0.0;
   virial    = 0.0;
   valp = 0.0;
   valv = 0.0;
   
   for(i=0;i<natoms;++i){
	 rxi = rx[i];
	 ryi = ry[i];
	 rzi = rz[i];
	 fxi = 0.0;
	 fyi = 0.0;
	 fzi = 0.0;
	 xi = (int)((rxi+0.5)/sfx) + 1;
	 yi = (int)((ryi+0.5)/sfy) + 1;
	 zi = (int)((rzi+0.5)/sfz) + 1;
         if(xi > mx) xi = mx;
         if(yi > my) yi = my;
         if(zi > mz) zi = mz;
	 icell = xi + (mx+2)*(yi+zi*(my+2));
//	 if (xi<0 || xi> mx) printf("\nxi = %d\n",xi);
//	 if (yi<0 || yi> my) printf("\nyi = %d\n",yi);
//	 if (zi<0 || zi> mz) printf("\nzi = %d\n",zi);
//	 if(icell<0||icell>(mx+2)*(my+2)*(mz+2)-1) printf("\nicell = %d\n",icell);
//	if(step==92&&i==4680){ printf("Particle %5d, (xi,yi,zi) = %d,%d,%d, icell = %5d\n",i,xi,yi,zi,icell);
//		printf("rx = %f, ry = %f, rz = %f\n",rxi,ryi,rzi);
//		fflush(stdout);
//	}
         for (ix=-1;ix<=1;ix++)
             for (jx=-1;jx<=1;jx++)
                 for (kx=-1;kx<=1;kx++){
		     xcell = ix+xi;
		     ycell = jx+yi;
		     zcell = kx+zi;
                     jcell = xcell + (mx+2)*(ycell+(my+2)*zcell);
//	printf("%d (%d,%d,%d); ",jcell,xcell,ycell,zcell);
                     j = head[jcell];
//	 if(jcell<0||jcell>(mx+2)*(my+2)*(mz+2)-1) printf("\njcell = %d\n",jcell);
                     while (j>=0) {
//			 if(j<0 || j>ntot-1) printf("\nj = %d\n",j);
                         if (j!=i) {
                             rxij = rxi - rx[j];
                             ryij = ryi - ry[j];
                             rzij = rzi - rz[j];
                             rijsq = rxij*rxij + ryij*ryij + rzij*rzij;
                             if (rijsq < rcutsq) {
			         force_ij(rijsq, rxij, ryij, rzij, sigsq, vrcut, dvrc12, rcut, dvrcut, &vij, &wij, &fxij, &fyij, &fzij);
                                 wij *= 0.5;
                                 vij *= 0.5;
 //                                potential += vij;
  //                               virial    += wij;
                                 fxi       += fxij;
                                 fyi       += fyij;
                                 fzi       += fzij;
				 valp += vij;
				 valv += wij;
                             }
			 }
                         j = list[j];
                      }
	         }
         *(fx+i) = 48.0*fxi;
         *(fy+i) = 48.0*fyi;
         *(fz+i) = 48.0*fzi;
	 if((i+1)%BLOCK_SIZE==0){
	     potential += valp;
	     virial += valv;
	     valp = valv = 0.0;
	 }
//	 vpArray[i] = 4.0*valp;
//	 vpArray[i+natoms] = 48.0*valv/3.0;
   }
   potential *= 4.0;
   virial    *= 48.0/3.0;
   *pval = potential;
   *vval = virial;
}
