#include <stdio.h>
#include "moldyn.h"

void movout (float *rx, float *ry, float *rz, float *vx, float *vy, float *vz, float sfx, float sfy, float sfz, int *head, int *list, int mx, int my, int mz, int natoms, int *ntot)
{
   int    i, j, k, icell, xi, yi, zi;
   int src, dst, scell, dcell, p, pptr;

   int ncells = (mx+2)*(my+2)*(mz+2);
//   printf("\nStarting movout: mx = %d, my = %d, mz = %d, natoms = %d, sfx = %f, sfy = %f, sfz = %f\n",mx, my, mz, natoms, sfx, sfy, sfz);
   for (icell=0;icell<ncells;++icell) head[icell] = -1;

 //  printf("\nStarting for loop in movout: mx = %d, my = %d, mz = %d, natoms = %d, sfx = %f, sfy = %f, sfz = %f\n",mx, my, mz, natoms, sfx, sfy, sfz);
   for(i=0;i<natoms;i++){
      //printf("\ni = %d",i);
      if(rx[i] < -0.5){ rx[i] += 1.0;}
      if(rx[i] >  0.5){ rx[i] -= 1.0;}
      if(ry[i] < -0.5){ ry[i] += 1.0;}
      if(ry[i] >  0.5){ ry[i] -= 1.0;}
      if(rz[i] < -0.5){ rz[i] += 1.0;}
      if(rz[i] >  0.5){ rz[i] -= 1.0;}
      if (rx[i]<-0.5 || rx[i] > 0.5) printf("\nrx[%d] = %f",i,rx[i]);
      if (ry[i]<-0.5 || ry[i] > 0.5) printf("\nry[%d] = %f",i,ry[i]);
      if (rz[i]<-0.5 || rz[i] > 0.5) printf("\nrz[%d] = %f",i,rz[i]);
      xi = (int)((rx[i]+0.5)/sfx) + 1;
      yi = (int)((ry[i]+0.5)/sfy) + 1;
      zi = (int)((rz[i]+0.5)/sfz) + 1;
      if(xi > mx) xi = mx;
      if(yi > my) yi = my;
      if(zi > mz) zi = mz;
      icell = xi + (mx+2)*(yi + zi*(my+2));
   //   printf("\nrx = %f, ry = %f, rz = %f",rx[i],ry[i],rz[i]);
  //    if(icell>=(mx+2)*(my+2)*(mz+2))printf("\nxi = %d, yi = %d, zi = %d, icell = %d",xi,yi,zi,icell);
//      fflush(stdout);
      list[i]      = head[icell];
      head [icell] = i;
   }
   //printf("\nFinished main for loop in movout\n");

   pptr = natoms;
   for(j=1;j<my+1;j++)
       for(i=1;i<mx+1;i++){
           src = mz;
	   dst = 0;
	   scell = i + (mx+2)*(j + (my+2)*src);
	   dcell = i + (mx+2)*(j + (my+2)*dst);
	   p = head[scell];
	   while(p>=0){
	       rx[pptr] = rx[p]; ry[pptr] = ry[p]; rz[pptr] = rz[p] - 1.0;
	       list[pptr] = head[dcell];
	       head[dcell] = pptr;
	       pptr++;
	       p = list[p];
	   }
           src = 1;
	   dst = mz+1;
	   scell = i + (mx+2)*(j + (my+2)*src);
	   dcell = i + (mx+2)*(j + (my+2)*dst);
	   p = head[scell];
	   while(p>=0){
	       rx[pptr] = rx[p]; ry[pptr] = ry[p]; rz[pptr] = rz[p] + 1.0;
	       list[pptr] = head[dcell];
	       head[dcell] = pptr;
	       pptr++;
	       p = list[p];
	   }
       }
   for(k=0;k<mz+2;k++)
       for(j=1;j<my+1;j++){
           src = mx;
	   dst = 0;
	   scell = src + (mx+2)*(j + (my+2)*k);
	   dcell = dst + (mx+2)*(j + (my+2)*k);
	   p = head[scell];
	   while(p>=0){
	       rx[pptr] = rx[p] - 1.0; ry[pptr] = ry[p]; rz[pptr] = rz[p];
	       list[pptr] = head[dcell];
	       head[dcell] = pptr;
	       pptr++;
	       p = list[p];
	   }
           src = 1;
	   dst = mx+1;
	   scell = src + (mx+2)*(j + (my+2)*k);
	   dcell = dst + (mx+2)*(j + (my+2)*k);
	   p = head[scell];
	   while(p>=0){
	       rx[pptr] = rx[p] + 1.0; ry[pptr] = ry[p]; rz[pptr] = rz[p];
	       list[pptr] = head[dcell];
	       head[dcell] = pptr;
	       pptr++;
	       p = list[p];
	   }
       }
   for(k=0;k<mz+2;k++)
       for(i=0;i<mx+2;i++){
           src = my;
	   dst = 0;
	   scell = i + (mx+2)*(src + (my+2)*k);
	   dcell = i + (mx+2)*(dst + (my+2)*k);
	   p = head[scell];
	   while(p>=0){
	       rx[pptr] = rx[p]; ry[pptr] = ry[p] - 1.0; rz[pptr] = rz[p];
	       list[pptr] = head[dcell];
	       head[dcell] = pptr;
	       pptr++;
	       p = list[p];
	   }
           src = 1;
	   dst = my+1;
	   scell = i + (mx+2)*(src + (my+2)*k);
	   dcell = i + (mx+2)*(dst + (my+2)*k);
	   p = head[scell];
	   while(p>=0){
	       rx[pptr] = rx[p]; ry[pptr] = ry[p] + 1.0; rz[pptr] = rz[p];
	       list[pptr] = head[dcell];
	       head[dcell] = pptr;
	       pptr++;
	       p = list[p];
	   }
       }
   *ntot = pptr;
 //  printf("\n***movout ntot = %d\n",*ntot);
//   printf("\nLeaving movout: mx = %d, my = %d, mz = %d, natoms = %d, sfx = %f, sfy = %f, sfz = %f\n",mx, my, mz, natoms, sfx, sfy, sfz);
}









