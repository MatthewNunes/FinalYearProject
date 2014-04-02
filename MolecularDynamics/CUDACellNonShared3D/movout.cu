#include <stdio.h>
#include "moldyn.h"

void movout (float *rx, float *ry, float *rz, float *vx, float *vy, float *vz, float sfx, float sfy, float sfz, int *head, int *list, int mx, int my, int mz, int natoms, int gtx, int gty, int gtz)
{
   int    i, j, k, icell, xi, yi, zi;
   int src, dst, scell, dcell, p, pptr;

   int ncells = gtx*gty*gtz;
//   printf("\nStarting movout: mx = %d, my = %d, mz = %d, natoms = %d, sfx = %f, sfy = %f, sfz = %f\n",mx, my, mz, natoms, sfx, sfy, sfz);
   for (icell=0;icell<ncells;++icell) head[icell] = -1;

 //  printf("\nStarting for loop in movout: mx = %d, my = %d, mz = %d, natoms = %d, sfx = %f, sfy = %f, sfz = %f\n",mx, my, mz, natoms, sfx, sfy, sfz);
   //printf("\ngtx = %d, gty = %d, gtz = %d",gtx,gty,gtz);
  // fflush(stdout);
   for(i=0;i<natoms;i++){
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
      if(xi < 1) xi = 1;
      if(yi < 1) yi = 1;
      if(zi < 1) zi = 1;
      icell = xi + gtx*(yi + zi*gty);
   //   printf("\nrx = %f, ry = %f, rz = %f",rx[i],ry[i],rz[i]);
      //printf("\nxi = %d, yi = %d, zi = %d, icell = %d",xi,yi,zi,icell);
//      fflush(stdout);
      list[i]      = head[icell];
      head [icell] = i;
   }
//   printf("\nFinished for loop in movout\n");

 //  check_cells(rx, ry, rz, head, list, mx, my, mz, natoms, 0, 0, gtx, gty, gtz);

  // printf("Copying slab 1\n");
   //fflush(stdout);
   pptr = natoms;
   for(j=1;j<my+1;j++)
       for(i=1;i<mx+1;i++){
           src = mz;
	   dst = 0;
	   scell = i + gtx*(j + gty*src);
	   dcell = i + gtx*(j + gty*dst);
	   p = head[scell];
	   while(p>=0){
//               if(pptr>=2*natoms){
//                   printf("1: p = %d, pptr = %d\n",p,pptr);
//                   fflush(stdout);
//               }
	       rx[pptr] = rx[p]; ry[pptr] = ry[p]; rz[pptr] = rz[p] - 1.0;
	       list[pptr] = head[dcell];
	       head[dcell] = pptr;
	       pptr++;
	       p = list[p];
	   }
           src = 1;
	   dst = mz+1;
	   scell = i + gtx*(j + gty*src);
	   dcell = i + gtx*(j + gty*dst);
	   p = head[scell];
	   while(p>=0){
//               if(pptr>=2*natoms){
//                   printf("2: p = %d, pptr = %d\n",p,pptr);
//                   fflush(stdout);
//               }
	       rx[pptr] = rx[p]; ry[pptr] = ry[p]; rz[pptr] = rz[p] + 1.0;
	       list[pptr] = head[dcell];
	       head[dcell] = pptr;
	       pptr++;
	       p = list[p];
	   }
       }
//    printf("Number of particles in z-slabs = %d\n",pptr-natoms);
//   printf("Copying slab 2\n");
//   fflush(stdout);
   for(k=0;k<mz+2;k++)
       for(j=1;j<my+1;j++){
           src = mx;
	   dst = 0;
	   scell = src + gtx*(j + gty*k);
	   dcell = dst + gtx*(j + gty*k);
	   p = head[scell];
	   while(p>=0){
//               if(pptr>=2*natoms){
//                   printf("1: p = %d, pptr = %d\n",p,pptr);
//                   fflush(stdout);
//               }
	       rx[pptr] = rx[p] - 1.0; ry[pptr] = ry[p]; rz[pptr] = rz[p];
	       list[pptr] = head[dcell];
	       head[dcell] = pptr;
	       pptr++;
	       p = list[p];
	   }
           src = 1;
	   dst = mx+1;
	   scell = src + gtx*(j + gty*k);
	   dcell = dst + gtx*(j + gty*k);
	   p = head[scell];
	   while(p>=0){
//               if(pptr>=2*natoms){
//                   printf("2: p = %d, pptr = %d\n",p,pptr);
//                   fflush(stdout);
//               }
	       rx[pptr] = rx[p] + 1.0; ry[pptr] = ry[p]; rz[pptr] = rz[p];
	       list[pptr] = head[dcell];
	       head[dcell] = pptr;
	       pptr++;
	       p = list[p];
	   }
       }
//    printf("Number of particles in x-slabs = %d\n",pptr-natoms);
//   printf("Copying slab 3\n");
//   fflush(stdout);
   for(k=0;k<mz+2;k++)
       for(i=0;i<mx+2;i++){
           src = my;
	   dst = 0;
	   scell = i + gtx*(src + gty*k);
	   dcell = i + gtx*(dst + gty*k);
	   p = head[scell];
	   while(p>=0){
//               if(pptr>=2*natoms){
//                   printf("1: p = %d, pptr = %d\n",p,pptr);
//                   fflush(stdout);
//               }
	       rx[pptr] = rx[p]; ry[pptr] = ry[p] - 1.0; rz[pptr] = rz[p];
	       list[pptr] = head[dcell];
	       head[dcell] = pptr;
	       pptr++;
	       p = list[p];
	   }
           src = 1;
	   dst = my+1;
	   scell = i + gtx*(src + gty*k);
	   dcell = i + gtx*(dst + gty*k);
	   p = head[scell];
	   while(p>=0){
//               if(pptr>=2*natoms){
//                   printf("2: p = %d, pptr = %d\n",p,pptr);
//                   fflush(stdout);
//               }
	       rx[pptr] = rx[p]; ry[pptr] = ry[p] + 1.0; rz[pptr] = rz[p];
	       list[pptr] = head[dcell];
	       head[dcell] = pptr;
	       pptr++;
	       p = list[p];
	   }
       }
//    printf("Number of particles in y-slabs = %d\n",pptr-natoms);
//   printf("\nLeaving movout: mx = %d, my = %d, mz = %d, natoms = %d, sfx = %f, sfy = %f, sfz = %f\n",mx, my, mz, natoms, sfx, sfy, sfz);
//                   fflush(stdout);
}









