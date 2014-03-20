#include <stdio.h>
#include "moldyn.h"

void movout (float *rx, float *ry, float *rz, float *vx, float *vy, float *vz, float sfx, float sfy, float sfz, int *head, int *list, int mx, int my, int mz, int natoms)
{
   int    i, j, k, icell, xi, yi, zi;
   int src, dst, scell, dcell, p, pptr;

   int ncells = (mx+2)*(my+2)*(mz+2);
//   printf("\nStarting movout: mx = %d, my = %d, mz = %d, natoms = %d, sfx = %f, sfy = %f, sfz = %f\n",mx, my, mz, natoms, sfx, sfy, sfz);
   for (icell=0;icell<ncells;++icell) head[icell] = -1;

   //printf("\nStarting for loop in movout: mx = %d, my = %d, mz = %d, natoms = %d, sfx = %f, sfy = %f, sfz = %f\n",mx, my, mz, natoms, sfx, sfy, sfz);
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
      if(xi < 1) xi = 1;
      if(yi < 1) yi = 1;
      if(zi < 1) zi = 1;
      icell = xi + (mx+2)*(yi + zi*(my+2));
   //   printf("\nrx = %f, ry = %f, rz = %f",rx[i],ry[i],rz[i]);
      //printf("\nxi = %d, yi = %d, zi = %d, icell = %d",xi,yi,zi,icell);
      fflush(stdout);
      list[i]      = head[icell];
      head [icell] = i;
   }
 //  printf("4\n");

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
	  // printf("5.2\n");
	   scell = src + (mx+2)*(j + (my+2)*k);
	  // printf("5.3\n");
	   dcell = dst + (mx+2)*(j + (my+2)*k);
	 //  printf("5.4\n");
	   p = head[scell];
	 //  printf("5.5\n");
	   while(p>=0){
	   //    printf("5.6\n");
	       rx[pptr] = rx[p] - 1.0; ry[pptr] = ry[p]; rz[pptr] = rz[p];
	    //   printf("5.7\n");
	       list[pptr] = head[dcell];
	      // printf("5.8\n");
	       head[dcell] = pptr;
	     //  printf("5.9\n");
	       pptr++;
	     //  printf("5.10\n");
	       p = list[p];
	   //    printf("5.11\n");
	   }
           src = 1;
         //  printf("5.12\n");
	   dst = mx+1;
	  // printf("5.13\n");
	   scell = src + (mx+2)*(j + (my+2)*k);
	  // printf("5.14\n");
	   dcell = dst + (mx+2)*(j + (my+2)*k);
	  // printf("5.15\n");
	   p = head[scell];
	  // printf("5.16\n");
	   while(p>=0){
	    //   printf("5.165\n");
	      // printf("pptr: %d\n",pptr );
	       rx[pptr] = rx[p] + 1.0; ry[pptr] = ry[p]; rz[pptr] = rz[p];
	       //printf("5.17\n");
	       list[pptr] = head[dcell];
	      // printf("5.18\n");
	       head[dcell] = pptr;
	       //printf("5.19\n");
	       pptr++;
	      // printf("5.20\n");
	       p = list[p];
	//       printf("5.21\n");
	   }
       }
      // printf("6\n");
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
       //printf("7\n");
//   printf("\nLeaving movout: mx = %d, my = %d, mz = %d, natoms = %d, sfx = %f, sfy = %f, sfz = %f\n",mx, my, mz, natoms, sfx, sfy, sfz);
}









