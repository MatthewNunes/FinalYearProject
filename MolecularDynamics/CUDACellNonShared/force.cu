
#include <math.h>
#include <stdio.h>

#include "moldyn.h"

#define BLOCK_SIZE 512

__global__ void force (float *potentialArray, float *virialArray, float *pval, float *vval, float *rx, float *ry, float *rz, float *fx, float *fy, float *fz, float sigma, float rcut, float vrcut, float dvrc12, float dvrcut, int *head, int *list, int *map, int mx, int my, int mz, int natoms, int step)
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
   int element = blockDim.x * blockIdx.x + threadIdx.x;

   potentialArray[element] = 0.0;
   virialArray[element] = 0.0;
   if(element < ((mx+2) * (my + 2) * (mz + 2)))
   {

      xi = element%(mx+2);
      yi = (element/(mx+2))%(my+2);
      zi = element/((mx+2)*(my+2));
      if(((xi>0) && (xi <(mx+1)))&&((yi>0) && (yi<(my+1)))&&((zi>0) && (zi<(mz+1))))
      {
        i = head[element];
        while (i>=0) 
        {
          rxi = rx[i];
          ryi = ry[i];
          rzi = rz[i];
//	 printf("Particle %5d, (xi,yi,zi) = %d,%d,%d, icel = %5d\n",i,xi,yi,zi,icell);
//               fxi = fx[i];
 //              fyi = fy[i];
  //             fzi = fz[i];
          fxi = fyi = fzi = 0.0;

          j = head[element];
          while (j>=0) 
          {
            rxij = rxi - rx[j];
            ryij = ryi - ry[j];
            rzij = rzi - rz[j];
            rijsq = rxij*rxij + ryij*ryij + rzij*rzij;
            if ((rijsq < rcutsq) && (j!=i)) 
            {
                    //START FORCE IJ
	               //force_ij(rijsq, rxij, ryij, rzij, sigsq, vrcut, dvrc12, rcut, dvrcut, &vij, &wij, &fxij, &fyij, &fzij);
                
              rij = (float) sqrt ((double)rijsq);
              sr2 = sigsq/rijsq;
              sr6 = sr2*sr2*sr2;
              vij = sr6*(sr6-1.0) - vrcut - dvrc12*(rij-rcut);
              wij = sr6*(sr6-0.5) + dvrcut*rij;
              fij = wij/rijsq;
              fxij = fij*rxij;
              fyij = fij*ryij;
              fzij = fij*rzij;
		          //END FORCE IJ
              vij *= 0.5;
		          wij *= 0.5;
		          valp += vij;
		          valv += wij;
//                       potential += 0.5*vij;
//                       virial    += 0.5*wij;
              fxi+= fxij;
              fyi+= fyij;
              fzi+= fzij;
            }           
            j = list[j];
          }
	        icount++;
//	      printf("\nCell %d at (%d,%d,%d) interacts with cells: ",icell,xi,yi,zi);
          for (ix=-1;ix<=1;ix++)
            for (jx=-1;jx<=1;jx++)
              for (kx=-1;kx<=1;kx++)
              {
                xcell = ix+xi;
                ycell = jx+yi;
                zcell = kx+zi;
                jcell = xcell + (mx+2)*(ycell+(my+2)*zcell);
			//       printf("%d (%d,%d,%d); ",jcell,xcell,ycell,zcell);
		            if(element!=jcell) 
                {
                  j = head[jcell];
                  while (j>=0) 
                  {
                    rxij = rxi - rx[j];
                    ryij = ryi - ry[j];
                    rzij = rzi - rz[j];
                    rijsq = rxij*rxij + ryij*ryij + rzij*rzij;
                    if (rijsq < rcutsq) 
                    {
                      //START FORCE IJ
                      rij = (float) sqrt ((double)rijsq);
                      sr2 = sigsq/rijsq;
                      sr6 = sr2*sr2*sr2;
                      vij = sr6*(sr6-1.0) - vrcut - dvrc12*(rij-rcut);
                      wij = sr6*(sr6-0.5) + dvrcut*rij;
                      fij = wij/rijsq;
                      fxij = fij*rxij;
                      fyij = fij*ryij;
                      fzij = fij*rzij;
                      //END FORCE IJ
                      wij *= 0.5;
                      vij *= 0.5;
				              valp += vij;
				              valv += wij;
                      fxi += fxij;
                      fyi += fyij;
                      fzi += fzij;
                    }
                    j = list[j];
			            }
                }		          
              }  
          *(fx+i) = 48.0*fxi;
          *(fy+i) = 48.0*fyi;
          *(fz+i) = 48.0*fzi;
          i = list[i];  
          //printf("valp: %f from element: %d\n", valp, element);
	        potential += valp;
	        virial += valv;
	        valp = valv = 0.0;           
	      }//While loop (current cell under consideration)
      }//if statement checking that cell's coordinates are within range
    }
    potentialArray[element] = potential;
    virialArray[element] = virial;
    //if statement over all cells
  // if(icount!=natoms) printf("\nProcessed %d particles in force routine instead of %d",icount,natoms);
  // potential *= 4.0;
  // virial    *= 48.0/3.0;
  // *pval = potential;
  // *vval = virial;

//   for (i=0;i<natoms;++i) {
//      *(fx+i) *= 48.0;
//      *(fy+i) *= 48.0;
//      *(fz+i) *= 48.0;
//   }
}
