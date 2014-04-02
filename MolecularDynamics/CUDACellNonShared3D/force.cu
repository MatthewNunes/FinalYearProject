#include <cuda.h>
#include <math.h>
#include <stdio.h>

#include "moldyn.h"

__global__ void force (float *potentialArray, float *virialArray, float *rx, float *ry, float *rz, float *fx, float *fy, float *fz, float sigma, float rcut, float vrcut, float dvrc12, float dvrcut, int *head, int *list, int mx, int my, int mz, int gtx, int gty, int gtz)
{
   float sigsq, rcutsq;
   float rxi, ryi, rzi, fxi, fyi, fzi;
   float rxij, ryij, rzij, rijsq;
   float rij, sr2, sr6, vij, wij, fij, fxij, fyij, fzij;
   float potential, virial;
   int i, j, jcell;
   int xi, yi, zi, ix, jx, kx, xcell, ycell, zcell;
   extern __shared__ float sharedArr[];

   sigsq  = sigma*sigma;
   rcutsq = rcut*rcut;

   potential = 0.0;
   virial    = 0.0;
   
// et is the thread index within the block
// eb is the global block index

   int et = blockDim.x*(blockDim.y*threadIdx.z+threadIdx.y)+threadIdx.x;
   int eb = gridDim.x*(gridDim.y*blockIdx.z+blockIdx.y)+blockIdx.x;

// xi is the global thread index in the x direction
// yi is the global thread index in the y direction
// zi is the global thread index in the z direction
// element is the global thread index

   xi = blockDim.x*blockIdx.x+threadIdx.x;
   yi = blockDim.y*blockIdx.y+threadIdx.y;
   zi = blockDim.z*blockIdx.z+threadIdx.z;
   int element = xi + gtx*(yi+gty*zi);

   if(((xi>0) && (xi <(mx+1)))&&((yi>0) && (yi<(my+1)))&&((zi>0) && (zi<(mz+1))))
      {   
        i = head[element];

        while (i>=0) 
        {
          rxi = rx[i];
          ryi = ry[i];
          rzi = rz[i];
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
//              vij = sr6*(sr6-1.0) - vrcut - dvrc12*(rij-rcut);
 //             wij = sr6*(sr6-0.5) + dvrcut*rij;
              vij = __fadd_rn(__fadd_rn(__fmul_rn(sr6, __fadd_rn(sr6,-1.0)), -vrcut), __fmul_rn(-dvrc12, __fadd_rn(rij, -rcut)));
              wij = __fadd_rn(__fmul_rn(sr6, __fadd_rn(sr6, -0.5)), __fmul_rn(dvrcut, rij));
              fij = wij/rijsq;
              fxij = fij*rxij;
              fyij = fij*ryij;
              fzij = fij*rzij;
              //END FORCE IJ
              vij *= 0.5;
              wij *= 0.5;
              potential += vij;
              virial    += wij;
              fxi+= fxij;
              fyi+= fyij;
              fzi+= fzij;
            }           
            j = list[j];
          }
          
          //	      printf("\nCell %d at (%d,%d,%d) interacts with cells: ",icell,xi,yi,zi);
          for (ix=-1;ix<=1;ix++)
            for (jx=-1;jx<=1;jx++)
              for (kx=-1;kx<=1;kx++)
              {
                xcell = ix+xi;
                ycell = jx+yi;
                zcell = kx+zi;
                jcell = xcell + gtx*(ycell+gty*zcell);
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
//                        vij = sr6*(sr6-1.0) - vrcut - dvrc12*(rij-rcut);
 //                       wij = sr6*(sr6-0.5) + dvrcut*rij;
                        vij = __fadd_rn(__fadd_rn(__fmul_rn(sr6, __fadd_rn(sr6,-1.0)), -vrcut), __fmul_rn(-dvrc12, __fadd_rn(rij, -rcut)));
                        wij = __fadd_rn(__fmul_rn(sr6, __fadd_rn(sr6, -0.5)), __fmul_rn(dvrcut, rij));
                        fij = wij/rijsq;
                        fxij = fij*rxij;
                        fyij = fij*ryij;  
                        fzij = fij*rzij;
                        //END FORCE IJ
                        wij *= 0.5;
                        vij *= 0.5;
	                potential += vij;
     	                virial += wij;
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
	}//While loop (current cell under consideration)
      }//if statement checking that cell's coordinates are within range
//    potentialArray[element] = potential;
 //   virialArray[element] = virial;
    int tpb = blockDim.x*blockDim.y*blockDim.z;
    sharedArr[et] = virial;
    sharedArr[et+tpb] = potential;
            unsigned int stride;
            for(stride = tpb/2; stride > 0; stride >>= 1)
            {
               __syncthreads();
               if (et<stride)
               {
                  sharedArr[et]+= sharedArr[et+stride];
                  sharedArr[et+tpb]+= sharedArr[et+tpb+stride];
                  //vArray[t]+= vArray[t+stride];
               }
            }
            __syncthreads();
            if (et == 0)
            {
               potentialArray[eb] = sharedArr[tpb];
               virialArray[eb] = sharedArr[0];
            }

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
