#include <cuda.h>
#include <math.h>
#include <stdio.h>

#include "moldyn.h"

__global__ void force (int maxP, float *potentialArray, float *virialArray, float *rx, float *ry, float *rz, float *fx, float *fy, float *fz, float sigma, float rcut, float vrcut, float dvrc12, float dvrcut, int *head, int *list, int mx, int my, int mz, int gtx, int gty, int gtz)
{
   float sigsq, rcutsq;
   float rxi, ryi, rzi, fxi, fyi, fzi;
   float rxij, ryij, rzij, rijsq;
   float rij, sr2, sr6, vij, wij, fij, fxij, fyij, fzij;
   float potential, virial;
   int i, j, jcell;
   int xi, yi, zi, ix, jx, kx, xcell, ycell, zcell;

   sigsq  = sigma*sigma;
   rcutsq = rcut*rcut;
   extern __shared__ float sharedArr[];

   potential = 0.0;
   virial    = 0.0;
   int iSh;
   int jTemp;
   int jSh;
   
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

   if((xi < (mx+2))&&(yi < (my+2))&&(zi < (mz+2)))
   {
        i = head[element];
        
        iSh = 0;
      
        while (i >= 0)
        {
          sharedArr[3*maxP*et + 3*iSh] = rx[i];
          sharedArr[3*maxP*et + 3*iSh+1] = ry[i];
          sharedArr[3*maxP*et + 3*iSh+2] = rz[i];
          i = list[i];
          iSh+=1;
          
        }
    }
    __syncthreads();

   if(((xi>0) && (xi <(mx+1)))&&((yi>0) && (yi<(my+1)))&&((zi>0) && (zi<(mz+1))))
      {   
        i = head[element];
        iSh = 0;

        while (i>=0) 
        {
          rxi = sharedArr[3*maxP*et + 3*iSh];
          ryi = sharedArr[3*maxP*et + 3*iSh+1];
          rzi = sharedArr[3*maxP*et + 3*iSh+2];
          fxi = fyi = fzi = 0.0;

          j = head[element];
          jTemp = 0;
          while (j>=0) 
          {
            rxij = rxi - sharedArr[3*maxP*et + 3*jTemp];
            ryij = ryi - sharedArr[3*maxP*et + 3*jTemp+1];
            rzij = rzi - sharedArr[3*maxP*et + 3*jTemp+2];
//            rxij = rxi - rx[j];
//            ryij = ryi - ry[j];
//            rzij = rzi - rz[j];
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
            jTemp+=1;
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
//                  if ( (jcell < ((blockIdx.x+1) * blockDim.x)) && (jcell >= ((blockIdx.x) * blockDim.x)))
                if((xcell/blockDim.x==blockIdx.x)&&(ycell/blockDim.y==blockIdx.y)&&(zcell/blockDim.z==blockIdx.z))
                  {
                    j = head[jcell];
                    jSh = 0;
                    jcell = (xcell%blockDim.x) + blockDim.x*((ycell%blockDim.y)+blockDim.y*(zcell%blockDim.z));
                    while (j>=0) 
                    {
                      rxij = rxi - sharedArr[3*maxP*jcell + 3*jSh];
                      ryij = ryi - sharedArr[3*maxP*jcell + 3*jSh+1];
                      rzij = rzi - sharedArr[3*maxP*jcell + 3*jSh+2];
//                      rxij = rxi - rx[j];
//                      ryij = ryi - ry[j];
//                      rzij = rzi - rz[j];
                      rijsq = rxij*rxij + ryij*ryij + rzij*rzij;
                      if (rijsq < rcutsq) 
                      {
                        //START FORCE IJ
                        rij = (float) sqrt ((double)rijsq);
                        sr2 = sigsq/rijsq;
                        sr6 = sr2*sr2*sr2;
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
                      jSh+=1;
                    }

                  }
                  else
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
              }  
          *(fx+i) = 48.0*fxi;
          *(fy+i) = 48.0*fyi;
          *(fz+i) = 48.0*fzi;
          i = list[i];  
          iSh+=1;
	      }//While loop (current cell under consideration)
      }//if statement checking that cell's coordinates are within range
//    potentialArray[element] = potential;
 //   virialArray[element] = virial;
    int tpb = blockDim.x*blockDim.y*blockDim.z;
    int offset = 3*maxP*tpb;
    sharedArr[offset+et] = virial;
    sharedArr[offset+et + tpb] = potential;
            unsigned int stride;
            for(stride = tpb/2; stride > 0; stride >>= 1)
            {
               __syncthreads();
               if (et<stride)
               {
                  sharedArr[offset+et]+= sharedArr[offset+et+stride];
                  sharedArr[offset+et+tpb]+= sharedArr[offset+et+tpb+stride];
                  //vArray[t]+= vArray[t+stride];
               }
            }
            __syncthreads();
            if (et == 0)
            {
               potentialArray[eb] = sharedArr[offset+tpb];
               virialArray[eb] = sharedArr[offset];
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
