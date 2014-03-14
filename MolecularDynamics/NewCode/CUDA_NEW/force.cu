#define BLOCK_SIZE 512
#include <stdio.h>
#include <cuda.h>
#include <math.h>

__constant__ float sigma;
__constant__ float rcut;
__constant__ float vrcut;
__constant__ float dvrc12;
__constant__ float dvrcut;
__constant__ int mx;
__constant__ int my;
__constant__ int mz;
__constant__ int natoms;
__constant__ int step;
__constant__ float sfx;
__constant__ float sfy;
__constant__ float sfz;

void copySigma(float *sig)
{
  cudaMemcpyToSymbol("sigma", sig, sizeof(float));
}

void copyRcut(float *rcu)
{
  cudaMemcpyToSymbol("rcut", rcu, sizeof(float));
}

void copyVrcut(float *vrc)
{
  cudaMemcpyToSymbol("vrcut", vrc, sizeof(float));
}

void copyDvrc12(float *dvr)
{
  cudaMemcpyToSymbol("dvrc12", dvr, sizeof(float));
}

void copyDvrcut(float *dvrc)
{
  cudaMemcpyToSymbol("dvrcut", dvrc, sizeof(float));
}

void copyMx(int * m)
{
  cudaMemcpyToSymbol("mx", m, sizeof(int));
}

void copyMy(int *mm)
{
  cudaMemcpyToSymbol("my", mm, sizeof(int));
}

void copyMz(int *mmm)
{
  cudaMemcpyToSymbol("mz", mmm, sizeof(int));
}

void copyNatoms(int *nat)
{
  cudaMemcpyToSymbol("natoms", nat, sizeof(int));
}

void copyStep(int *ste)
{
  cudaMemcpyToSymbol("step", ste, sizeof(int));
}

void copySfx(float *sf)
{
  cudaMemcpyToSymbol("sfx", sf, sizeof(float));
}

void copySfy(float *sff)
{
  cudaMemcpyToSymbol("sfy", sff, sizeof(float));
}

void copySfz(float *sfff)
{
  cudaMemcpyToSymbol("sfz", sfff, sizeof(float));
}



__global__ 
void force (float *virialArray, float *potentialArray, float *pval, float *vval, float *rx, float *ry, float *rz, float *fx, float *fy, float *fz, float sigma, float rcut, float vrcut, float dvrc12, float dvrcut, int *head, int *list, int mx, int my, int mz, int natoms, float sfx, float sfy, float sfz)
{
   int element = blockIdx.x * blockDim.x + threadIdx.x;
   /**
   if (element == (natoms - 1))
   {
    printf("\n------------------------------------START------------------------------\n");
    printf("SIGMA: %f\n", sigma);
   }
   */
   float sigsq, rcutsq;
   float rxi, ryi, rzi, fxi, fyi, fzi;
   float rxij, ryij, rzij, rijsq;
   float rij, sr2, sr6, vij, wij, fij, fxij, fyij, fzij;
   int j, jcell;
   float potential, virial;
   int xi, yi, zi, ix, jx, kx, xcell, ycell, zcell;
   __shared__ float vArray[BLOCK_SIZE];
   __shared__ float pArray[BLOCK_SIZE];
   sigsq  = __fmul_rn(sigma, sigma);
   rcutsq = __fmul_rn(rcut,rcut);
   potential = (float)0.0;
   virial = (float)0.0;
   vArray[threadIdx.x] = (float)0.0;
   pArray[threadIdx.x] = (float)0.0;
   if (element < natoms)
   {
  	 rxi = rx[element];
  	 ryi = ry[element];
  	 rzi = rz[element];
     /**
     if (element == (natoms - 1))
     {
       printf("%d: rxi=%f\n",element, rxi);
       printf("%d: ryi=%f\n",element, ryi);
       printf("%d: rzi=%f\n",element, rzi);
     }
     */
     
  	 fxi = (float)0.0;
  	 fyi = (float)0.0;
  	 fzi = (float)0.0;
     //(int)((rxi+0.5)/sfx) + 1;
     //
     xi = (int)( (rxi + (float)0.5) / sfx);
     xi += 1;
  	 yi = (int)( (ryi + (float)0.5) / sfy);
     yi += 1;
  	 zi = (int)( (rzi + (float)0.5) / sfz);
     zi += 1;
     
     /**
     if (element == (natoms - 1))
     {
        printf("%d: xi=%d\n", element, xi);
        printf("%d: yi=%d\n", element, yi);
        printf("%d: zi=%d\n", element, zi);
     } 
     */
     
           if(xi > mx) 
           {
              xi = mx;  
           }
           if(yi > my)
           {
             yi = my;
           }
           if(zi > mz)
           {
             zi = mz;
           }
     
     /**
     if (element == (natoms - 1))
     {
        printf("%d: xi=%d\n", element, xi);
        printf("%d: yi=%d\n", element, yi);
        printf("%d: zi=%d\n", element, zi);
     } 
     */
    
     //CHANGED THIS
     //xi + (mx+2)*(yi+zi*(my+2))
     //icell = __float2int_rn(__fadd_rn(xi,  __fmul_rn(__fadd_rn(mx,2), __fadd_rn(yi, __fmul_rn(zi, __fadd_rn(my,2))))));
     //TO THIS
     //icell = xi;
     
     /**
     if (element == (natoms - 1))
     {
        printf("%d: icell=%d\n",element, icell);
     }
     */
      //icell += (mx+2)*(yi+zi*(my+2));
     
     /**
     if (element == (natoms - 1))
     {
        printf("%d: icell=%d\n",element, icell);
     }
     */

     
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
          /**   
          if (element == (natoms - 1))
          {
            printf("%d: xcell=%d\n",element, xcell);
            printf("%d: ycell=%d\n",element, ycell);
            printf("%d: zcell=%d\n",element, zcell);
                        
          }
          */

                       //CHANGED THIS
                      //jcell = xcell + (mx+2)*(ycell+(my+2)*zcell);
                      //jcell = __float2int_rn(__fadd_rn(xcell, __fmul_rn(__fadd_rn(mx,2),__fadd_rn(ycell,__fmul_rn(__fadd_rn(my,2),zcell)))));
                       jcell = xcell + (mx+2)*(ycell+(my+2)*zcell);
                       //TO THIS
                       //jcell = xcell;
                      /**
                      if (element == (natoms - 1))
                      {
                        printf("%d: jcell=%d\n",element, jcell);              
                      }
                            
                      // jcell += (mx+2)*(ycell+(my+2)*zcell);
                      
                      if (element == (natoms - 1))
                      {
                        printf("%d: jcell=%d\n",element, jcell);              
                      }
                      */
                      
  //	printf("%d (%d,%d,%d); ",jcell,xcell,ycell,zcell);
                       j = head[jcell];
                       /**
                       if (element == (natoms - 1))
                       {
                          printf("%d: j=%d\n",element, j);              
                       }
                       */
  //	 if(jcell<0||jcell>(mx+2)*(my+2)*(mz+2)-1) printf("\njcell = %d\n",jcell);
                       while (j>=0) 
                       {
  //			 if(j<0 || j>ntot-1) printf("\nj = %d\n",j);
                           if (j!=element) 
                           {
                               rxij = __fadd_rn(rxi, -rx[j]);
                               ryij = __fadd_rn(ryi, -ry[j]);
                               rzij = __fadd_rn(rzi, -rz[j]);
                               
                               /**     
                               if (element == (natoms - 1))
                               {
                                  printf("%d: rxij=%f\n",element,rxij);
                                  printf("%d: ryij=%f\n",element,ryij);
                                  printf("%d: rzij=%f\n",element,rzij);
                               }
                               */
                               
                               //rijsq = (rxij*rxij) + (ryij*ryij) + (rzij*rzij);
                               rijsq = __fadd_rn(__fadd_rn(__fmul_rn(rxij,rxij), __fmul_rn(ryij,ryij)), __fmul_rn(rzij,rzij));
                               //TO THIS
                               //rijsq = rxij*rxij;
                               //rijsq += ryij*ryij;
                               //rijsq += rzij*rzij;
                               
                               /**
                               if (element == (natoms - 1))
                               {
                                  printf("%d: rijsq=%f\n",element, rijsq);
                                  printf("%d: rijsq=%f\n",element, rijsq);
                                  printf("%d: rijsq=%f\n",element, rijsq);
                               }
                               */
                               
                               if (rijsq < rcutsq) 
                               {
  			                           //START FORCE_IJ
                                    
                                    rij = __fsqrt_rn(rijsq);
                                    sr2 = __fdiv_rn(sigsq,rijsq);
                                    sr6 = __fmul_rn(__fmul_rn(sr2,sr2),sr2);
                                    //CHANGED THIS
                                    //*vij = sr6*(sr6-1.0) - vrcut - dvrc12*(rij-rcut);
                                    vij = __fadd_rn(__fadd_rn(__fmul_rn(sr6,__fadd_rn(sr6,(float)-1.0)), -vrcut), __fmul_rn(-dvrc12, __fadd_rn(rij,-rcut)));
                                    //TO THIS
                                    //vij = sr6*(sr6-1.0);
                                    //vij -= vrcut;
                                    //vij -= dvrc12*(rij - rcut);
                                    //*wij = sr6*(sr6-0.5) + dvrcut*rij;
                                    wij = __fadd_rn(__fmul_rn(sr6,__fadd_rn(sr6,(float)-0.5)), __fmul_rn(dvrcut,rij));
                                    //TO THIS
                                    //wij = sr6*(sr6-0.5);
                                    //wij += dvrcut*rij;
                                    fij = __fdiv_rn(wij, rijsq);
                                    fxij = __fmul_rn(fij, rxij);
                                    fyij = __fmul_rn(fij, ryij);
                                    fzij = __fmul_rn(fij, rzij);
                                   //END FORCE_IJ
                                   wij = __fmul_rn(wij, (float)0.5);
                                   vij = __fmul_rn(vij, (float)0.5);
                                   potential = __fadd_rn(potential, vij);
                                   virial    = __fadd_rn(virial, wij);
                                   fxi       += fxij;
                                   fyi       += fyij;
                                   fzi       += fzij;
                                  
                                  /**
                                  if (element == (natoms - 1))
                                  {
                                    printf("%d: rij=%f\n",element, rij);
                                    printf("%d: sr2=%f\n",element, sr2);
                                    printf("%d: sr6=%f\n",element, sr6);
                                    printf("%d: vij=%f\n",element, vij);
                                    printf("%d: wij=%f\n",element, wij);
                                    printf("%d: fij=%f\n",element, fij);
                                    printf("%d: fxij=%f\n",element, fxij);
                                    printf("%d: fyij=%f\n",element, fyij);
                                    printf("%d: fzij=%f\n",element, fzij);
                                    printf("%d: potential=%f\n",element, potential);
                                    printf("%d: virial=%f\n",element, virial);
                                    printf("%d: fxi=%f\n",element, fxi);
                                    printf("%d: fyi=%f\n",element, fyi);
                                    printf("%d: fzi=%f\n",element, fzi);
                                  }
                                  */
                               }
  			                    }
                           j = list[j];
                           /**
                           if (element == (natoms - 1))
                           {
                              printf("%d j=%d\n",element, j);
                           }
                           */
                        }
  	         }
           *(fx+element) = __fmul_rn((float)48.0, fxi);
           *(fy+element) = __fmul_rn((float)48.0, fyi);
           *(fz+element) = __fmul_rn((float)48.0, fzi);
            
            /**
            if (element == (natoms - 1))
            {
              printf("%d: fx+element= %f\n",element, *(fx+element));
              printf("%d: fy+element= %f\n",element, *(fy+element));
              printf("%d: fz+element= %f\n",element, *(fz+element));
            }
            */

            vArray[threadIdx.x] = virial;
            //pArray[threadIdx.x] = potential;
            pArray[threadIdx.x] = potential;
            unsigned int t = threadIdx.x;
            unsigned int stride;
            for(stride = blockDim.x / 2; stride >0; stride >>= 1)
            {
               __syncthreads();
               if (t<stride)
               {
                  vArray[t]+= vArray[t+stride];
                  pArray[t]+= pArray[t+stride];
                  //vArray[t]+= vArray[t+stride];
               }
            }
            
            /**
            __syncthreads();
            if (t < 32)
            {
              vArray[t] += vArray[t + 32];
              vArray[t] += vArray[t + 16];
              vArray[t] += vArray[t + 8];
              vArray[t] += vArray[t + 4];
              vArray[t] += vArray[t + 2];
              vArray[t] += vArray[t + 1];
              pArray[t] += pArray[t + 32];
              pArray[t] += pArray[t + 16];
              pArray[t] += pArray[t + 8];
              pArray[t] += pArray[t + 4];
              pArray[t] += pArray[t + 2];
              pArray[t] += pArray[t + 1];
            }
            */
            __syncthreads();
            if (threadIdx.x == 0)
            {
               virialArray[blockIdx.x] = vArray[0];
               potentialArray[blockIdx.x] = pArray[0];
            }            
            /**
            if ((element == 0) || (element == 512))
            {
              printf("%d: vArray[0]= %f\n", element, vArray[0]);
              printf("%d: pArray[0]= %f\n", element, pArray[0]);
            }
            */
            
            /**if (element == (natoms - 1))
            {
              printf("------------------------------------END------------------------------\n");
            }
            */
     }
     //return;

}
