#define BLOCK_SIZE 512
__global__ 
void force (float *virialArray, float *potentialArray, float *pval, float *vval, float *rx, float *ry, float *rz, float *fx, float *fy, float *fz, float sigma, float rcut, float vrcut, float dvrc12, float dvrcut, int *head, int *list, int mx, int my, int mz, int natoms, int step, float sfx, float sfy, float sfz)
{
   float sigsq, rcutsq;
   float rxi, ryi, rzi, fxi, fyi, fzi;
   float rxij, ryij, rzij, rijsq;
   float rij, sr2, sr6, vij, wij, fij, fxij, fyij, fzij;
   float potential, virial;
   int i, icell, j, jcell, nabor;
   int xi, yi, zi, ix, jx, kx, xcell, ycell, zcell;
   extern __shared__ float vpArray[];
   int p_start = BLOCK_SIZE;
   sigsq  = sigma*sigma;
   rcutsq = rcut*rcut;

   potential = 0.0;
   virial    = 0.0;
   
   int element = blockIdx.x * blockDim.x + threadIdx.x;
   if (element < natoms)
   {
  	 rxi = rx[element];
  	 ryi = ry[element];
  	 rzi = rz[element];
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
                           if (j!=element) {
                               rxij = rxi - rx[j];
                               ryij = ryi - ry[j];
                               rzij = rzi - rz[j];
                               rijsq = rxij*rxij + ryij*ryij + rzij*rzij;
                               if (rijsq < rcutsq) {
  			                           //START FORCE_IJ
                                    
                                    rij = (float) sqrt ((double)rijsq);
                                    sr2 = sigsq/rijsq;
                                    sr6 = sr2*sr2*sr2;
                                    vij = sr6*(sr6-1.0) - vrcut - dvrc12*(rij-rcut);
                                    wij = sr6*(sr6-0.5) + dvrcut*rij;
                                    fij = wij/rijsq;
                                    fxij = fij*rxij;
                                    fyij = fij*ryij;
                                    fzij = fij*rzij;
                                   //END FORCE_IJ
                                   wij *= 0.5;
                                   vij *= 0.5;
                                   potential += vij;
                                   virial    += wij;
                                   fxi       += fxij;
                                   fyi       += fyij;
                                   fzi       += fzij;
                               }
  			 }
                           j = list[j];
                        }
  	         }
           *(fx+element) = 48.0*fxi;
           *(fy+element) = 48.0*fyi;
           *(fz+element) = 48.0*fzi;

            vpArray[threadIdx.x] = virial;
            //pArray[threadIdx.x] = potential;
            vpArray[threadIdx.x + p_start] = potential;
            unsigned int stride;
            unsigned int t = threadIdx.x;
            for(stride = blockDim.x / 2; stride > 0; stride >>= 1)
            {
               __syncthreads();
               if (t<stride)
               {
                  vpArray[t]+= vpArray[t+stride];
                  vpArray[t+p_start]+= vpArray[t+p_start+stride];
                  //vArray[t]+= vArray[t+stride];
               }
            }
            __syncthreads();
            if (threadIdx.x == 0)
            {
               potentialArray[blockIdx.x] = vpArray[p_start];
               virialArray[blockIdx.x] = vpArray[0];
            }
     }

}

__global__
void finalResult(float *potentialArray, float *virialArray, float *potentialValue, float *virialValue, int n)
{

   unsigned int stride;
   unsigned int t = threadIdx.x;
   int p_start = n;
   extern __shared__ float vpArray[];
 //  extern __shared__ float vArray[];
   float potential;
   float virial;
   if (threadIdx.x < n)
   {
      vpArray[threadIdx.x] = virialArray[threadIdx.x];
      vpArray[threadIdx.x+p_start] = potentialArray[threadIdx.x];
      //vArray[threadIdx.x] = virialArray[threadIdx.x];
      for(stride = blockDim.x / 2; stride > 0; stride >>= 1)
      {
         __syncthreads();
         if (t<stride)
         {
            vpArray[threadIdx.x] += vpArray[threadIdx.x + stride];
            vpArray[t+p_start]+= vpArray[t+p_start+stride];
            //vArray[t]+= vArray[t+stride];
         }
      }

   }
    __syncthreads();
   if (threadIdx.x == 0)
   {
      potential = vpArray[p_start];
      virial = vpArray[0];
      potential *= 4.0;
      virial    *= 48.0/3.0;
      *potentialValue = potential;
      *virialValue = virial;
   }

   //*pval = potential;
   //*vval = virial;
}