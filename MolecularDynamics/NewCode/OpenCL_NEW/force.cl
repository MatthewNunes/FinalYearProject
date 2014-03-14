#define BLOCK_WIDTH 512
__kernel void force (__local float *vpArray, __global float *virialArray, __global float *potentialArray, __global float *pval, __global float *vval, __global float *rx, __global float *ry, __global float *rz, __global float *fx, __global float *fy, __global float *fz, __private float sigma, __private float rcut, __private float vrcut, __private float dvrc12, __private float dvrcut, __global int *head, __global int *list, __private int mx, __private int my, __private int mz, __private int natoms, __private int step, __private float sfx, __private float sfy, __private float sfz)
{
   __private float sigsq, rcutsq;
   __private float rxi, ryi, rzi, fxi, fyi, fzi;
   __private float rxij, ryij, rzij, rijsq;
   __private float rij, sr2, sr6, vij, wij, fij, fxij, fyij, fzij;
   __private float potential, virial;
   __private int i, icell, j, jcell, nabor;
   __private int xi, yi, zi, ix, jx, kx, xcell, ycell, zcell;
   __private int p_start = BLOCK_WIDTH;

   sigsq  = sigma*sigma;
   rcutsq = rcut*rcut;
   *pval = 0.0;
   *vval = 0.0;
   potential = 0.0;
   virial    = 0.0;
   
   __private unsigned int t = get_local_id(0);
   __private unsigned int element = get_global_id(0);
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
  			                           //float rij, sr2, sr6, fij;

                                   rij = (float) sqrt ((float)rijsq);
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
            vpArray[t] = virial;
            //pArray[threadIdx.x] = potential;
            vpArray[t + p_start] = potential;
            __private unsigned int stride;
            //__private unsigned int t = threadIdx.x;
            __private int local_size = get_local_size(0);
            for(stride = local_size / 2; stride > 0; stride >>= 1)
            {
               barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
               if (t<stride)
               {
                  vpArray[t]+= vpArray[t+stride];
                  vpArray[t+p_start]+= vpArray[t+p_start+stride];
                  //vArray[t]+= vArray[t+stride];
               }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            if (t == 0)
            {
               potentialArray[get_group_id(0)] = vpArray[p_start];
               virialArray[get_group_id(0)] = vpArray[0];
            }
   }
}

__kernel void finalResult(__global float *potentialArray, __global float *virialArray, __global float *potentialValue, __global float *virialValue)
{

   __private unsigned int stride;
   __private unsigned int t = get_global_id(0);
   __private int p_start = 1;
   __private float potential;
   __private float virial;
   __private int local_size = get_global_size(0);
  // vpArray[t] = virialArray[t];
  // vpArray[t+p_start] = potentialArray[t];
      //vArray[threadIdx.x] = virialArray[threadIdx.x];
      for(stride = local_size / 2; stride > 0; stride >>= 1)
      {
         barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
         if (t<stride)
         {
            virialArray[t] += virialArray[t + stride];
            potentialArray[t]+= potentialArray[t+stride];
            //virialArray[get_group_id] = 0.0;
            //potentialArray[get_group_id] = 0.0;
            //vArray[t]+= vArray[t+stride];
         }
      }
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
   if (t == 0)
   {
      potential = potentialArray[0];
      virial = virialArray[0];
      potential *= 4.0;
      virial    *= 48.0/3.0;
      *potentialValue = potential;
      *virialValue = virial;
   }

   //*pval = potential;
   //*vval = virial;
}
