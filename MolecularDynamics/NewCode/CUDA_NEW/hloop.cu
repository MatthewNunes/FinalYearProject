#include "moldyn.h"
#include <stdio.h>
#include <cuda.h>
#include "matthew_CUDA.h"
void hloop (float kinetic, int step, float vg, float wg, float kg, float freex, float dens, float sigma, float eqtemp, float *tmpx, float *ace, float *acv, float *ack, float *acp, float *acesq, float *acvsq, float *acksq, float *acpsq, float *vx, float *vy, float *vz, int iscale, int iprint, int nequil, int natoms, int block_width, float *d_vx, float *d_vy, float *d_vz)
{
   float e, en, vn, kn, pres;
  // float *d_tmpx;
   e = kg + vg;
   en = e/(float)natoms;
   vn = vg/(float)natoms;
   kn = kg/(float)natoms;
   *tmpx = 2.0*kg/freex;
   pres = dens*(*tmpx) + wg;
   pres = pres*sigma*sigma*sigma;
   
   if (step>nequil) {
      *ace += en;
      *acv += vn;
      *ack += kn;
      *acp += pres;
      *acesq += en*en;
      *acvsq += vn*vn;
      *acksq += kn*kn;
      *acpsq += pres*pres;
   }

/* If still equilibrating call subroutine to scale velocities */
   //CUDA_CHECK_RETURN(cudaMalloc((void **) &d_tmpx, sizeof(float)));
   //CUDA_CHECK_RETURN(cudaMemcpy(d_tmpx, tmpx, sizeof(float), cudaMemcpyHostToDevice));
   int numBlocks = ceil(natoms/ (float) block_width);
   if (nequil > step) 
   {
      scalet<<<numBlocks, block_width>>> (d_vx, d_vy, d_vz, kinetic, eqtemp, *tmpx, iscale, natoms, step);
      CUDA_CHECK_RETURN(cudaPeekAtLastError());
      CUDA_CHECK_RETURN(cudaDeviceSynchronize());
      //CUDA_CHECK_RETURN(cudaMemcpy(tmpx, d_tmpx, sizeof(float), cudaMemcpyDeviceToHost));
   }
/* Optionally print information */
      if (step%iprint == 0)
         printf("\n%8d%12.6f%12.6f%12.6f%12.6f%12.6f",step, en, kn, vn,
                 pres, *tmpx);
}