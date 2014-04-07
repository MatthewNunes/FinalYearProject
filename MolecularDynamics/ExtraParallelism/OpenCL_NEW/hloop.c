#include "moldyn.h"
#include <stdio.h>
#include <CL/cl.h>
#include <math.h>

void hloop (float kinetic, int step, float vg, float wg, float kg, float freex, float dens, float sigma, float eqtemp, float *tmpx, float *ace, float *acv, float *ack, float *acp, float *acesq, float *acvsq, float *acksq, float *acpsq, float *vx, float *vy, float *vz, int iscale, int iprint, int nequil, int natoms, cl_kernel *scaletKernel, cl_command_queue *myQueue, cl_mem *d_vx, cl_mem *d_vy, cl_mem *d_vz)
{
   float e, en, vn, kn, pres;
   cl_int err;
   e = kg + vg;
   en = e/(float)natoms;
   vn = vg/(float)natoms;
   kn = kg/(float)natoms;
   *tmpx = 2.0*kg/freex;
   pres = dens*(*tmpx) + wg;
   pres = pres*sigma*sigma*sigma;
   size_t global_size = BLOCK_WIDTH * ceil(natoms / (float) BLOCK_WIDTH);
   size_t local_size = BLOCK_WIDTH;   
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

   if (nequil > step)
   {
      err = clSetKernelArg(*scaletKernel, 0, sizeof(cl_mem), d_vx);
      err |= clSetKernelArg(*scaletKernel, 1, sizeof(cl_mem), d_vy);
      err |= clSetKernelArg(*scaletKernel, 2, sizeof(cl_mem), d_vz);
      err |= clSetKernelArg(*scaletKernel, 3, sizeof(kinetic), &kinetic);
      err |= clSetKernelArg(*scaletKernel, 4, sizeof(eqtemp), &eqtemp);
      err |= clSetKernelArg(*scaletKernel, 5, sizeof(float), tmpx);
      err |= clSetKernelArg(*scaletKernel, 6, sizeof(iscale), &iscale);
      err |= clSetKernelArg(*scaletKernel, 7, sizeof(natoms), &natoms);
      err |= clSetKernelArg(*scaletKernel, 8, sizeof(step), &step);
      if(err < 0) {
        printf("Couldn't set an argument for the scalet kernel");
        exit(1);   
      }

      err = clEnqueueNDRangeKernel(*myQueue, *scaletKernel, 1, NULL, &global_size, &local_size, 0, NULL, NULL);
      if(err < 0) {
        printf("Couldn't enqueue the scalet kernel");
        exit(1);   
      }
      clFinish(*myQueue);
   }
/* Optionally print information */
      if (step%iprint == 0)
         printf("\n%8d%12.6f%12.6f%12.6f%12.6f%12.6f",step, en, kn, vn,
                 pres, *tmpx);
}
