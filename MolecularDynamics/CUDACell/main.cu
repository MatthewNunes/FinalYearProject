/*
FILE  main.c

DOC   This program simulates a three-dimensional Lennard-Jones fluid.

LANG  C plus MPI message passing.

HIS   1) Parallel version originally written by University of
HIS      Southampton.
HIS   2) Adopted to run on the Intel iPSC/2 computer at Daresbury
HIS      Laboratory, and placed in the public domain through the
HIS      CCP5 programme.
HIS   3) Adopted to run on the Intel Paragon and enhanced by David W.
HIS      Walker at Oak Ridge National Laboratory, Tennessess, USA
HIS   4) Converted to use MPI message passing library by David W. Walker
HIS      at University of Wales Cardiff in July 1997.
HIS   5) Converted from Fortran to C in November 1997.

*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <time.h>
#include "moldyn.h"
#include "matthew_CUDA.h"
#define BLOCK_WIDTH 512

long unsigned int get_tick()
{
   struct timespec ts;
   if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) return (0);
   return ts.tv_sec*(long int)1000 + ts.tv_nsec / (long int) 1000000;
}

int main ( int argc, char *argv[])
{
   float sigma, rcut, dt, eqtemp, dens, boxlx, boxly, boxlz, sfx, sfy, sfz, sr6, vrcut, dvrcut, dvrc12, freex; 
   int nstep, nequil, iscale, nc, mx, my, mz, iprint, map[MAPSIZ];
   float *rx, *ry, *rz, *vx, *vy, *vz, *fx, *fy, *fz, *pArray, *vArray;
   float *d_rx, *d_ry, *d_rz, *d_fx, *d_fy, *d_fz, *d_potential, *d_virial, *potentialPointer, *virialPointer, *d_pArray, *d_vArray;
   float ace, acv, ack, acp, acesq, acvsq, acksq, acpsq, vg, kg, wg;
   int   *head, *list;
   int *d_head, *d_list;
   int   natoms=0;
   int ierror;
   int jstart, step, itemp;
   float potential, virial, kinetic;
   float tmpx;
   int i, icell;
   int numBlocks;
   int *d_map;
   ierror = input_parameters (&sigma, &rcut, &dt, &eqtemp, &dens, &boxlx, &boxly, &boxlz, &sfx, &sfy, &sfz, &sr6, &vrcut, &dvrcut, &dvrc12, &freex, &nstep, &nequil, &iscale, &nc, &natoms, &mx, &my, &mz, &iprint, map);
   //printf ("\nReturned from input_parameters, natoms = %d\n", natoms);
   rx = (float *)malloc(2*natoms*sizeof(float));
   ry = (float *)malloc(2*natoms*sizeof(float));
   rz = (float *)malloc(2*natoms*sizeof(float));
   vx = (float *)malloc(natoms*sizeof(float));
   vy = (float *)malloc(natoms*sizeof(float));
   vz = (float *)malloc(natoms*sizeof(float));
   fx = (float *)malloc(natoms*sizeof(float));
   fy = (float *)malloc(natoms*sizeof(float));
   fz = (float *)malloc(natoms*sizeof(float));
   list = (int *)malloc(2*natoms*sizeof(int));
   head = (int *)malloc((mx+2)*(my+2)*(mz+2)*sizeof(int));
   pArray = (float *)malloc((mx+2)*(my+2)*(mz+2)*sizeof(float));
   vArray = (float *)malloc((mx+2)*(my+2)*(mz+2)*sizeof(float));
   potentialPointer = (float *)malloc(sizeof(float));
   virialPointer = (float *)malloc(sizeof(float));
   initialise_particles (rx, ry, rz, vx, vy, vz, nc);

   loop_initialise(&ace, &acv, &ack, &acp, &acesq, &acvsq, &acksq, &acpsq, sigma, rcut, dt);

   output_particles(rx,ry,rz,vx,vy,vz,fx,fy,fz,0);
      movout (rx, ry, rz, vx, vy, vz, sfx, sfy, sfz, head, list, mx, my, mz, natoms);

   numBlocks = ceil(((mx+2)*(my+2)*(mz+2))/ (float) BLOCK_WIDTH);
   *potentialPointer = 0.0;
   *virialPointer = 0.0;
   CUDA_CHECK_RETURN(cudaMalloc((void **) &d_rx, 2 * natoms * sizeof(float)));
   CUDA_CHECK_RETURN(cudaMalloc((void **) &d_ry, 2 * natoms * sizeof(float)));
   CUDA_CHECK_RETURN(cudaMalloc((void **) &d_rz, 2 * natoms * sizeof(float)));
   CUDA_CHECK_RETURN(cudaMalloc((void **) &d_fx, natoms * sizeof(float)));
   CUDA_CHECK_RETURN(cudaMalloc((void **) &d_fy, natoms * sizeof(float)));
   CUDA_CHECK_RETURN(cudaMalloc((void **) &d_fz, natoms * sizeof(float)));
   CUDA_CHECK_RETURN(cudaMalloc((void **) &d_head, (mx+2) * (my+2) * (mz+2) * sizeof(int)));
   CUDA_CHECK_RETURN(cudaMalloc((void **) &d_list, 2 * natoms * sizeof(int)));
   CUDA_CHECK_RETURN(cudaMalloc((void **) &d_potential, sizeof(float)));
   CUDA_CHECK_RETURN(cudaMalloc((void **) &d_virial, sizeof(float)));
   CUDA_CHECK_RETURN(cudaMalloc((void **) &d_map, sizeof(int) * MAPSIZ));
   CUDA_CHECK_RETURN(cudaMalloc((void **) &d_pArray, (mx+2) * (my+2) * (mz+2) * sizeof(float)));
   CUDA_CHECK_RETURN(cudaMalloc((void **) &d_vArray, (mx+2) * (my+2) * (mz+2) * sizeof(float)));

   CUDA_CHECK_RETURN(cudaMemcpy(d_rx, rx, 2 * natoms * sizeof(float), cudaMemcpyHostToDevice));
   CUDA_CHECK_RETURN(cudaMemcpy(d_ry, ry, 2 * natoms * sizeof(float), cudaMemcpyHostToDevice));
   CUDA_CHECK_RETURN(cudaMemcpy(d_rz, rz, 2 * natoms * sizeof(float), cudaMemcpyHostToDevice));
   CUDA_CHECK_RETURN(cudaMemcpy(d_fx, fx, natoms * sizeof(float), cudaMemcpyHostToDevice));
   CUDA_CHECK_RETURN(cudaMemcpy(d_fy, fy, natoms * sizeof(float), cudaMemcpyHostToDevice));
   CUDA_CHECK_RETURN(cudaMemcpy(d_fz, fz, natoms * sizeof(float), cudaMemcpyHostToDevice));
   CUDA_CHECK_RETURN(cudaMemcpy(d_head, head, (mx+2) * (my+2) * (mz+2) * sizeof(int), cudaMemcpyHostToDevice));
   CUDA_CHECK_RETURN(cudaMemcpy(d_list, list, 2 * natoms * sizeof(int), cudaMemcpyHostToDevice));
   CUDA_CHECK_RETURN(cudaMemcpy(d_potential, potentialPointer, sizeof(float), cudaMemcpyHostToDevice));
   CUDA_CHECK_RETURN(cudaMemcpy(d_virial, virialPointer, sizeof(float), cudaMemcpyHostToDevice));
   CUDA_CHECK_RETURN(cudaMemcpy(d_map, map, sizeof(int) * MAPSIZ, cudaMemcpyHostToDevice));
   CUDA_CHECK_RETURN(cudaMemcpy(d_pArray, pArray, sizeof(float) * (mx+2) * (my+2) * (mz+2)  , cudaMemcpyHostToDevice));
   CUDA_CHECK_RETURN(cudaMemcpy(d_vArray, vArray, sizeof(float) * (mx+2) * (my+2) * (mz+2)  , cudaMemcpyHostToDevice));
   long double elapsedTime = (float)0;
   long unsigned int startTime;
   long unsigned int endTime;
   startTime = get_tick();
   force<<<numBlocks, BLOCK_WIDTH>>>(d_pArray, d_vArray, d_potential, d_virial, d_rx, d_ry, d_rz, d_fx, d_fy, d_fz, sigma, rcut, vrcut, dvrc12, dvrcut, d_head, d_list, d_map, mx, my, mz, natoms, 0);
   CUDA_CHECK_RETURN(cudaPeekAtLastError());
   CUDA_CHECK_RETURN(cudaDeviceSynchronize());
   endTime = get_tick();
   elapsedTime += (endTime - startTime);
   CUDA_CHECK_RETURN(cudaMemcpy(fx, d_fx, natoms * sizeof(float), cudaMemcpyDeviceToHost));
   CUDA_CHECK_RETURN(cudaMemcpy(fy, d_fy, natoms * sizeof(float), cudaMemcpyDeviceToHost));
   CUDA_CHECK_RETURN(cudaMemcpy(fz, d_fz, natoms * sizeof(float), cudaMemcpyDeviceToHost));
   CUDA_CHECK_RETURN(cudaMemcpy(vArray, d_vArray, sizeof(float) * (mx+2) * (my+2) * (mz+2), cudaMemcpyDeviceToHost));
   CUDA_CHECK_RETURN(cudaMemcpy(pArray, d_pArray, sizeof(float) * (mx+2) * (my+2) * (mz+2), cudaMemcpyDeviceToHost));
   virial = 0.0;
   potential = 0.0;
   int tempInd =0;
   for (tempInd = 0; tempInd < ((mx+2) * (my+2) * (mz+2)); tempInd++)
   {
      virial += vArray[tempInd];
      potential += pArray[tempInd];
   }
   potential *= 4.0;
   virial    *= 48.0/3.0;
   //force (&potential, &virial, rx, ry, rz, fx, fy, fz, sigma, rcut, vrcut, dvrc12, dvrcut, head, list, map, mx, my, mz, natoms,0);
   //printf ("\nReturned from force: potential = %f, virial = %f, kinetic = %f\n",potential, virial, kinetic);
   output_particles(rx,ry,rz,vx,vy,vz,fx,fy,fz,0);


   for(step=1;step<=nstep;step++){
   //printf ("\nStarted step %d\n",step);
      movea (rx, ry, rz, vx, vy, vz, fx, fy, fz, dt, natoms);
//      check_cells(rx, ry, rz, head, list, mx, my, mz, natoms,step,step);
   //printf ("\nReturned from movea\n");
      movout (rx, ry, rz, vx, vy, vz, sfx, sfy, sfz, head, list, mx, my, mz, natoms);
  // printf ("\nReturned from movout\n");
  //    check_cells(rx, ry, rz, head, list, mx, my, mz, natoms,step,step);
      CUDA_CHECK_RETURN(cudaMemcpy(d_rx, rx, 2 * natoms * sizeof(float), cudaMemcpyHostToDevice));
      CUDA_CHECK_RETURN(cudaMemcpy(d_ry, ry, 2 * natoms * sizeof(float), cudaMemcpyHostToDevice));
      CUDA_CHECK_RETURN(cudaMemcpy(d_rz, rz, 2 * natoms * sizeof(float), cudaMemcpyHostToDevice));
      CUDA_CHECK_RETURN(cudaMemcpy(d_fx, fx, natoms * sizeof(float), cudaMemcpyHostToDevice));
      CUDA_CHECK_RETURN(cudaMemcpy(d_fy, fy, natoms * sizeof(float), cudaMemcpyHostToDevice));
      CUDA_CHECK_RETURN(cudaMemcpy(d_fz, fz, natoms * sizeof(float), cudaMemcpyHostToDevice));
      CUDA_CHECK_RETURN(cudaMemcpy(d_head, head, (mx+2) * (my+2) * (mz+2) * sizeof(int), cudaMemcpyHostToDevice));
      CUDA_CHECK_RETURN(cudaMemcpy(d_list, list, 2 * natoms * sizeof(int), cudaMemcpyHostToDevice));
      CUDA_CHECK_RETURN(cudaMemcpy(d_potential, potentialPointer, sizeof(float), cudaMemcpyHostToDevice));
      CUDA_CHECK_RETURN(cudaMemcpy(d_virial, virialPointer, sizeof(float), cudaMemcpyHostToDevice));
      CUDA_CHECK_RETURN(cudaMemcpy(d_map, map, sizeof(int) * MAPSIZ, cudaMemcpyHostToDevice));
      CUDA_CHECK_RETURN(cudaMemcpy(d_pArray, pArray, sizeof(float) * (mx+2) * (my+2) * (mz+2)  , cudaMemcpyHostToDevice));
      CUDA_CHECK_RETURN(cudaMemcpy(d_vArray, vArray, sizeof(float) * (mx+2) * (my+2) * (mz+2)  , cudaMemcpyHostToDevice));
      startTime = get_tick();
      force<<<numBlocks, BLOCK_WIDTH>>>(d_pArray, d_vArray, d_potential, d_virial, d_rx, d_ry, d_rz, d_fx, d_fy, d_fz, sigma, rcut, vrcut, dvrc12, dvrcut, d_head, d_list, d_map, mx, my, mz, natoms, 0);
      cudaDeviceSynchronize();
      endTime = get_tick();
      elapsedTime += (endTime - startTime);
      CUDA_CHECK_RETURN(cudaMemcpy(fx, d_fx, natoms * sizeof(float), cudaMemcpyDeviceToHost));
      CUDA_CHECK_RETURN(cudaMemcpy(fy, d_fy, natoms * sizeof(float), cudaMemcpyDeviceToHost));
      CUDA_CHECK_RETURN(cudaMemcpy(fz, d_fz, natoms * sizeof(float), cudaMemcpyDeviceToHost));
      CUDA_CHECK_RETURN(cudaMemcpy(vArray, d_vArray, sizeof(float) * (mx+2) * (my+2) * (mz+2), cudaMemcpyDeviceToHost));
      CUDA_CHECK_RETURN(cudaMemcpy(pArray, d_pArray, sizeof(float) * (mx+2) * (my+2) * (mz+2), cudaMemcpyDeviceToHost));
      virial = 0.0;
      potential = 0.0;
      int tempInd =0;
      for (tempInd = 0; tempInd < ((mx+2) * (my+2) * (mz+2)); tempInd++)
      {
         virial += vArray[tempInd];
         potential += pArray[tempInd];
      }
      potential *= 4.0;
      virial    *= 48.0/3.0;
   //printf ("\nReturned from force: potential = %f, virial = %f, kinetic = %f\n",potential, virial, kinetic);
      moveb (&kinetic, vx, vy, vz, fx, fy, fz, dt, natoms);
 //     check_cells(rx, ry, rz, head, list, mx, my, mz, natoms,step,step);
//   printf ("\nReturned from moveb: potential = %f, virial = %f, kinetic = %f\n",potential, virial, kinetic);
      sum_energies (potential, kinetic, virial, &vg, &wg, &kg);
      hloop (kinetic, step, vg, wg, kg, freex, dens, sigma, eqtemp, &tmpx, &ace, &acv, &ack, &acp, &acesq, &acvsq, &acksq, &acpsq, vx, vy, vz, iscale, iprint, nequil, natoms);
   }

   tidyup (ace, ack, acv, acp, acesq, acksq, acvsq, acpsq, nstep, nequil);
   elapsedTime = elapsedTime / (float) 1000;
   printf("\n%Lf seconds have elapsed\n", elapsedTime);

   return 0;
}
