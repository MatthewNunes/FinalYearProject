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
#include <math.h>
#include "moldyn.h"
#include <time.h>

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
   int nstep, nequil, iscale, nc, mx, my, mz, iprint;
   float *rx, *ry, *rz, *vx, *vy, *vz, *fx, *fy, *fz, *potentialPointer, *virialPointer, *virialArray, *potentialArray;
   float *d_rx, *d_ry, *d_rz, *d_fx, *d_fy, *d_fz, *d_potential, *d_virial, *d_virialArray, *d_potentialArray;
   int *d_head, *d_list;
   float ace, acv, ack, acp, acesq, acvsq, acksq, acpsq, vg, kg, wg;
   int   head[NCELL], list[NMAX];
   int   natoms=0;
   int ierror;
   int jstart, step, itemp;
   float potential, virial, kinetic;
   float tmpx;
   int err;
   int i, icell;
   

   ierror = input_parameters (&sigma, &rcut, &dt, &eqtemp, &dens, &boxlx, &boxly, &boxlz, &sfx, &sfy, &sfz, &sr6, &vrcut, &dvrcut, &dvrc12, &freex, &nstep, &nequil, &iscale, &nc, &natoms, &mx, &my, &mz, &iprint);
   //printf ("\nReturned from input_parameters, natoms = %d\n", natoms);
   //float virialArray[natoms], potentialArray[natoms];
   //cudaSetDevice(1);
   rx = (float *)malloc(2*natoms*sizeof(float));
   ry = (float *)malloc(2*natoms*sizeof(float));
   rz = (float *)malloc(2*natoms*sizeof(float));
   vx = (float *)malloc(natoms*sizeof(float));
   vy = (float *)malloc(natoms*sizeof(float));
   vz = (float *)malloc(natoms*sizeof(float));
   fx = (float *)malloc(natoms*sizeof(float));
   fy = (float *)malloc(natoms*sizeof(float));
   fz = (float *)malloc(natoms*sizeof(float));   
   virialPointer = (float *)malloc(sizeof(float));
   potentialPointer = (float *)malloc(sizeof(float));
   int index = 0;
   /**

   */
   int numBlocks = ceil(natoms/(float)BLOCK_WIDTH);
   virialArray = (float *)malloc( numBlocks* sizeof(float));
   potentialArray = (float *)malloc(numBlocks * sizeof(float));
   for (index = 0; index < numBlocks; index++)
   {
      virialArray[index] = (float)0;
      potentialArray[index] = (float)0;
   }
   cudaMalloc((void **) &d_rx, 2*natoms*sizeof(float));
   cudaMalloc((void **) &d_ry, 2*natoms*sizeof(float));
   cudaMalloc((void **) &d_rz, 2*natoms*sizeof(float));
   cudaMalloc((void **) &d_fx, natoms*sizeof(float));
   cudaMalloc((void **) &d_fy, natoms*sizeof(float));
   cudaMalloc((void **) &d_fz, natoms*sizeof(float));
   cudaMalloc((void **) &d_head, NCELL*sizeof(int));
   cudaMalloc((void **) &d_list, NMAX*sizeof(int));
   cudaMalloc((void **) &d_potential, sizeof(float));
   cudaMalloc((void **) &d_virial, sizeof(float));
   cudaMalloc((void **) &d_virialArray, sizeof(float) * (numBlocks + 1));
   cudaMalloc((void **) &d_potentialArray, sizeof(float) * (numBlocks + 1));
   

   initialise_particles (rx, ry, rz, vx, vy, vz, nc);
   //printf ("\nReturned from initialise_particles\n");

   loop_initialise(&ace, &acv, &ack, &acp, &acesq, &acvsq, &acksq, &acpsq, sigma, rcut, dt);
   //printf ("\nReturned from loop_initialise\n");

   output_particles(rx,ry,rz,vx,vy,vz,fx,fy,fz,0);
      movout (rx, ry, rz, vx, vy, vz, sfx, sfy, sfz, head, list, mx, my, mz, natoms);
   //printf ("\nReturned from movout\n");
   //   check_cells(rx, ry, rz, head, list, mx, my, mz, natoms,0,0);
   
      *potentialPointer = (float)0;
      *virialPointer = (float)0;
      cudaMemcpy(d_rx, rx, 2*natoms*sizeof(float), cudaMemcpyHostToDevice);
      cudaMemcpy(d_ry, ry, 2*natoms*sizeof(float), cudaMemcpyHostToDevice);
      cudaMemcpy(d_rz, rz, 2*natoms*sizeof(float), cudaMemcpyHostToDevice);
      cudaMemcpy(d_fx, fx, natoms*sizeof(float), cudaMemcpyHostToDevice);
      cudaMemcpy(d_fy, fy, natoms*sizeof(float), cudaMemcpyHostToDevice);
      cudaMemcpy(d_fz, fz, natoms*sizeof(float), cudaMemcpyHostToDevice);
      cudaMemcpy(d_head, head, NCELL*sizeof(int), cudaMemcpyHostToDevice);
      cudaMemcpy(d_list, list, NMAX*sizeof(int), cudaMemcpyHostToDevice);
      cudaMemcpy(d_potential, potentialPointer, sizeof(float), cudaMemcpyHostToDevice);
      cudaMemcpy(d_virial, virialPointer, sizeof(float), cudaMemcpyHostToDevice);
      cudaMemcpy(d_virialArray, virialArray, sizeof(float) * numBlocks, cudaMemcpyHostToDevice);
      cudaMemcpy(d_potentialArray, potentialArray, sizeof(float) * numBlocks, cudaMemcpyHostToDevice);
      //float *virialTest = (float *)malloc(sizeof(float) * numBlocks);
      //float *potentialTest = (float *)malloc(sizeof(float) * numBlocks);
      long double elapsedTime = (float)0;
      long unsigned int startTime;
      long unsigned int endTime;
      startTime = get_tick();
      force<<<numBlocks, BLOCK_WIDTH, 2* BLOCK_WIDTH * sizeof(float)>>>(d_virialArray, d_potentialArray, d_potential, d_virial, d_rx, d_ry, d_rz, d_fx, d_fy, d_fz, sigma, rcut, vrcut, dvrc12, dvrcut, d_head, d_list, mx, my, mz, natoms,0, sfx, sfy, sfz);
      cudaError_t err = cudaPeekAtLastError();
      if (err != cudaSuccess)
      { 
         printf("Error: %s\n", cudaGetErrorString(err));
      }
      cudaDeviceSynchronize();
      endTime = get_tick();   
      elapsedTime += endTime - startTime;
      cudaMemcpy(fx, d_fx, natoms*sizeof(float), cudaMemcpyDeviceToHost);
      cudaMemcpy(fy, d_fy, natoms*sizeof(float), cudaMemcpyDeviceToHost);
      cudaMemcpy(fz, d_fz, natoms*sizeof(float), cudaMemcpyDeviceToHost);
      //cudaMemcpy(virialTest, d_virialArray, sizeof(float) * numBlocks, cudaMemcpyDeviceToHost);
      //cudaMemcpy(potentialTest, d_potentialArray, sizeof(float) * numBlocks, cudaMemcpyDeviceToHost);
      //cudaMemcpy(virialArray, d_virialArray);
      startTime = get_tick();
      finalResult<<<1,numBlocks>>>(d_potentialArray, d_virialArray, d_potential, d_virial, numBlocks);
      cudaError_t err = cudaPeekAtLastError();
      if (err != cudaSuccess)
      { 
         printf("Error: %s\n", cudaGetErrorString(err));
      }
      cudaDeviceSynchronize();
      endTime = get_tick();
      elapsedTime += endTime - startTime;
      cudaMemcpy(potentialPointer, d_potential, sizeof(float), cudaMemcpyDeviceToHost);
      cudaMemcpy(virialPointer, d_virial, sizeof(float), cudaMemcpyDeviceToHost);
      virial = *virialPointer;
      potential = *potentialPointer;
      
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
      cudaMemcpy(d_rx, rx, 2*natoms*sizeof(float), cudaMemcpyHostToDevice);
      cudaMemcpy(d_ry, ry, 2*natoms*sizeof(float), cudaMemcpyHostToDevice);
      cudaMemcpy(d_rz, rz, 2*natoms*sizeof(float), cudaMemcpyHostToDevice);
      cudaMemcpy(d_fx, fx, natoms*sizeof(float), cudaMemcpyHostToDevice);
      cudaMemcpy(d_fy, fy, natoms*sizeof(float), cudaMemcpyHostToDevice);
      cudaMemcpy(d_fz, fz, natoms*sizeof(float), cudaMemcpyHostToDevice);
      cudaMemcpy(d_head, head, NCELL*sizeof(int), cudaMemcpyHostToDevice);
      cudaMemcpy(d_list, list, NMAX*sizeof(int), cudaMemcpyHostToDevice);
      cudaMemcpy(d_potential, potentialPointer, sizeof(int), cudaMemcpyHostToDevice);
      cudaMemcpy(d_virial, virialPointer, sizeof(int), cudaMemcpyHostToDevice);
      cudaMemcpy(d_virialArray, virialArray, sizeof(float) * numBlocks, cudaMemcpyHostToDevice);
      cudaMemcpy(d_potentialArray, potentialArray, sizeof(float) * numBlocks, cudaMemcpyHostToDevice);
      //int numBlocks = ceil(natoms/(float)BLOCK_WIDTH);
      startTime = get_tick();
      force<<<numBlocks, BLOCK_WIDTH, 2*BLOCK_WIDTH * sizeof(float)>>>(d_virialArray, d_potentialArray, d_potential, d_virial, d_rx, d_ry, d_rz, d_fx, d_fy, d_fz, sigma, rcut, vrcut, dvrc12, dvrcut, d_head, d_list, mx, my, mz, natoms,0, sfx, sfy, sfz);
      cudaDeviceSynchronize();
      cudaError_t err = cudaPeekAtLastError();
      if (err != cudaSuccess)
      { 
         printf("Error: %s\n", cudaGetErrorString(err));
      }
      cudaDeviceSynchronize();
      endTime = get_tick();
      elapsedTime += endTime - startTime;
      cudaMemcpy(fx, d_fx, natoms*sizeof(float), cudaMemcpyDeviceToHost);
      cudaMemcpy(fy, d_fy, natoms*sizeof(float), cudaMemcpyDeviceToHost);
      cudaMemcpy(fz, d_fz, natoms*sizeof(float), cudaMemcpyDeviceToHost);
      //cudaMemcpy(virialTest, d_virialArray, sizeof(float) * numBlocks, cudaMemcpyDeviceToHost);
      //cudaMemcpy(potentialTest, d_potentialArray, sizeof(float) * numBlocks, cudaMemcpyDeviceToHost);
      startTime = get_tick();
      finalResult<<<1, numBlocks>>>(d_potentialArray, d_virialArray, d_potential, d_virial, numBlocks);
      cudaError_t err = cudaPeekAtLastError();
      if (err != cudaSuccess)
      { 
         printf("Error: %s\n", cudaGetErrorString(err));
      }
      cudaDeviceSynchronize();
      endTime = get_tick();
      elapsedTime += endTime - startTime;
      cudaMemcpy(potentialPointer, d_potential, sizeof(float), cudaMemcpyDeviceToHost);
      cudaMemcpy(virialPointer, d_virial, sizeof(float), cudaMemcpyDeviceToHost);
      virial = *virialPointer;
      potential = *potentialPointer;
      
      //potential = potentialTest[0];
      //virial = virialTest[0];
      //printf ("\nReturned from force: potential = %f, virial = %f, kinetic = %f\n",potential, virial, kinetic);      
      moveb (&kinetic, vx, vy, vz, fx, fy, fz, dt, natoms);
 //     check_cells(rx, ry, rz, head, list, mx, my, mz, natoms,step,step);
//   printf ("\nReturned from moveb: potential = %f, virial = %f, kinetic = %f\n",potential, virial, kinetic);
      sum_energies (potential, kinetic, virial, &vg, &wg, &kg);
      hloop (kinetic, step, vg, wg, kg, freex, dens, sigma, eqtemp, &tmpx, &ace, &acv, &ack, &acp, &acesq, &acvsq, &acksq, &acpsq, vx, vy, vz, iscale, iprint, nequil, natoms);
   }
   elapsedTime = elapsedTime / (float) 1000;
   tidyup (ace, ack, acv, acp, acesq, acksq, acvsq, acpsq, nstep, nequil);
   printf("\n%Lf seconds have elapsed\n", elapsedTime);
   
   return 0;
}
