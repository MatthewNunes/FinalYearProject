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
#include "moldyn.h"
#include <time.h>
#include <cuda.h>
#include <math.h>
#include "matthew_CUDA.h"




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
   float *rx, *ry, *rz, *vx, *vy, *vz, *fx, *fy, *fz, *potentialPointer, *virialPointer, *virialArray, *potentialArray, *virialArrayTemp, *potentialArrayTemp, *kineticArray, *rxTemp, *ryTemp, *rzTemp;
   float *d_rx, *d_ry, *d_rz, *d_fx, *d_fy, *d_fz, *d_vx, *d_vy, *d_vz, *d_potential, *d_virial, *d_virialArray, *d_potentialArray, *d_kineticArray, *d_kinetic, *d_rxTemp, *d_ryTemp, *d_rzTemp;
   int *d_head, *d_list;
   float ace, acv, ack, acp, acesq, acvsq, acksq, acpsq, vg, kg, wg;
   int   *head, *list;
   int   natoms=0;
   int ierror;
   int jstart, step, itemp;
   float potential, virial, kinetic;
   float tmpx;
   int i, icell;
   cudaSetDevice(0);
   ierror = input_parameters (&sigma, &rcut, &dt, &eqtemp, &dens, &boxlx, &boxly, &boxlz, &sfx, &sfy, &sfz, &sr6, &vrcut, &dvrcut, &dvrc12, &freex, &nstep, &nequil, &iscale, &nc, &natoms, &mx, &my, &mz, &iprint);
   //printf ("\nReturned from input_parameters, natoms = %d\n", natoms);
  // CUDA_CHECK_RETURN(cudaSetDevice(1));
   //printf("\nmx = %d, my = %d, mz = %d\n",mx,my,mz);
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
   head= (int *)malloc((mx+2)*(my+2)*(mz+2)*sizeof(int));
   virialPointer = (float *)malloc(sizeof(float));
   potentialPointer = (float *)malloc(sizeof(float));
   kineticArray = (float *)malloc(sizeof(float) * ceil(natoms/(float)BLOCK_WIDTH));


   

   
  // printf ("\nFinished allocating memory\n");

   initialise_particles (rx, ry, rz, vx, vy, vz, nc);
 //  printf ("\nReturned from initialise_particles\n");

   loop_initialise(&ace, &acv, &ack, &acp, &acesq, &acvsq, &acksq, &acpsq, sigma, rcut, dt);
//   printf ("\nReturned from loop_initialise\n");

//   output_particles(rx,ry,rz,vx,vy,vz,fx,fy,fz,0);
   int numBlocks = ceil(natoms/(float)(BLOCK_WIDTH));
   CUDA_CHECK_RETURN(cudaMalloc((void **) &d_rx, 2*natoms * sizeof(float)));
   CUDA_CHECK_RETURN(cudaMalloc((void **) &d_ry, 2*natoms * sizeof(float)));
   CUDA_CHECK_RETURN(cudaMalloc((void **) &d_rz, 2*natoms * sizeof(float)));


   CUDA_CHECK_RETURN(cudaMemcpy(d_rx, rx, sizeof(float) * natoms * 2, cudaMemcpyHostToDevice));
   CUDA_CHECK_RETURN(cudaMemcpy(d_ry, ry, sizeof(float) * natoms * 2, cudaMemcpyHostToDevice));
   CUDA_CHECK_RETURN(cudaMemcpy(d_rz, rz, sizeof(float) * natoms * 2, cudaMemcpyHostToDevice));
   
   int numBlocks2 = ceil(natoms/(float)(BLOCK_WIDTH2));
   initialMovout<<<numBlocks2, BLOCK_WIDTH2>>>(d_rx, d_ry, d_rz, natoms);
   CUDA_CHECK_RETURN(cudaPeekAtLastError());
   CUDA_CHECK_RETURN(cudaDeviceSynchronize());
   CUDA_CHECK_RETURN(cudaMemcpy(rx, d_rx, sizeof(float) * natoms, cudaMemcpyDeviceToHost));
   CUDA_CHECK_RETURN(cudaMemcpy(ry, d_ry, sizeof(float) * natoms, cudaMemcpyDeviceToHost));
   CUDA_CHECK_RETURN(cudaMemcpy(rz, d_rz, sizeof(float) * natoms, cudaMemcpyDeviceToHost));
   movout (rx, ry, rz, vx, vy, vz, sfx, sfy, sfz, head, list, mx, my, mz, natoms); 
   virialArrayTemp = (float *)malloc(numBlocks * sizeof(float));
   potentialArrayTemp = (float *)malloc(numBlocks * sizeof(float));
   virialArray = (float *)malloc(numBlocks * sizeof(float));
   potentialArray = (float *)malloc(numBlocks * sizeof(float));
   int index;
   for (index = 0; index < numBlocks; index++)
   {
      virialArray[index] = (float)0;
      potentialArray[index] = (float)0;
      virialArrayTemp[index] = (float)0;
      potentialArrayTemp[index] = (float)0;
   }

   CUDA_CHECK_RETURN(cudaMalloc((void **) &d_fx, natoms * sizeof(float)));
   CUDA_CHECK_RETURN(cudaMalloc((void **) &d_fy, natoms * sizeof(float)));
   CUDA_CHECK_RETURN(cudaMalloc((void **) &d_fz, natoms * sizeof(float)));
   CUDA_CHECK_RETURN(cudaMalloc((void **) &d_vx, natoms * sizeof(float)));
   CUDA_CHECK_RETURN(cudaMalloc((void **) &d_vy, natoms * sizeof(float)));
   CUDA_CHECK_RETURN(cudaMalloc((void **) &d_vz, natoms * sizeof(float)));
   CUDA_CHECK_RETURN(cudaMalloc((void **) &d_head, (mx+2) * (my+2) * (mz+2) * sizeof(int)));
   CUDA_CHECK_RETURN(cudaMalloc((void **) &d_list, 2 * natoms * sizeof(int)));
   CUDA_CHECK_RETURN(cudaMalloc((void **) &d_potential, sizeof(float)));
   CUDA_CHECK_RETURN(cudaMalloc((void **) &d_virial, sizeof(float)));
   CUDA_CHECK_RETURN(cudaMalloc((void **) &d_virialArray, sizeof(float) * numBlocks));
   CUDA_CHECK_RETURN(cudaMalloc((void **) &d_potentialArray, sizeof(float) * numBlocks));
   CUDA_CHECK_RETURN(cudaMalloc((void **) &d_kineticArray, sizeof(float) * numBlocks));


   CUDA_CHECK_RETURN(cudaMemcpy(&d_rx[natoms], &rx[natoms], natoms * sizeof(float), cudaMemcpyHostToDevice));
   CUDA_CHECK_RETURN(cudaMemcpy(&d_ry[natoms], &ry[natoms], natoms * sizeof(float), cudaMemcpyHostToDevice));
   CUDA_CHECK_RETURN(cudaMemcpy(&d_rz[natoms], &rz[natoms], natoms * sizeof(float), cudaMemcpyHostToDevice));
   CUDA_CHECK_RETURN(cudaMemcpy(d_fx, fx, natoms * sizeof(float), cudaMemcpyHostToDevice));
   CUDA_CHECK_RETURN(cudaMemcpy(d_fy, fy, natoms * sizeof(float), cudaMemcpyHostToDevice));
   CUDA_CHECK_RETURN(cudaMemcpy(d_fz, fz, natoms * sizeof(float), cudaMemcpyHostToDevice));
   CUDA_CHECK_RETURN(cudaMemcpy(d_vx, vx, natoms * sizeof(float), cudaMemcpyHostToDevice));
   CUDA_CHECK_RETURN(cudaMemcpy(d_vy, vy, natoms * sizeof(float), cudaMemcpyHostToDevice));
   CUDA_CHECK_RETURN(cudaMemcpy(d_vz, vz, natoms * sizeof(float), cudaMemcpyHostToDevice));
   CUDA_CHECK_RETURN(cudaMemcpy(d_head, head, (mx+2) * (my+2) * (mz+2) * sizeof(int), cudaMemcpyHostToDevice));
   CUDA_CHECK_RETURN(cudaMemcpy(d_list, list, 2*natoms * sizeof(int), cudaMemcpyHostToDevice));
   CUDA_CHECK_RETURN(cudaMemcpy(d_potential, potentialPointer, sizeof(float), cudaMemcpyHostToDevice));
   CUDA_CHECK_RETURN(cudaMemcpy(d_virial, virialPointer, sizeof(float), cudaMemcpyHostToDevice));
   CUDA_CHECK_RETURN(cudaMemcpy(d_virialArray, virialArray, sizeof(float) * numBlocks, cudaMemcpyHostToDevice));
   CUDA_CHECK_RETURN(cudaMemcpy(d_potentialArray, potentialArray, sizeof(float) *numBlocks, cudaMemcpyHostToDevice));
   
   //check_cells(rx, ry, rz, head, list, mx, my, mz, natoms,0,0);

   long double elapsedTimeMoving = (float)0;
   long double elapsedTimeExecuting = (float)0;
   long unsigned int startTime;
   long unsigned int endTime;
   printf("numBlocks: %d\n", numBlocks);
   printf("BLOCK_WIDTH: %d\n", BLOCK_WIDTH);
   //copySigma(&sigma);
   //printf("main sigma: %f\n", sigma);
   
   int stepTemp = 0;
   int numBlocks3 = ceil(natoms/(float) BLOCK_WIDTH3);
   //copyToConstant(&sigma, &rcut, &vrcut, &dvrc12, &dvrcut, &mx, &my, &mz, &natoms, &stepTemp, &sfx, &sfy, &sfz);
   //force<<<numBlocks, BLOCK_WIDTH>>>(d_virialArray, d_potentialArray, d_potential, d_virial, d_rx, d_ry, d_rz, d_fx, d_fy, d_fz, d_head, d_list);
   force<<<numBlocks3, BLOCK_WIDTH3>>>(d_virialArray, d_potentialArray, d_rx, d_ry, d_rz, d_fx, d_fy, d_fz, sigma, rcut, vrcut, dvrc12, dvrcut, d_head, d_list, mx, my, mz, natoms, sfx, sfy, sfz);
   //force<<<numBlocks, BLOCK_WIDTH>>>(d_virialArray, d_potentialArray, d_potential, d_virial, d_rx, d_ry, d_rz, d_fx, d_fy, d_fz, sigma, rcut, vrcut, dvrc12, dvrcut, d_head, d_list, mx, my, mz, natoms, sfx, sfy, sfz);
   CUDA_CHECK_RETURN(cudaPeekAtLastError());
   CUDA_CHECK_RETURN(cudaDeviceSynchronize());
   CUDA_CHECK_RETURN(cudaMemcpy(potentialArrayTemp, d_potentialArray, sizeof(float) * numBlocks, cudaMemcpyDeviceToHost));
   CUDA_CHECK_RETURN(cudaMemcpy(virialArrayTemp, d_virialArray, sizeof(float) * numBlocks, cudaMemcpyDeviceToHost));
   int tempInd = 0;
   virial = 0.0;
   potential = 0.0;
   for (tempInd = 0; tempInd < numBlocks; tempInd++)
   {
      virial += virialArrayTemp[tempInd];
      potential += potentialArrayTemp[tempInd];
   }
   virial *= 48.0/3.0;
   potential *= 4.0;
   printf("virial: %f\n", virial);
   printf("potential: %f\n", potential);
  // printf ("\nReturned from force: potential = %f, virial = %f, kinetic = %f\n",potential, virial, kinetic);
//   output_particles(rx,ry,rz,vx,vy,vz,fx,fy,fz,0);

   
   for(step=1;step<=nstep;step++){
     // if(step>=85)printf ("\nStarted step %d\n",step);
      startTime = get_tick();
      movea<<<numBlocks, BLOCK_WIDTH>>>(d_rx, d_ry, d_rz, d_vx, d_vy, d_vz, d_fx, d_fy, d_fz, dt, natoms);
      CUDA_CHECK_RETURN(cudaPeekAtLastError());
      CUDA_CHECK_RETURN(cudaDeviceSynchronize());
      endTime = get_tick();
      elapsedTimeExecuting += (endTime - startTime);
      
      startTime = get_tick();
      initialMovout<<<numBlocks2, BLOCK_WIDTH2>>>(d_rx, d_ry, d_rz, natoms);
      CUDA_CHECK_RETURN(cudaPeekAtLastError());
      CUDA_CHECK_RETURN(cudaDeviceSynchronize());
      endTime = get_tick();
      elapsedTimeExecuting += (endTime - startTime);

      startTime = get_tick();
      CUDA_CHECK_RETURN(cudaMemcpy(rx, d_rx, sizeof(float) * natoms, cudaMemcpyDeviceToHost));
      CUDA_CHECK_RETURN(cudaMemcpy(ry, d_ry, sizeof(float) * natoms, cudaMemcpyDeviceToHost));
      CUDA_CHECK_RETURN(cudaMemcpy(rz, d_rz, sizeof(float) * natoms, cudaMemcpyDeviceToHost));
      endTime = get_tick();
      elapsedTimeMoving += (endTime - startTime);
      movout (rx, ry, rz, vx, vy, vz, sfx, sfy, sfz, head, list, mx, my, mz, natoms);
      
      startTime = get_tick();
      CUDA_CHECK_RETURN(cudaMemcpy(&d_rx[natoms], &rx[natoms], natoms * sizeof(float), cudaMemcpyHostToDevice));
      CUDA_CHECK_RETURN(cudaMemcpy(&d_ry[natoms], &ry[natoms], natoms * sizeof(float), cudaMemcpyHostToDevice));
      CUDA_CHECK_RETURN(cudaMemcpy(&d_rz[natoms], &rz[natoms], natoms * sizeof(float), cudaMemcpyHostToDevice));
      CUDA_CHECK_RETURN(cudaMemcpy(d_head, head, (mx+2)*(my+2)*(mz+2) * sizeof(int), cudaMemcpyHostToDevice));
      CUDA_CHECK_RETURN(cudaMemcpy(d_list, list, 2*natoms * sizeof(int), cudaMemcpyHostToDevice));
      endTime = get_tick();
      elapsedTimeMoving += (endTime - startTime);
      //copyToConstant(&sigma, &rcut, &vrcut, &dvrc12, &dvrcut, &mx, &my, &mz, &natoms, &stepTemp, &sfx, &sfy, &sfz);
      //printf("main sigma: %f\n", sigma);
      //force<<<numBlocks, BLOCK_WIDTH>>>(d_virialArray, d_potentialArray, d_potential, d_virial, d_rx, d_ry, d_rz, d_fx, d_fy, d_fz, d_head, d_list);
      startTime = get_tick();
      force<<<numBlocks3, BLOCK_WIDTH3>>>(d_virialArray, d_potentialArray, d_rx, d_ry, d_rz, d_fx, d_fy, d_fz, sigma, rcut, vrcut, dvrc12, dvrcut, d_head, d_list, mx, my, mz, natoms, sfx, sfy, sfz);
      //force<<<numBlocks, BLOCK_WIDTH>>>(d_virialArray, d_potentialArray, d_potential, d_virial, d_rx, d_ry, d_rz, d_fx, d_fy, d_fz, sigma, rcut, vrcut, dvrc12, dvrcut, d_head, d_list, mx, my, mz, natoms, sfx, sfy, sfz);
      CUDA_CHECK_RETURN(cudaPeekAtLastError());
      CUDA_CHECK_RETURN(cudaDeviceSynchronize());
      endTime = get_tick();
      elapsedTimeExecuting += (endTime - startTime);

      startTime = get_tick();
      CUDA_CHECK_RETURN(cudaMemcpy(potentialArrayTemp, d_potentialArray, sizeof(float) * numBlocks, cudaMemcpyDeviceToHost));
      CUDA_CHECK_RETURN(cudaMemcpy(virialArrayTemp, d_virialArray, sizeof(float) * numBlocks, cudaMemcpyDeviceToHost));
      endTime = get_tick();
      elapsedTimeMoving += (endTime - startTime);

      tempInd = 0;
      virial = 0.0;
      potential = 0.0;
      for (tempInd = 0; tempInd < numBlocks; tempInd++)
      {
         virial += virialArrayTemp[tempInd];
         potential += potentialArrayTemp[tempInd];
      }
      virial *= 48.0/3.0;
      potential *= 4.0;
      
      startTime = get_tick();
      moveb<<<numBlocks, BLOCK_WIDTH>>> (d_kineticArray, d_vx, d_vy, d_vz, d_fx, d_fy, d_fz, dt, natoms);
      CUDA_CHECK_RETURN(cudaPeekAtLastError());
      CUDA_CHECK_RETURN(cudaDeviceSynchronize());
      endTime = get_tick();
      elapsedTimeExecuting += (endTime - startTime);

      startTime = get_tick();
      CUDA_CHECK_RETURN(cudaMemcpy(kineticArray, d_kineticArray, sizeof(float) * numBlocks, cudaMemcpyDeviceToHost));
      endTime = get_tick();
      elapsedTimeMoving += (endTime - startTime);
      int sumInd = 0;
      for(sumInd = 1; sumInd < numBlocks; sumInd++)
      {
         kineticArray[0] += kineticArray[sumInd];
      }
     //ineticArray[0] *= 0.5;
      kinetic = kineticArray[0];
      kinetic *= 0.5;
      sum_energies (potential, kinetic, virial, &vg, &wg, &kg);
      hloop (kinetic, step, vg, wg, kg, freex, dens, sigma, eqtemp, &tmpx, &ace, &acv, &ack, &acp, &acesq, &acvsq, &acksq, &acpsq, vx, vy, vz, iscale, iprint, nequil, natoms, BLOCK_WIDTH, d_vx, d_vy, d_vz);

   }


   tidyup (ace, ack, acv, acp, acesq, acksq, acvsq, acpsq, nstep, nequil);
   elapsedTimeExecuting = elapsedTimeExecuting / (float) 1000;
   elapsedTimeMoving = elapsedTimeMoving / (float) 1000;
   printf("\n%Lf seconds have elapsed executing the kernel\n", elapsedTimeExecuting);
   printf("\n%Lf seconds have elapsed moving the data\n", elapsedTimeMoving);
   cudaFree(d_fx);
   cudaFree(d_fy);
   cudaFree(d_fz);
   cudaFree(d_rx);
   cudaFree(d_ry);
   cudaFree(d_rz);
   cudaFree(d_head);
   cudaFree(d_list);
   cudaFree(virialArray);
   cudaFree(potentialArray);
   cudaFree(virialPointer);
   cudaFree(potentialPointer);
   return 0;
}
