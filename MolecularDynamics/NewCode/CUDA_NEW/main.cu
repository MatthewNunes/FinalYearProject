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
   float *rx, *ry, *rz, *vx, *vy, *vz, *fx, *fy, *fz, *potentialPointer, *virialPointer, *virialArray, *potentialArray, *virialArrayTemp, *potentialArrayTemp;
   float *d_rx, *d_ry, *d_rz, *d_fx, *d_fy, *d_fz, *d_potential, *d_virial, *d_virialArray, *d_potentialArray;
   int *d_head, *d_list;
   float ace, acv, ack, acp, acesq, acvsq, acksq, acpsq, vg, kg, wg;
   int   *head, *list;
   int   natoms=0;
   int ierror;
   int jstart, step, itemp;
   float potential, virial, kinetic;
   float tmpx;
   int i, icell;
   cudaSetDevice(2);
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

   

   
  // printf ("\nFinished allocating memory\n");

   initialise_particles (rx, ry, rz, vx, vy, vz, nc);
 //  printf ("\nReturned from initialise_particles\n");

   loop_initialise(&ace, &acv, &ack, &acp, &acesq, &acvsq, &acksq, &acpsq, sigma, rcut, dt);
//   printf ("\nReturned from loop_initialise\n");

//   output_particles(rx,ry,rz,vx,vy,vz,fx,fy,fz,0);
      movout (rx, ry, rz, vx, vy, vz, sfx, sfy, sfz, head, list, mx, my, mz, natoms);

   int numBlocks = ceil(natoms/(float)(BLOCK_WIDTH));
   virialArrayTemp = (float *)malloc(numBlocks * sizeof(float));
   potentialArrayTemp = (float *)malloc(numBlocks * sizeof(float));
   virialArray = (float *)malloc(numBlocks * sizeof(float));
   potentialArray = (float *)malloc(numBlocks * sizeof(float));
   int index;
   for (index = 0; index < numBlocks; index++)
   {
      virialArray[index] = (float)0;
      potentialArray[index] = (float)0;
   }
   CUDA_CHECK_RETURN(cudaMalloc((void **) &d_rx, 2*natoms * sizeof(float)));
   CUDA_CHECK_RETURN(cudaMalloc((void **) &d_ry, 2*natoms * sizeof(float)));
   CUDA_CHECK_RETURN(cudaMalloc((void **) &d_rz, 2*natoms * sizeof(float)));
   CUDA_CHECK_RETURN(cudaMalloc((void **) &d_fx, natoms * sizeof(float)));
   CUDA_CHECK_RETURN(cudaMalloc((void **) &d_fy, natoms * sizeof(float)));
   CUDA_CHECK_RETURN(cudaMalloc((void **) &d_fz, natoms * sizeof(float)));
   CUDA_CHECK_RETURN(cudaMalloc((void **) &d_head, (mx+2) * (my+2) * (mz+2) * sizeof(int)));
   CUDA_CHECK_RETURN(cudaMalloc((void **) &d_list, 2 * natoms * sizeof(int)));
   CUDA_CHECK_RETURN(cudaMalloc((void **) &d_potential, sizeof(float)));
   CUDA_CHECK_RETURN(cudaMalloc((void **) &d_virial, sizeof(float)));
   CUDA_CHECK_RETURN(cudaMalloc((void **) &d_virialArray, sizeof(float) * numBlocks));
   CUDA_CHECK_RETURN(cudaMalloc((void **) &d_potentialArray, sizeof(float) * numBlocks));

      //*potentialPointer = (float)0;
      //*virialPointer = (float)0;
      CUDA_CHECK_RETURN(cudaMemcpy(d_rx, rx, 2*natoms * sizeof(float), cudaMemcpyHostToDevice));
      CUDA_CHECK_RETURN(cudaMemcpy(d_ry, ry, 2*natoms * sizeof(float), cudaMemcpyHostToDevice));
      CUDA_CHECK_RETURN(cudaMemcpy(d_rz, rz, 2*natoms * sizeof(float), cudaMemcpyHostToDevice));
      CUDA_CHECK_RETURN(cudaMemcpy(d_fx, fx, natoms * sizeof(float), cudaMemcpyHostToDevice));
      CUDA_CHECK_RETURN(cudaMemcpy(d_fy, fy, natoms * sizeof(float), cudaMemcpyHostToDevice));
      CUDA_CHECK_RETURN(cudaMemcpy(d_fz, fz, natoms * sizeof(float), cudaMemcpyHostToDevice));
      CUDA_CHECK_RETURN(cudaMemcpy(d_head, head, (mx+2) * (my+2) * (mz+2) * sizeof(int), cudaMemcpyHostToDevice));
      CUDA_CHECK_RETURN(cudaMemcpy(d_list, list, 2*natoms * sizeof(int), cudaMemcpyHostToDevice));
      CUDA_CHECK_RETURN(cudaMemcpy(d_potential, potentialPointer, sizeof(float), cudaMemcpyHostToDevice));
      CUDA_CHECK_RETURN(cudaMemcpy(d_virial, virialPointer, sizeof(float), cudaMemcpyHostToDevice));
      CUDA_CHECK_RETURN(cudaMemcpy(d_virialArray, virialArray, sizeof(float) * numBlocks, cudaMemcpyHostToDevice));
      CUDA_CHECK_RETURN(cudaMemcpy(d_potentialArray, potentialArray, sizeof(float) *numBlocks, cudaMemcpyHostToDevice));
 //  printf ("\nReturned from movout\n");
   //   check_cells(rx, ry, rz, head, list, mx, my, mz, natoms,0,0);

      long double elapsedTime = (float)0;
      long unsigned int startTime;
      long unsigned int endTime;
      printf("numBlocks: %d\n", numBlocks);
      printf("BLOCK_WIDTH: %d\n", BLOCK_WIDTH);
      //copySigma(&sigma);
      //printf("main sigma: %f\n", sigma);
      /**
      int stepTemp = 0;
      copySigma(&sigma);
      copyRcut(&rcut);
      copyVrcut(&vrcut);
      copyDvrc12(&dvrc12);
      copyDvrcut(&dvrcut);
      copyMx(&mx);
      copyMy(&my);
      copyMz(&mz);
      copyNatoms(&natoms);
      copyStep(&stepTemp);
      copySfx(&sfx);
      copySfy(&sfy);
      copySfz(&sfz);
      */
      startTime = get_tick();
      force<<<numBlocks, BLOCK_WIDTH>>>(d_virialArray, d_potentialArray, d_potential, d_virial, d_rx, d_ry, d_rz, d_fx, d_fy, d_fz, sigma, rcut, vrcut, dvrc12, dvrcut, d_head, d_list, mx, my, mz, natoms, sfx, sfy, sfz);
      CUDA_CHECK_RETURN(cudaPeekAtLastError());
      CUDA_CHECK_RETURN(cudaDeviceSynchronize());
      endTime = get_tick();   
      elapsedTime += endTime - startTime;
      CUDA_CHECK_RETURN(cudaMemcpy(fx, d_fx, natoms * sizeof(float), cudaMemcpyDeviceToHost));
      CUDA_CHECK_RETURN(cudaMemcpy(fy, d_fy, natoms * sizeof(float), cudaMemcpyDeviceToHost));
      CUDA_CHECK_RETURN(cudaMemcpy(fz, d_fz, natoms * sizeof(float), cudaMemcpyDeviceToHost));
      CUDA_CHECK_RETURN(cudaMemcpy(potentialArrayTemp, d_potentialArray, sizeof(float) * numBlocks, cudaMemcpyDeviceToHost));
      CUDA_CHECK_RETURN(cudaMemcpy(virialArrayTemp, d_virialArray, sizeof(float) * numBlocks, cudaMemcpyDeviceToHost));
     // CUDA_CHECK_RETURN(cudaMemcpy(ppTemp, d_potential, sizeof(float), cudaMemcpyDeviceToHost));
     // CUDA_CHECK_RETURN(cudaMemcpy(vvTemp, d_virial, sizeof(float), cudaMemcpyDeviceToHost));
      //cudaMemcpy(virialTest, d_virialArray, sizeof(float) * numBlocks, cudaMemcpyDeviceToHost);
      //cudaMemcpy(potentialTest, d_potentialArray, sizeof(float) * numBlocks, cudaMemcpyDeviceToHost);
      //numBlocks +=1;
      //finalResult<<<numBlocks, 1>>>(d_potentialArray, d_virialArray, d_potential, d_virial, numBlocks);
      //CUDA_CHECK_RETURN(cudaPeekAtLastError());
      //CUDA_CHECK_RETURN(cudaDeviceSynchronize());
      //cudaMemcpy(potentialArrayTemp, d_potentialArray, sizeof(float) * (numBlocks + 1), cudaMemcpyDeviceToHost);
      //cudaMemcpy(virialArrayTemp, d_virialArray, sizeof(float) * (numBlocks + 1), cudaMemcpyDeviceToHost);
      int tempInd = 0;
      virial = 0.0;
      potential = 0.0;
      for (tempInd = 0; tempInd < numBlocks; tempInd++)
      {
         virial += virialArrayTemp[tempInd];
         potential += potentialArrayTemp[tempInd];
      }
      //printf("virial: %f\n", virial);
      //printf("potential: %f\n", potential);
     // virial = virialArrayTemp[0] * (48.0/3.0);
      //potential = potentialArrayTemp[0] * 4.0;
      virial *= 48.0/3.0;
      potential *= 4.0;
      
  // printf ("\nReturned from force: potential = %f, virial = %f, kinetic = %f\n",potential, virial, kinetic);
//   output_particles(rx,ry,rz,vx,vy,vz,fx,fy,fz,0);


   for(step=1;step<=nstep;step++){
     // if(step>=85)printf ("\nStarted step %d\n",step);
      movea (rx, ry, rz, vx, vy, vz, fx, fy, fz, dt, natoms);
//      check_cells(rx, ry, rz, head, list, mx, my, mz, natoms,step,step);
    //  if(step>85)printf ("\nReturned from movea\n");
      movout (rx, ry, rz, vx, vy, vz, sfx, sfy, sfz, head, list, mx, my, mz, natoms);

      CUDA_CHECK_RETURN(cudaMemcpy(d_rx, rx, 2*natoms * sizeof(float), cudaMemcpyHostToDevice));
      CUDA_CHECK_RETURN(cudaMemcpy(d_ry, ry, 2*natoms * sizeof(float), cudaMemcpyHostToDevice));
      CUDA_CHECK_RETURN(cudaMemcpy(d_rz, rz, 2*natoms * sizeof(float), cudaMemcpyHostToDevice));
      CUDA_CHECK_RETURN(cudaMemcpy(d_fx, fx, natoms * sizeof(float), cudaMemcpyHostToDevice));
      CUDA_CHECK_RETURN(cudaMemcpy(d_fy, fy, natoms * sizeof(float), cudaMemcpyHostToDevice));
      CUDA_CHECK_RETURN(cudaMemcpy(d_fz, fz, natoms * sizeof(float), cudaMemcpyHostToDevice));
      CUDA_CHECK_RETURN(cudaMemcpy(d_head, head, (mx+2)*(my+2)*(mz+2) * sizeof(int), cudaMemcpyHostToDevice));
      CUDA_CHECK_RETURN(cudaMemcpy(d_list, list, 2*natoms * sizeof(int), cudaMemcpyHostToDevice));
      CUDA_CHECK_RETURN(cudaMemcpy(d_virialArray, virialArray, sizeof(float) * numBlocks, cudaMemcpyHostToDevice));
      CUDA_CHECK_RETURN(cudaMemcpy(d_potentialArray, potentialArray, sizeof(float) * numBlocks, cudaMemcpyHostToDevice));

   //  if(step>85) printf ("\nReturned from movout\n");
  //    check_cells(rx, ry, rz, head, list, mx, my, mz, natoms,step,step);
      
      //numBlocks -= 1;
     // int stepTemp = 0;
      /**
      copySigma(&sigma);
      copyRcut(&rcut);
      copyVrcut(&vrcut);
      copyDvrc12(&dvrc12);
      copyDvrcut(&dvrcut);
      copyMx(&mx);
      copyMy(&my);
      copyMz(&mz);
      copyNatoms(&natoms);
      copyStep(&stepTemp);
      copySfx(&sfx);
      copySfy(&sfy);
      copySfz(&sfz);
      */
      //printf("main sigma: %f\n", sigma);
      startTime = get_tick();
      force<<<numBlocks, BLOCK_WIDTH>>>(d_virialArray, d_potentialArray, d_potential, d_virial, d_rx, d_ry, d_rz, d_fx, d_fy, d_fz, sigma, rcut, vrcut, dvrc12, dvrcut, d_head, d_list, mx, my, mz, natoms, sfx, sfy, sfz);
      CUDA_CHECK_RETURN(cudaPeekAtLastError());
      CUDA_CHECK_RETURN(cudaDeviceSynchronize());
      endTime = get_tick();
      elapsedTime += endTime - startTime;
      CUDA_CHECK_RETURN(cudaMemcpy(fx, d_fx, natoms * sizeof(float), cudaMemcpyDeviceToHost));
      CUDA_CHECK_RETURN(cudaMemcpy(fy, d_fy, natoms * sizeof(float), cudaMemcpyDeviceToHost));
      CUDA_CHECK_RETURN(cudaMemcpy(fz, d_fz, natoms * sizeof(float), cudaMemcpyDeviceToHost));
      CUDA_CHECK_RETURN(cudaMemcpy(potentialArrayTemp, d_potentialArray, sizeof(float) * numBlocks, cudaMemcpyDeviceToHost));
      CUDA_CHECK_RETURN(cudaMemcpy(virialArrayTemp, d_virialArray, sizeof(float) * numBlocks, cudaMemcpyDeviceToHost));
      //cudaMemcpy(virialTest, d_virialArray, sizeof(float) * numBlocks, cudaMemcpyDeviceToHost);
      //cudaMemcpy(potentialTest, d_potentialArray, sizeof(float) * numBlocks, cudaMemcpyDeviceToHost);
      //numBlocks +=1;
      //finalResult<<<numBlocks, 1>>>(d_potentialArray, d_virialArray, d_potential, d_virial, numBlocks);
      //CUDA_CHECK_RETURN(cudaPeekAtLastError());
      //CUDA_CHECK_RETURN(cudaDeviceSynchronize());
      //cudaMemcpy(potentialArrayTemp, d_potentialArray, sizeof(float) * (numBlocks + 1), cudaMemcpyDeviceToHost);
      //cudaMemcpy(virialArrayTemp, d_virialArray, sizeof(float) * (numBlocks + 1), cudaMemcpyDeviceToHost);
      tempInd = 0;
      virial = 0.0;
      potential = 0.0;
      for (tempInd = 0; tempInd < numBlocks; tempInd++)
      {
         virial += virialArrayTemp[tempInd];
         potential += potentialArrayTemp[tempInd];
      }
     // virial = virialArrayTemp[0] * (48.0/3.0);
      //potential = potentialArrayTemp[0] * 4.0;
     // printf("virial: %f\n", virial);
      //printf("potential: %f\n", potential);
      virial *= 48.0/3.0;
      potential *= 4.0;
//      if(step>85)printf ("\nReturned from force: potential = %f, virial = %f, kinetic = %f\n",potential, virial, kinetic);
 //     fflush(stdout);
      moveb (&kinetic, vx, vy, vz, fx, fy, fz, dt, natoms);
 //     check_cells(rx, ry, rz, head, list, mx, my, mz, natoms,step,step);
  // if(step>85)   printf ("\nReturned from moveb: potential = %f, virial = %f, kinetic = %f\n",potential, virial, kinetic);
      sum_energies (potential, kinetic, virial, &vg, &wg, &kg);
      hloop (kinetic, step, vg, wg, kg, freex, dens, sigma, eqtemp, &tmpx, &ace, &acv, &ack, &acp, &acesq, &acvsq, &acksq, &acpsq, vx, vy, vz, iscale, iprint, nequil, natoms);
   }

   tidyup (ace, ack, acv, acp, acesq, acksq, acvsq, acpsq, nstep, nequil);
   elapsedTime = elapsedTime / (float) 1000;
   printf("\n%Lf seconds have elapsed\n", elapsedTime);
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
