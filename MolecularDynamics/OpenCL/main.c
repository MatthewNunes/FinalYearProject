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
#include <time.h>
#include <CL/cl.h>
#include <math.h>
#include "moldyn.h"

#define PROGRAM_FILE "force.cl"
#define FORCE_KERNEL "force"
#define REDUCTION_KERNEL "finalResult"
#define BLOCK_WIDTH 512

cl_device_id create_device() {

   cl_platform_id platform;
   cl_device_id dev;
   int err;

   /* Identify a platform */
   err = clGetPlatformIDs(1, &platform, NULL);
   if(err < 0) {
     perror("Couldn't identify a platform");
     exit(1);
   } 

   /* Access a device */
   err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &dev, NULL);
   if(err == CL_DEVICE_NOT_FOUND) {
     err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &dev, NULL);
   }
   if(err < 0) {
     perror("Couldn't access any devices");
     exit(1);   
   }

   return dev;
}

/* Create program from a file and compile it */
cl_program build_program(cl_context ctx, cl_device_id dev, const char* filename) {

   cl_program program;
   FILE *program_handle;
   char *program_buffer, *program_log;
   size_t program_size, log_size;
   int err;

   /* Read program file and place content into buffer */
   program_handle = fopen(filename, "r");
   if(program_handle == NULL) {
     perror("Couldn't find the program file");
     exit(1);
   }
   fseek(program_handle, 0, SEEK_END);
   program_size = ftell(program_handle);
   rewind(program_handle);
   program_buffer = (char*)malloc(program_size + 1);
   program_buffer[program_size] = '\0';
   fread(program_buffer, sizeof(char), program_size, program_handle);
   fclose(program_handle);

   /* Create program from file */
   program = clCreateProgramWithSource(ctx, 1, 
     (const char**)&program_buffer, &program_size, &err);
   if(err < 0) {
     perror("Couldn't create the program");
     exit(1);
   }
   free(program_buffer);

   /* Build program */
   err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
   if(err < 0) {

     /* Find size of log and print to std output */
     clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG, 
         0, NULL, &log_size);
     program_log = (char*) malloc(log_size + 1);
     program_log[log_size] = '\0';
     clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG, 
         log_size + 1, program_log, NULL);
     printf("%s\n", program_log);
     free(program_log);
     exit(1);
   }
   return program;
}


int main ( int argc, char *argv[])
{
   float sigma, rcut, dt, eqtemp, dens, boxlx, boxly, boxlz, sfx, sfy, sfz, sr6, vrcut, dvrcut, dvrc12, freex; 
   int nstep, nequil, iscale, nc, mx, my, mz, iprint;
   float *rx, *ry, *rz, *vx, *vy, *vz, *fx, *fy, *fz, *potentialPointer, *virialPointer, *virialArray, *potentialArray;
   //float *d_rx, *d_ry, *d_rz, *d_fx, *d_fy, *d_fz, *d_potential, *d_virial, *d_virialArray, *d_potentialArray;
   //int *d_head, *d_list;
   float ace, acv, ack, acp, acesq, acvsq, acksq, acpsq, vg, kg, wg;
   int   head[NCELL], list[NMAX];
   int   natoms=0;
   int ierror;
   int jstart, step, itemp;
   float potential, virial, kinetic;
   float tmpx;
   int i, icell;
   cl_int err;

   cl_device_id device;
   cl_context context;
   cl_command_queue queue;
   cl_program program;
   cl_kernel force_kernel;
   cl_kernel add_kernel;

   cl_mem d_rx, d_ry, d_rz, d_fx, d_fy, d_fz, d_head, d_list, d_potential, d_virial, d_virialArray, d_potentialArray;

   device = create_device();
   context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
   if(err < 0) {
     perror("Couldn't create a context");
     exit(1);   
   }
   queue = clCreateCommandQueue(context, device, NULL, &err);
   if(err < 0) {
     perror("Couldn't create a command queue");
     exit(1);   
   }

   /* Build the program */
   program = build_program(context, device, PROGRAM_FILE);

   force_kernel = clCreateKernel(program, FORCE_KERNEL, &err);
   if(err < 0) {
     perror("Couldn't create a kernel");
     exit(1);
   }
   add_kernel = clCreateKernel(program, REDUCTION_KERNEL, &err);
   if(err < 0) {
     perror("Couldn't create a kernel");
     exit(1);
   }

   ierror = input_parameters (&sigma, &rcut, &dt, &eqtemp, &dens, &boxlx, &boxly, &boxlz, &sfx, &sfy, &sfz, &sr6, &vrcut, &dvrcut, &dvrc12, &freex, &nstep, &nequil, &iscale, &nc, &natoms, &mx, &my, &mz, &iprint);
   //printf ("\nReturned from input_parameters, natoms = %d\n", natoms);
   //float virialArray[natoms], potentialArray[natoms];

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
   virialArray = (float *)malloc( (numBlocks + 1)* sizeof(float));
   potentialArray = (float *)malloc((numBlocks + 1) * sizeof(float));
   for (index = 0; index < numBlocks + 1; index++)
   {
      virialArray[index] = (float)0;
      potentialArray[index] = (float)0;
   }


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
   d_rx = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * natoms * 2, rx, &err);
   if(err < 0) {
     perror("Couldn't create a command queue");
     exit(1);   
   }   
   d_ry = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * natoms * 2, ry, &err);
   if(err < 0) {
     perror("Couldn't create a command queue");
     exit(1);   
   }   
   d_rz = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * natoms * 2, rz, &err);
   if(err < 0) {
     perror("Couldn't create a command queue");
     exit(1);   
   }   
   d_fx = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * natoms, fx, &err);
   if(err < 0) {
     perror("Couldn't create a command queue");
     exit(1);   
   }   
   d_fy = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * natoms, fy, &err);
   if(err < 0) {
     perror("Couldn't create a command queue");
     exit(1);   
   }   
   d_fz = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * natoms, fz, &err);
   if(err < 0) {
     perror("Couldn't create a command queue");
     exit(1);   
   }   
   d_head = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int) * NCELL, head, &err);
   if(err < 0) {
     perror("Couldn't create a command queue");
     exit(1);   
   }   
   d_list = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int) * NMAX, list, &err);
   if(err < 0) {
     perror("Couldn't create a command queue");
     exit(1);   
   }   
   d_potential = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float), potentialPointer, &err);
   if(err < 0) {
     perror("Couldn't create a command queue");
     exit(1);   
   }   
   d_virial = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float), virialPointer, &err);
   if(err < 0) {
     perror("Couldn't create a command queue");
     exit(1);   
   }   
   d_virialArray = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,  sizeof(float) * (numBlocks + 1), virialArray, &err);
   if(err < 0) {
     perror("Couldn't create a command queue");
     exit(1);   
   }   
   d_potentialArray = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,  sizeof(float) * (numBlocks + 1), potentialArray, &err);
   if(err < 0) {
     perror("Couldn't create a command queue");
     exit(1);   
   }
   //cl_mem d_rx, d_ry, d_rz, d_fx, d_fy, d_fz, d_head, d_list, d_potential, d_virial, d_virialArray, d_potentialArray;
   printf("Before allocating shared memory\n");
   err = clSetKernelArg(force_kernel, 0, sizeof(float) * BLOCK_WIDTH * 2, NULL);
   printf("Shared memory allocated\n");
   err |= clSetKernelArg(force_kernel, 1, sizeof(cl_mem), &d_virialArray);
   printf("I'm here 1\n");
   err |= clSetKernelArg(force_kernel, 2, sizeof(cl_mem), &d_potentialArray);
   printf("I'm here 2\n");
   err |= clSetKernelArg(force_kernel, 3, sizeof(cl_mem), &d_potential);
   printf("I'm here 3\n");
   err |= clSetKernelArg(force_kernel, 4, sizeof(cl_mem), &d_virial);
   printf("I'm here 4\n");
   err |= clSetKernelArg(force_kernel, 5, sizeof(cl_mem), &d_rx);
   printf("I'm here 5\n");
   err |= clSetKernelArg(force_kernel, 6, sizeof(cl_mem), &d_ry);
   printf("I'm here 6\n");
   err |= clSetKernelArg(force_kernel, 7, sizeof(cl_mem), &d_rz);
   printf("I'm here 7\n");
   err |= clSetKernelArg(force_kernel, 8, sizeof(cl_mem), &d_fx);
   printf("I'm here 8\n");
   err |= clSetKernelArg(force_kernel, 9, sizeof(cl_mem), &d_fy);
   printf("I'm here 9\n");
   err |= clSetKernelArg(force_kernel, 10, sizeof(cl_mem), &d_fz);
   printf("I'm here 10\n");
   err |= clSetKernelArg(force_kernel, 11, sizeof(float), &sigma);
   printf("I'm here 11\n");
   err |= clSetKernelArg(force_kernel, 12, sizeof(float), &rcut);
   printf("I'm here 12\n");
   err |= clSetKernelArg(force_kernel, 13, sizeof(float), &vrcut);
   printf("I'm here 13\n");
   err |= clSetKernelArg(force_kernel, 14, sizeof(float), &dvrc12);
   printf("I'm here 14\n");
   err |= clSetKernelArg(force_kernel, 15, sizeof(float), &dvrcut);
   printf("I'm here 15\n");
   err |= clSetKernelArg(force_kernel, 16, sizeof(cl_mem), &d_head);
   printf("I'm here 16\n");
   err |= clSetKernelArg(force_kernel, 17, sizeof(cl_mem), &d_list);
   printf("I'm here 17\n");
   err |= clSetKernelArg(force_kernel, 18, sizeof(int), &mx);
   printf("I'm here 18\n");
   err |= clSetKernelArg(force_kernel, 19, sizeof(int), &my);
   printf("I'm here 19\n");
   err |= clSetKernelArg(force_kernel, 20, sizeof(int), &mz);
   printf("I'm here 20\n");
   err |= clSetKernelArg(force_kernel, 21, sizeof(int), &natoms);
   printf("I'm here 21\n");
   err |= clSetKernelArg(force_kernel, 22, sizeof(int), &step);
   printf("I'm here 22\n");
   err |= clSetKernelArg(force_kernel, 23, sizeof(float), &sfx);
   printf("I'm here 23\n");
   err |= clSetKernelArg(force_kernel, 24, sizeof(float), &sfy);
   printf("I'm here 24\n");
   err |= clSetKernelArg(force_kernel, 25, sizeof(float), &sfz);
   printf("I'm here 25\n");
   err |= clSetKernelArg(add_kernel, 0, sizeof(float) * numBlocks * 2, NULL);
   printf("I'm here 1\n");
   err |= clSetKernelArg(add_kernel, 1, sizeof(cl_mem), &d_potentialArray);
   printf("I'm here 2\n");
   err |= clSetKernelArg(add_kernel, 2, sizeof(cl_mem), &d_virialArray);
   printf("I'm here 3\n");
   err |= clSetKernelArg(add_kernel, 3, sizeof(cl_mem), &d_potential);
   printf("I'm here 4\n");
   err |= clSetKernelArg(add_kernel, 4, sizeof(cl_mem), &d_virial);
   printf("I'm here 5\n");
   err |= clSetKernelArg(add_kernel, 5, sizeof(int), &numBlocks);
   if(err < 0) {
     printf("Couldn't set an argument for the transpose kernel");
     exit(1);   
   }
   printf("All kernel arguments set\n");
   int global_size[1];
   int local_size[1];
   global_size[0] = BLOCK_WIDTH * ceil(natoms / (float) BLOCK_WIDTH);
   local_size[0] = BLOCK_WIDTH;
   printf("Before force kernel\n");
   err = clEnqueueNDRangeKernel(queue, force_kernel, 1, NULL, global_size, local_size, 0, NULL, NULL);
   if(err < 0) {
     printf("Couldn't enqueue force kernel");
     printf("%d\n", err);
     exit(1);   
   }
   printf("I get here\n");   
   //(__shared float vpArray[], __global float *potentialArray, __global float *virialArray, __global float *potentialValue, __global float *virialValue, __private int n)
   err = clEnqueueReadBuffer(queue, d_fx, CL_TRUE, 0, sizeof(float)*natoms, fx, 0, NULL, NULL);
   err |= clEnqueueReadBuffer(queue, d_fy, CL_TRUE, 0, sizeof(float)*natoms, fy, 0, NULL, NULL);
   err |= clEnqueueReadBuffer(queue, d_fz, CL_TRUE, 0, sizeof(float)*natoms, fz, 0, NULL, NULL);
   if(err < 0) {
     printf("Couldn't read buffer");
     exit(1);   
   }   
   printf("I get here1\n");
   global_size[0] = ceil(natoms/(float) BLOCK_WIDTH);
   local_size[0] = ceil(natoms/(float) BLOCK_WIDTH);
   
   err = clEnqueueNDRangeKernel(queue, add_kernel, 1, NULL, global_size, local_size, 0, NULL, NULL);
   printf("I get here2\n");
   clFinish(queue);
   if(err < 0) {
     printf("Couldn't enqueue force kernel");
     exit(1);   
   }
   err = clEnqueueReadBuffer(queue, d_potential, CL_TRUE, 0, sizeof(float), potentialPointer, 0, NULL, NULL);
   err |= clEnqueueReadBuffer(queue, d_virial, CL_TRUE, 0, sizeof(float), virialPointer, 0, NULL, NULL);
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
      err = clEnqueueWriteBuffer(queue, d_rx, CL_TRUE, 0, 2*natoms*sizeof(float), rx, 0, NULL, NULL);
      err |= clEnqueueWriteBuffer(queue, d_ry, CL_TRUE, 0, sizeof(float) * natoms * 2, ry, 0, NULL, NULL);
      err |= clEnqueueWriteBuffer(queue, d_rz, CL_TRUE, 0, sizeof(float) * natoms * 2, rz, 0, NULL, NULL);
      err |= clEnqueueWriteBuffer(queue, d_fx, CL_TRUE, 0, sizeof(float) * natoms, fx, 0, NULL, NULL);
      err |= clEnqueueWriteBuffer(queue, d_fy, CL_TRUE, 0, sizeof(float) * natoms, fy, 0, NULL, NULL);
      err |= clEnqueueWriteBuffer(queue, d_fz, CL_TRUE, 0, sizeof(float) * natoms, fz, 0, NULL, NULL);
      err |= clEnqueueWriteBuffer(queue, d_head, CL_TRUE, 0, sizeof(int) * NCELL, d_head, 0, NULL, NULL);
      err |= clEnqueueWriteBuffer(queue, d_list, CL_TRUE, 0, sizeof(int) * NMAX, d_list, 0, NULL, NULL);
      err |= clEnqueueWriteBuffer(queue, d_virialArray, CL_TRUE, 0, sizeof(float) * (numBlocks + 1), virialArray, 0, NULL, NULL);
      err |= clEnqueueWriteBuffer(queue, d_potentialArray, CL_TRUE, 0, sizeof(float) * (numBlocks + 1), potentialArray, 0, NULL, NULL);
      if(err < 0) 
      {
        printf("Couldn't enqueue write buffer");
        exit(1);   
      }      
      global_size[0] = BLOCK_WIDTH * ceil(natoms / (float) BLOCK_WIDTH);
      local_size[0] = BLOCK_WIDTH;
   
      err = clEnqueueNDRangeKernel(queue, force_kernel, 1, NULL, global_size, local_size, 0, NULL, NULL);
      err = clEnqueueReadBuffer(queue, d_fx, CL_TRUE, 0, sizeof(float)*natoms, fx, 0, NULL, NULL);
      err |= clEnqueueReadBuffer(queue, d_fy, CL_TRUE, 0, sizeof(float)*natoms, fy, 0, NULL, NULL);
      err |= clEnqueueReadBuffer(queue, d_fz, CL_TRUE, 0, sizeof(float)*natoms, fz, 0, NULL, NULL);

      clFinish(queue);
      if(err < 0) {
        printf("Couldn't enqueue force kernel");
        exit(1);   
      }

      global_size[0] = ceil(natoms/(float) BLOCK_WIDTH);
      local_size[0] = ceil(natoms/(float) BLOCK_WIDTH);
   
      err = clEnqueueNDRangeKernel(queue, add_kernel, 1, NULL, global_size, local_size, 0, NULL, NULL);
      clFinish(queue);
      if(err < 0) {
        printf("Couldn't enqueue force kernel");
        exit(1);   
      }
      err = clEnqueueReadBuffer(queue, d_potential, CL_TRUE, 0, sizeof(float), potentialPointer, 0, NULL, NULL);
      err |= clEnqueueReadBuffer(queue, d_virial, CL_TRUE, 0, sizeof(float), virialPointer, 0, NULL, NULL);
      virial = *virialPointer;
      potential = *potentialPointer;    
      moveb (&kinetic, vx, vy, vz, fx, fy, fz, dt, natoms);
 //     check_cells(rx, ry, rz, head, list, mx, my, mz, natoms,step,step);
//   printf ("\nReturned from moveb: potential = %f, virial = %f, kinetic = %f\n",potential, virial, kinetic);
      sum_energies (potential, kinetic, virial, &vg, &wg, &kg);
      hloop (kinetic, step, vg, wg, kg, freex, dens, sigma, eqtemp, &tmpx, &ace, &acv, &ack, &acp, &acesq, &acvsq, &acksq, &acpsq, vx, vy, vz, iscale, iprint, nequil, natoms);
   }

   tidyup (ace, ack, acv, acp, acesq, acksq, acvsq, acpsq, nstep, nequil);
   
   return 0;
}
