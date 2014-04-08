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
#include <math.h>
#include "moldyn.h"
#include <time.h>
#define PROGRAM_FILE "force.cl"
#define FORCE_KERNEL "force"
#define MOVEA_FILE "movea.cl"
#define MOVEA_KERNEL "movea"
#define MOVEB_FILE "moveb.cl"
#define MOVEB_KERNEL "moveb"
#define SCALET_FILE "scalet.cl"
#define SCALET_KERNEL "scalet"
#define INITIAL_MOVOOUT_FILE "initialMovout.cl"
#define INITIAL_MOVOUT_KERNEL "initialMovout"
#define REDUCTION_KERNEL "finalResult"

long unsigned int get_tick()
{
   struct timespec ts;
   if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) return (0);
   return ts.tv_sec*(long int)1000 + ts.tv_nsec / (long int) 1000000;
}

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
  // if(err < 0) {

     /* Find size of log and print to std output */
     clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG, 
         0, NULL, &log_size);
     program_log = (char*) malloc(log_size + 1);
     program_log[log_size] = '\0';
     clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG, 
         log_size + 1, program_log, NULL);
     printf("%s\n", program_log);
     free(program_log);
     //exit(1);
   //}
   return program;
}

int main ( int argc, char *argv[])
{
   float sigma, rcut, dt, eqtemp, dens, boxlx, boxly, boxlz, sfx, sfy, sfz, sr6, vrcut, dvrcut, dvrc12, freex; 
   int nstep, nequil, iscale, nc, mx, my, mz, iprint;
   float *rx, *ry, *rz, *vx, *vy, *vz, *fx, *fy, *fz, *potentialPointer, *virialPointer, *virialArray, *potentialArray, *virialArrayTemp, *potentialArrayTemp, *kineticArray;
   float ace, acv, ack, acp, acesq, acvsq, acksq, acpsq, vg, kg, wg;
   int   *head, *list;
   int   natoms=0;
   int ierror;
   int jstart, step, itemp;
   float potential, virial, kinetic;
   float tmpx;
   int i, icell;
   cl_int err;
   size_t global_size[1];
   size_t local_size[1];

   cl_device_id device;
   cl_context context;
   cl_command_queue queue;
   cl_program program, moveaProgram, movebProgram, scaletProgram, initialMovoutProgram;
   cl_kernel force_kernel, moveaKernel, movebKernel, scaletKernel, initialMovoutKernel;
   cl_kernel add_kernel;

   cl_mem d_rx, d_ry, d_rz, d_fx, d_fy, d_fz, d_head, d_list, d_potential, d_virial, d_virialArray, d_potentialArray, d_kineticArray, d_vx, d_vy, d_vz;

   ierror = input_parameters (&sigma, &rcut, &dt, &eqtemp, &dens, &boxlx, &boxly, &boxlz, &sfx, &sfy, &sfz, &sr6, &vrcut, &dvrcut, &dvrc12, &freex, &nstep, &nequil, &iscale, &nc, &natoms, &mx, &my, &mz, &iprint);
   //printf ("\nReturned from input_parameters, natoms = %d\n", natoms);
   device = create_device();
   context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
   if(err < 0) {
     perror("Couldn't create a context");
     exit(1);   
   }
   queue = clCreateCommandQueue(context, device, 0, &err);
   if(err < 0) {
     perror("Couldn't create a command queue");
     exit(1);   
   }
   /* Build the program */
   program = build_program(context, device, PROGRAM_FILE);
   moveaProgram = build_program(context, device, MOVEA_FILE);
   movebProgram = build_program(context, device, MOVEB_FILE);
   scaletProgram = build_program(context, device, SCALET_FILE);
   initialMovoutProgram = build_program(context, device, INITIAL_MOVOOUT_FILE);

   force_kernel = clCreateKernel(program, FORCE_KERNEL, &err);
   if(err < 0) {
     perror("Couldn't create a force kernel");
     exit(1);
   }   
   moveaKernel = clCreateKernel(moveaProgram, MOVEA_KERNEL, &err);
   if(err < 0) {
     perror("Couldn't create a movea kernel");
     exit(1);
   }
   movebKernel = clCreateKernel(movebProgram, MOVEB_KERNEL, &err);
   if(err < 0) {
     perror("Couldn't create a moveb kernel");
     exit(1);
   }
   scaletKernel = clCreateKernel(scaletProgram, SCALET_KERNEL, &err);
   if(err < 0) {
     perror("Couldn't create a scalet kernel");
     exit(1);
   }
   initialMovoutKernel = clCreateKernel(initialMovoutProgram, INITIAL_MOVOUT_KERNEL, &err);
   if(err < 0) {
     perror("Couldn't create a initialMovout kernel");
     exit(1);
   }
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
   int index = 0;

   int numBlocks = ceil(natoms/(float)BLOCK_WIDTH);
   kineticArray = (float *)malloc(sizeof(float) * numBlocks);   
   virialArray = (float *)malloc( (numBlocks)* sizeof(float));
   potentialArray = (float *)malloc((numBlocks) * sizeof(float));
   virialArrayTemp = (float *)malloc(numBlocks * sizeof(float));
   potentialArrayTemp = (float *)malloc(numBlocks * sizeof(float));
   for (index = 0; index < numBlocks; index++)
   {
      virialArray[index] = (float)0;
      potentialArray[index] = (float)0;
   }


  // printf ("\nFinished allocating memory\n");

   initialise_particles (rx, ry, rz, vx, vy, vz, nc);
 //  printf ("\nReturned from initialise_particles\n")


   loop_initialise(&ace, &acv, &ack, &acp, &acesq, &acvsq, &acksq, &acpsq, sigma, rcut, dt);
//   printf ("\nReturned from loop_initialise\n");
   d_rx = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * natoms * 2, rx, &err);
   if(err < 0) {
     perror("Couldn't create d_rx buffer");
     exit(1);   
   }   
   d_ry = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * natoms * 2, ry, &err);
   if(err < 0) {
     perror("Couldn't create d_ry buffer");
     exit(1);   
   }   
   d_rz = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * natoms * 2, rz, &err);
   if(err < 0) {
     perror("Couldn't create d_rz buffer");
     exit(1);   
   }    
   err = clSetKernelArg(initialMovoutKernel, 0, sizeof(cl_mem), &d_rx);
   err |= clSetKernelArg(initialMovoutKernel, 1, sizeof(cl_mem), &d_ry);
   err |= clSetKernelArg(initialMovoutKernel, 2, sizeof(cl_mem), &d_rz);
   err |= clSetKernelArg(initialMovoutKernel, 3, sizeof(natoms), &natoms);
   if(err < 0) {
     printf("Couldn't set an argument for the initial movout kernel");
     exit(1);   
   }
   global_size[0] = BLOCK_WIDTH * ceil(natoms / (float) BLOCK_WIDTH);
   local_size[0] = BLOCK_WIDTH;
   err = clEnqueueNDRangeKernel(queue, initialMovoutKernel, 1, NULL, global_size, local_size, 0, NULL, NULL);
   clFinish(queue);
   err = clEnqueueReadBuffer(queue, d_rx, CL_TRUE, 0, sizeof(float) * natoms, rx, 0, NULL, NULL);
   err |= clEnqueueReadBuffer(queue, d_ry, CL_TRUE, 0, sizeof(float) * natoms, ry, 0, NULL, NULL);
   err |= clEnqueueReadBuffer(queue, d_rz, CL_TRUE, 0, sizeof(float) * natoms, rz, 0, NULL, NULL);
   if(err < 0) {
     printf("Couldn't read rx, ry and rz buffer\n");
     printf("%d\n",err );
     printf("CL_INVALID_COMMAND_QUEUE: %d\n",CL_INVALID_COMMAND_QUEUE);
     printf("CL_INVALID_CONTEXT: %d\n", CL_INVALID_CONTEXT);
     printf("CL_INVALID_MEM_OBJECT: %d\n", CL_INVALID_MEM_OBJECT);
     printf("CL_INVALID_VALUE: %d\n",CL_INVALID_VALUE);
     printf("CL_INVALID_EVENT_WAIT_LIST: %d\n", CL_INVALID_EVENT_WAIT_LIST);
     printf("CL_MEM_OBJECT_ALLOCATION_FAILURE: %d\n",CL_MEM_OBJECT_ALLOCATION_FAILURE);
     printf("CL_OUT_OF_HOST_MEMORY: %d\n", CL_OUT_OF_HOST_MEMORY);
     exit(1);   
   }
//   output_particles(rx,ry,rz,vx,vy,vz,fx,fy,fz,0);
      movout (rx, ry, rz, vx, vy, vz, sfx, sfy, sfz, head, list, mx, my, mz, natoms);
 //  printf ("\nReturned from movout\n");
   //   check_cells(rx, ry, rz, head, list, mx, my, mz, natoms,0,0);
   *potentialPointer = (float)0;
   *virialPointer = (float)0;
   d_kineticArray = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * numBlocks, kineticArray, &err);
   if(err < 0) 
   {
     perror("Couldn't create a kinetic buffer");
     exit(1);   
   }
   d_vx = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * natoms, vx, &err);
   if(err < 0) 
   {
     perror("Couldn't create a vx buffer");
     exit(1);   
   }
   d_vy = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * natoms, vy, &err);
   if(err < 0) 
   {
     perror("Couldn't create a vy buffer");
     exit(1);   
   }   
   d_vz = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * natoms, vz, &err);
   if(err < 0) 
   {
     perror("Couldn't create a vz buffer");
     exit(1);   
   }
   d_rx = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * natoms * 2, rx, &err);
   if(err < 0) {
     perror("Couldn't create a rx buffer");
     exit(1);   
   }   
   d_ry = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * natoms * 2, ry, &err);
   if(err < 0) {
     perror("Couldn't create a ry buffer");
     exit(1);   
   }   
   d_rz = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * natoms * 2, rz, &err);
   if(err < 0) {
     perror("Couldn't create a rz buffer");
     exit(1);   
   }   
   d_fx = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * natoms, fx, &err);
   if(err < 0) {
     perror("Couldn't create a fx buffer");
     exit(1);   
   }   
   d_fy = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * natoms, fy, &err);
   if(err < 0) {
     perror("Couldn't create a fy buffer");
     exit(1);   
   }   
   d_fz = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * natoms, fz, &err);
   if(err < 0) {
     perror("Couldn't create a fz buffer");
     exit(1);   
   }   
   d_head = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, (mx+2)*(my+2)*(mz+2)*sizeof(int), head, &err);
   if(err < 0) {
     perror("Couldn't create a head buffer");
     exit(1);   
   }   
   d_list = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 2*natoms*sizeof(int), list, &err);
   if(err < 0) {
     perror("Couldn't create a list buffer");
     exit(1);   
   }   
   d_virialArray = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,  sizeof(float) * (numBlocks), virialArray, &err);
   if(err < 0) {
     perror("Couldn't create a virial buffer");
     exit(1);   
   }   
   d_potentialArray = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,  sizeof(float) * (numBlocks), potentialArray, &err);
   if(err < 0) {
     perror("Couldn't create a potential buffer");
     exit(1);   
   }

    
   err = clSetKernelArg(force_kernel, 0, sizeof(cl_mem), &d_virialArray);
   err |= clSetKernelArg(force_kernel, 1, sizeof(cl_mem), &d_potentialArray);
   err |= clSetKernelArg(force_kernel, 2, sizeof(cl_mem), &d_rx);
   err |= clSetKernelArg(force_kernel, 3, sizeof(cl_mem), &d_ry);
   err |= clSetKernelArg(force_kernel, 4, sizeof(cl_mem), &d_rz);
   err |= clSetKernelArg(force_kernel, 5, sizeof(cl_mem), &d_fx);
   err |= clSetKernelArg(force_kernel, 6, sizeof(cl_mem), &d_fy);
   err |= clSetKernelArg(force_kernel, 7, sizeof(cl_mem), &d_fz);
   err |= clSetKernelArg(force_kernel, 8, sizeof(sigma), &sigma);
   err |= clSetKernelArg(force_kernel, 9, sizeof(rcut), &rcut);
   err |= clSetKernelArg(force_kernel, 10, sizeof(vrcut), &vrcut);
   err |= clSetKernelArg(force_kernel, 11, sizeof(dvrc12), &dvrc12);
   err |= clSetKernelArg(force_kernel, 12, sizeof(dvrcut), &dvrcut);
   err |= clSetKernelArg(force_kernel, 13, sizeof(cl_mem), &d_head);
   err |= clSetKernelArg(force_kernel, 14, sizeof(cl_mem), &d_list);
   err |= clSetKernelArg(force_kernel, 15, sizeof(mx), &mx);
   err |= clSetKernelArg(force_kernel, 16, sizeof(my), &my);
   err |= clSetKernelArg(force_kernel, 17, sizeof(mz), &mz);
   err |= clSetKernelArg(force_kernel, 18, sizeof(natoms), &natoms);
   err |= clSetKernelArg(force_kernel, 19, sizeof(sfx), &sfx);
   err |= clSetKernelArg(force_kernel, 20, sizeof(sfy), &sfy);
   err |= clSetKernelArg(force_kernel, 21, sizeof(sfz), &sfz);



   if(err < 0) {
     printf("Couldn't set an argument for the force kernel");
     exit(1);   
   }
   //size_t max_size;
   //clGetKernelWorkGroupInfo(add_kernel, device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(max_size), &max_size, NULL);
   //printf("\nMAX SIZE: %d\n", max_size);
   


   global_size[0] = BLOCK_WIDTH * ceil(natoms / (float) BLOCK_WIDTH);
   local_size[0] = BLOCK_WIDTH;
   long double elapsedTime = (float)0.0;
   long unsigned int startTime;
   long unsigned int endTime;
   
   err = clEnqueueNDRangeKernel(queue, force_kernel, 1, NULL, global_size, local_size, 0, NULL, NULL);
   if(err < 0) {
     printf("Couldn't enqueue force kernel\n");
     printf("%d\n", err);
     printf("CL_INVALID_PROGRAM_EXECUTABLE: %d\n", CL_INVALID_PROGRAM_EXECUTABLE);
     printf("CL_INVALID_COMMAND_QUEUE: %d\n",CL_INVALID_COMMAND_QUEUE );
     printf("CL_INVALID_KERNEL: %d\n", CL_INVALID_KERNEL);
     printf("CL_INVALID_CONTEXT: %d\n", CL_INVALID_CONTEXT);
     printf("CL_INVALID_KERNEL_ARGS: %d\n", CL_INVALID_KERNEL_ARGS);
     printf("CL_INVALID_WORK_DIMENSION: %d\n", CL_INVALID_WORK_DIMENSION);
     printf("CL_INVALID_GLOBAL_WORK_SIZE: %d\n", CL_INVALID_GLOBAL_WORK_SIZE);
     printf("CL_INVALID_GLOBAL_OFFSET: %d\n", CL_INVALID_GLOBAL_OFFSET);
     printf("CL_INVALID_WORK_GROUP_SIZE: %d\n", CL_INVALID_WORK_GROUP_SIZE);
     exit(1);   
   }


   err = clEnqueueReadBuffer(queue, d_virialArray, CL_TRUE, 0, sizeof(float) * numBlocks, virialArrayTemp, 0, NULL, NULL);
   err |= clEnqueueReadBuffer(queue, d_potentialArray, CL_TRUE, 0, sizeof(float) * numBlocks, potentialArrayTemp, 0, NULL, NULL);
   if(err < 0) {
     printf("Couldn't read virial and potential buffer\n");
     printf("%d\n",err );
     printf("CL_INVALID_COMMAND_QUEUE: %d\n",CL_INVALID_COMMAND_QUEUE);
     printf("CL_INVALID_CONTEXT: %d\n", CL_INVALID_CONTEXT);
     printf("CL_INVALID_MEM_OBJECT: %d\n", CL_INVALID_MEM_OBJECT);
     printf("CL_INVALID_VALUE: %d\n",CL_INVALID_VALUE);
     printf("CL_INVALID_EVENT_WAIT_LIST: %d\n", CL_INVALID_EVENT_WAIT_LIST);
     printf("CL_MEM_OBJECT_ALLOCATION_FAILURE: %d\n",CL_MEM_OBJECT_ALLOCATION_FAILURE);
     printf("CL_OUT_OF_HOST_MEMORY: %d\n", CL_OUT_OF_HOST_MEMORY);
     exit(1);   
   }
   clFinish(queue);
   virial = 0.0;
   potential = 0.0;
   int tempInd = 0;
   for (tempInd =0; tempInd < numBlocks; tempInd++)
   {
      potential += potentialArrayTemp[tempInd];
      virial += virialArrayTemp[tempInd];
   }
   virial *= 48.0/3.0;
   potential *= 4.0;

  // printf ("\nReturned from force: potential = %f, virial = %f, kinetic = %f\n",potential, virial, kinetic);
//   output_particles(rx,ry,rz,vx,vy,vz,fx,fy,fz,0);
   startTime = get_tick();
   for(step=1;step<=nstep;step++){
     // if(step>=85)printf ("\nStarted step %d\n",step);
    
    err = clSetKernelArg(moveaKernel, 0, sizeof(cl_mem), &d_rx);
    err |= clSetKernelArg(moveaKernel, 1, sizeof(cl_mem), &d_ry);
    err |= clSetKernelArg(moveaKernel, 2, sizeof(cl_mem), &d_rz);
    err |= clSetKernelArg(moveaKernel, 3, sizeof(cl_mem), &d_vx);
    err |= clSetKernelArg(moveaKernel, 4, sizeof(cl_mem), &d_vy);
    err |= clSetKernelArg(moveaKernel, 5, sizeof(cl_mem), &d_vz);
    err |= clSetKernelArg(moveaKernel, 6, sizeof(cl_mem), &d_fx);
    err |= clSetKernelArg(moveaKernel, 7, sizeof(cl_mem), &d_fy);
    err |= clSetKernelArg(moveaKernel, 8, sizeof(cl_mem), &d_fz);
    err |= clSetKernelArg(moveaKernel, 9, sizeof(dt), &dt);
    err |= clSetKernelArg(moveaKernel, 10, sizeof(natoms), &natoms);  
    if(err < 0) {
      perror("Couldn't set kernel arguments for moveaKernel");
      exit(1);   
      }

      err = clEnqueueNDRangeKernel(queue, moveaKernel, 1, NULL, global_size, local_size, 0, NULL, NULL);
      if(err < 0) {
        perror("Couldn't enqueue moveaKernel");
        exit(1);   
      }
      clFinish(queue);
    
      err = clSetKernelArg(initialMovoutKernel, 0, sizeof(cl_mem), &d_rx);
      err |= clSetKernelArg(initialMovoutKernel, 1, sizeof(cl_mem), &d_ry);
      err |= clSetKernelArg(initialMovoutKernel, 2, sizeof(cl_mem), &d_rz);
      err |= clSetKernelArg(initialMovoutKernel, 3, sizeof(natoms), &natoms);
      if(err < 0) {
       printf("Couldn't set an argument for the initialMovout kernel");
       exit(1);   
      }
      err = clEnqueueNDRangeKernel(queue, initialMovoutKernel, 1, NULL, global_size, local_size, 0, NULL, NULL);
      if(err < 0) {
        perror("Couldn't enqueue initialMovoutKernel");
        exit(1);   
      }
      clFinish(queue);
      
      err = clEnqueueReadBuffer(queue, d_rx, CL_TRUE, 0, sizeof(float) * natoms, rx, 0, NULL, NULL);
      err |= clEnqueueReadBuffer(queue, d_ry, CL_TRUE, 0, sizeof(float) * natoms, ry, 0, NULL, NULL);
      err |= clEnqueueReadBuffer(queue, d_rz, CL_TRUE, 0, sizeof(float) * natoms, rz, 0, NULL, NULL);
      if(err < 0) {
       printf("Couldn't read rx, ry and rz buffer\n");
       printf("%d\n",err );
       printf("CL_INVALID_COMMAND_QUEUE: %d\n",CL_INVALID_COMMAND_QUEUE);
       printf("CL_INVALID_CONTEXT: %d\n", CL_INVALID_CONTEXT);
       printf("CL_INVALID_MEM_OBJECT: %d\n", CL_INVALID_MEM_OBJECT);
       printf("CL_INVALID_VALUE: %d\n",CL_INVALID_VALUE);
       printf("CL_INVALID_EVENT_WAIT_LIST: %d\n", CL_INVALID_EVENT_WAIT_LIST);
       printf("CL_MEM_OBJECT_ALLOCATION_FAILURE: %d\n",CL_MEM_OBJECT_ALLOCATION_FAILURE);
       printf("CL_OUT_OF_HOST_MEMORY: %d\n", CL_OUT_OF_HOST_MEMORY);
       exit(1);   
      }
      clFinish(queue);
//      check_cells(rx, ry, rz, head, list, mx, my, mz, natoms,step,step);
    //  if(step>85)printf ("\nReturned from movea\n");
      movout (rx, ry, rz, vx, vy, vz, sfx, sfy, sfz, head, list, mx, my, mz, natoms);
   //  if(step>85) printf ("\nReturned from movout\n");
  //    check_cells(rx, ry, rz, head, list, 
   clReleaseMemObject(d_rx);
   clReleaseMemObject(d_ry);
   clReleaseMemObject(d_rz);
   clReleaseMemObject(d_head);
   clReleaseMemObject(d_list);
   d_rx = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * natoms * 2, rx, &err);
   if(err < 0) {
     perror("Couldn't create an rx buffer");
     exit(1);   
   }   
   d_ry = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * natoms * 2, ry, &err);
   if(err < 0) {
     perror("Couldn't create an ry buffer");
     exit(1);   
   }   
   d_rz = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * natoms * 2, rz, &err);
   if(err < 0) {
     perror("Couldn't create an rz buffer");
     exit(1);   
   }    
   d_head = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, (mx+2)*(my+2)*(mz+2)*sizeof(int), head, &err);
   if(err < 0) {
     perror("Couldn't create an head buffer");
     exit(1);   
   }   
   d_list = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 2*natoms*sizeof(int), list, &err);
   if(err < 0) {
     perror("Couldn't create a list buffer");
     exit(1);   
   }      

   err = clSetKernelArg(force_kernel, 0, sizeof(cl_mem), &d_virialArray);
   err |= clSetKernelArg(force_kernel, 1, sizeof(cl_mem), &d_potentialArray);
   err |= clSetKernelArg(force_kernel, 2, sizeof(cl_mem), &d_rx);
   err |= clSetKernelArg(force_kernel, 3, sizeof(cl_mem), &d_ry);
   err |= clSetKernelArg(force_kernel, 4, sizeof(cl_mem), &d_rz);
   err |= clSetKernelArg(force_kernel, 5, sizeof(cl_mem), &d_fx);
   err |= clSetKernelArg(force_kernel, 6, sizeof(cl_mem), &d_fy);
   err |= clSetKernelArg(force_kernel, 7, sizeof(cl_mem), &d_fz);
   err |= clSetKernelArg(force_kernel, 8, sizeof(sigma), &sigma);
   err |= clSetKernelArg(force_kernel, 9, sizeof(rcut), &rcut);
   err |= clSetKernelArg(force_kernel, 10, sizeof(vrcut), &vrcut);
   err |= clSetKernelArg(force_kernel, 11, sizeof(dvrc12), &dvrc12);
   err |= clSetKernelArg(force_kernel, 12, sizeof(dvrcut), &dvrcut);
   err |= clSetKernelArg(force_kernel, 13, sizeof(cl_mem), &d_head);
   err |= clSetKernelArg(force_kernel, 14, sizeof(cl_mem), &d_list);
   err |= clSetKernelArg(force_kernel, 15, sizeof(mx), &mx);
   err |= clSetKernelArg(force_kernel, 16, sizeof(my), &my);
   err |= clSetKernelArg(force_kernel, 17, sizeof(mz), &mz);
   err |= clSetKernelArg(force_kernel, 18, sizeof(natoms), &natoms);
   err |= clSetKernelArg(force_kernel, 19, sizeof(sfx), &sfx);
   err |= clSetKernelArg(force_kernel, 20, sizeof(sfy), &sfy);
   err |= clSetKernelArg(force_kernel, 21, sizeof(sfz), &sfz);

   if(err < 0) {
     printf("Couldn't set an argument for the force kernel");
     exit(1);   
   }

      //printf("Global Size: %d\n", global_size[0]);
      //printf("Local Size: %d\n", local_size[0]);
      err = clEnqueueNDRangeKernel(queue, force_kernel, 1, NULL, global_size, local_size, 0, NULL, NULL);
      
      if(err < 0) {
        printf("Couldn't enqueue force kernel\n");
        exit(1);   
      }
      clFinish(queue);
      //float fxTest[natoms];
      //size_t sizy = sizeof(float) * (natoms);
      err = clEnqueueReadBuffer(queue, d_virialArray, CL_TRUE, 0, sizeof(float) * numBlocks, virialArrayTemp, 0, NULL, NULL);
      err |= clEnqueueReadBuffer(queue, d_potentialArray, CL_TRUE, 0, sizeof(float) * numBlocks, potentialArrayTemp, 0, NULL, NULL);
      if(err < 0) {
        printf("Couldn't read buffer for virial and potential array\n");
        printf("%d\n",err );
        printf("CL_INVALID_COMMAND_QUEUE: %d\n",CL_INVALID_COMMAND_QUEUE);
        printf("CL_INVALID_CONTEXT: %d\n", CL_INVALID_CONTEXT);
        printf("CL_INVALID_MEM_OBJECT: %d\n", CL_INVALID_MEM_OBJECT);
        printf("CL_INVALID_VALUE: %d\n",CL_INVALID_VALUE);
        printf("CL_INVALID_EVENT_WAIT_LIST: %d\n", CL_INVALID_EVENT_WAIT_LIST);
        printf("CL_MEM_OBJECT_ALLOCATION_FAILURE: %d\n",CL_MEM_OBJECT_ALLOCATION_FAILURE);
        printf("CL_OUT_OF_HOST_MEMORY: %d\n", CL_OUT_OF_HOST_MEMORY);
        exit(1);   
      }
      //numInc = 0;
      //globalThreads = ceil(numBlocks / (float)BLOCK_WIDTH);
     // startTime = get_tick();
      virial = 0.0;
      potential = 0.0;
      int tempInd = 0;
      for (tempInd =0; tempInd < numBlocks; tempInd++)
      {
        potential += potentialArrayTemp[tempInd];
        virial += virialArrayTemp[tempInd];
      }
      virial *= 48.0/3.0;
      potential *= 4.0;

//      if(step>85)printf ("\nReturned from force: potential = %f, virial = %f, kinetic = %f\n",potential, virial, kinetic);
 //     fflush(stdout);
      err = clSetKernelArg(movebKernel, 0, sizeof(cl_mem), &d_kineticArray);
      err |= clSetKernelArg(movebKernel, 1, sizeof(cl_mem), &d_vx);
      err |= clSetKernelArg(movebKernel, 2, sizeof(cl_mem), &d_vy);
      err |= clSetKernelArg(movebKernel, 3, sizeof(cl_mem), &d_vz);
      err |= clSetKernelArg(movebKernel, 4, sizeof(cl_mem), &d_fx);
      err |= clSetKernelArg(movebKernel, 5, sizeof(cl_mem), &d_fy);
      err |= clSetKernelArg(movebKernel, 6, sizeof(cl_mem), &d_fz);
      err |= clSetKernelArg(movebKernel, 7, sizeof(dt), &dt);
      err |= clSetKernelArg(movebKernel, 8, sizeof(natoms), &natoms);
      if(err < 0) {
       printf("Couldn't set an argument for the moveb kernel");
       exit(1);   
      }
      err = clEnqueueNDRangeKernel(queue, movebKernel, 1, NULL, global_size, local_size, 0, NULL, NULL);
      if(err < 0) {
       printf("Couldn't enqueue the moveb kernel");
       exit(1);   
      }
      clFinish(queue);

      err = clEnqueueReadBuffer(queue, d_kineticArray, CL_TRUE, 0, sizeof(float) * numBlocks, kineticArray, 0, NULL, NULL);
      if(err < 0) {
       printf("%d\n",err );
       printf("CL_INVALID_COMMAND_QUEUE: %d\n",CL_INVALID_COMMAND_QUEUE);
       printf("CL_INVALID_CONTEXT: %d\n", CL_INVALID_CONTEXT);
       printf("CL_INVALID_MEM_OBJECT: %d\n", CL_INVALID_MEM_OBJECT);
       printf("CL_INVALID_VALUE: %d\n",CL_INVALID_VALUE);
       printf("CL_INVALID_EVENT_WAIT_LIST: %d\n", CL_INVALID_EVENT_WAIT_LIST);
       printf("CL_MEM_OBJECT_ALLOCATION_FAILURE: %d\n",CL_MEM_OBJECT_ALLOCATION_FAILURE);
       printf("CL_OUT_OF_HOST_MEMORY: %d\n", CL_OUT_OF_HOST_MEMORY);
       exit(1);   
      }
      int stepInd = 0;
      for (stepInd = 1; stepInd < numBlocks; stepInd++)
      {
        kineticArray[0] += kineticArray[stepInd];
      }
      kinetic = kineticArray[0];
      kinetic *= 0.5;

 //     check_cells(rx, ry, rz, head, list, mx, my, mz, natoms,step,step);
  // if(step>85)   printf ("\nReturned from moveb: potential = %f, virial = %f, kinetic = %f\n",potential, virial, kinetic);
      sum_energies (potential, kinetic, virial, &vg, &wg, &kg);
      hloop (kinetic, step, vg, wg, kg, freex, dens, sigma, eqtemp, &tmpx, &ace, &acv, &ack, &acp, &acesq, &acvsq, &acksq, &acpsq, vx, vy, vz, iscale, iprint, nequil, natoms, &scaletKernel, &queue, &d_vx, &d_vy, &d_vz);

   }
   endTime = get_tick();
   elapsedTime += (endTime - startTime);
   tidyup (ace, ack, acv, acp, acesq, acksq, acvsq, acpsq, nstep, nequil);
   elapsedTime = elapsedTime / (float) 1000;
   printf("\n%Lf seconds have elapsed \n", elapsedTime);
   
   return 0;
}
