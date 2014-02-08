#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <CL/cl.h>
#define NSTEPS 500
#define TX 16
#define TY 32
#define NPTSX 200
#define NPTSY 200
#define UPDATE_KERNEL "performUpdatesKernel"
#define COPY_KERNEL "doCopyKernel"
#define PROGRAM_FILE "laplaceOpenCL.cl"


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

void performUpdates(float *h_phi, float * h_oldphi, int *h_mask, int nptsx, int nptsy, int nsteps)
{
   	cl_device_id device;
   	cl_context context;
   	cl_command_queue queue;
   	cl_program program;
   	cl_kernel updates_kernel;
	cl_kernel copy_kernel;
   	cl_event timing_event;
   	cl_ulong time_start, time_end, total_time;
   	size_t global_size[2];
   	size_t local_size[2];
   	cl_ulong mem_size;
   	cl_int i, j, k, err, check;

   
   	cl_mem phi_buffer, oldphi_buffer, dmask_buffer;

	device = create_device();
   	context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
   	if(err < 0) {
		perror("Couldn't create a context");
	  	exit(1);   
   	}

   	/* Build the program */
   	program = build_program(context, device, PROGRAM_FILE);


   	/* Create a kernel for the multiplication function */
   	updates_kernel = clCreateKernel(program, UPDATE_KERNEL, &err);
   	if(err < 0) {
	  	perror("Couldn't create a kernel");
	  	exit(1);
   	}
   	
	copy_kernel = clCreateKernel(program, COPY_KERNEL, &err);
	if(err < 0) {
		perror("Couldn't create a kernel");
		exit(1);
	}


   	/* Create buffers */
   	phi_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * nptsx * nptsy, h_phi, &err);
	oldphi_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * nptsx * nptsy, h_oldphi, &err);
   	dmask_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(int) * nptsx * nptsy, h_mask, &err);
	if(err < 0) {
	  	perror("Couldn't create a buffer");
	  	exit(1);   
   	}
   
   	queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
   	if(err < 0) {
		perror("Couldn't create a command queue");
		exit(1);   
   	}
	
	err = clSetKernelArg(updates_kernel, 0, sizeof(cl_mem), &phi_buffer);
   	err |= clSetKernelArg(updates_kernel, 1, sizeof(cl_mem), &oldphi_buffer);
   	err |= clSetKernelArg(updates_kernel, 2, sizeof(cl_mem), &dmask_buffer);
   	err |= clSetKernelArg(updates_kernel, 3, sizeof(uint), &nptsx);
	err |= clSetKernelArg(updates_kernel, 4, sizeof(uint), &nptsy);
   	if(err < 0) {
		printf("Couldn't set an argument for the transpose kernel");
		exit(1);   
   	}
	
	err = clSetKernelArg(copy_kernel, 0, sizeof(cl_mem), &phi_buffer);
	err |= clSetKernelArg(copy_kernel, 1, sizeof(cl_mem), &oldphi_buffer);
	err |= clSetKernelArg(copy_kernel, 2, sizeof(cl_mem), &dmask_buffer);
	err |= clSetKernelArg(copy_kernel, 3, sizeof(uint), &nptsx);
	err |= clSetKernelArg(copy_kernel, 4, sizeof(uint), &nptsy);
	if(err < 0) {
		printf("Couldn't set an argument for the transpose kernel");
		exit(1);   
	}
	//int k;
	global_size[0] = ceil(nptsx/TX) * TX;
	global_size[1] = ceil(nptsy/TY) * TY;
	local_size[0] = TX;
	local_size[1] = TY;
	for(k=0;k<nsteps;++k){
		err = clEnqueueNDRangeKernel(queue, updates_kernel, 2, NULL, global_size, local_size, 0, NULL, &timing_event);
		printf("CL_INVALID_PROGRAM_EXECUTABLE: %d\n", CL_INVALID_PROGRAM_EXECUTABLE);
		printf("CL_INVALID_COMMAND_QUEUE: %d\n", CL_INVALID_COMMAND_QUEUE);
		printf("CL_INVALID_KERNEL: %d\n", CL_INVALID_KERNEL);
		printf("CL_INVALID_CONTEXT: %d\n", CL_INVALID_CONTEXT);
		printf("CL_INVALID_KERNEL_ARGS: %d\n", CL_INVALID_KERNEL_ARGS);
		printf("CL_INVALID_WORK_DIMENSION: %d\n", CL_INVALID_WORK_DIMENSION);
		printf("CL_INVALID_WORK_GROUP_SIZE: %d\n", CL_INVALID_WORK_GROUP_SIZE);
		printf("CL_INVALID_WORK_ITEM_SIZE: %d\n", CL_INVALID_WORK_ITEM_SIZE);
		printf("CL_INVALID_GLOBAL_OFFSET: %d\n", CL_INVALID_GLOBAL_OFFSET);
		printf("CL_OUT_OF_RESOURCES: %d\n", CL_OUT_OF_RESOURCES);
		printf("CL_MEM_OBJECT_ALLOCATION_FAILURE: %d\n", CL_MEM_OBJECT_ALLOCATION_FAILURE);
		printf("CL_INVALID_EVENT_WAIT_LIST: %d\n", CL_INVALID_EVENT_WAIT_LIST);
		printf("CL_OUT_OF_HOST_MEMORY: %d\n", CL_OUT_OF_HOST_MEMORY);   		
		if(err < 0) {
	  		printf("Err: %d\n", err);
	  		perror("Couldn't enqueue the updates kernel");
	  		exit(1);    
   		} 
   		clFinish(queue);
		
		err = clEnqueueNDRangeKernel(queue, copy_kernel, 2, NULL, global_size, local_size, 0, NULL, &timing_event);
		if(err < 0) {
			printf("Err: %d\n", err);
			perror("Couldn't enqueue the copy kernel");
			exit(1);    
		}
		clFinish(queue);
	} 
	err = clEnqueueReadBuffer(queue, oldphi_buffer, CL_TRUE, 0, sizeof(float)*nptsx*nptsy, h_phi, 0, NULL, NULL);
   	if(err < 0) 
	{
		printf("Error: %d\n", err);
		perror("Couldn't read the buffer");
	 	exit(1);   
	} 
	
}
   
int RGBval(float x){
	int R, B, G, pow8 = 256;
	if(x<=0.5){
		B = (int)((1.0-2.0*x)*255.0);
		G = (int)(2.0*x*255.0);
	R = 0; 
	}
	else{
		B = 0;
		G = (int)((2.0-2.0*x)*255.0);
		R = (int)((2.0*x-1.0)*255.0);
	}
	return (B+(G+R*pow8)*pow8);
}

int setup_grid (float  *h_phi, int nptsx, int nptsy, int  *h_mask)
{
	int i, j, nx2, ny2;

	for(j=0;j<nptsy;j++)
	   for(i=0;i<nptsx;i++){
		  h_phi[j*nptsx+i]  = 0.0;
		  h_mask[j*nptsx+i] = 1;
	   }

	for(i=0;i<nptsx;i++) h_mask[i] = 0;

	for(i=0;i<nptsx;i++) h_mask[(nptsy-1)*nptsx+i] = 0;

	for(j=0;j<nptsy;j++) h_mask[j*nptsx] = 0;

	for(j=0;j<nptsy;j++) h_mask[j*nptsx+nptsx-1] = 0;

	nx2 = nptsx/2;
	ny2 = nptsy/2;
	h_mask[ny2*nptsx+nx2] = 0;
	h_mask[ny2*nptsx+nx2-1] = 0;
	h_mask[(ny2-1)*nptsx+nx2] = 0;
	h_mask[(ny2-1)*nptsx+nx2-1] = 0;
	h_phi[ny2*nptsx+nx2]  = 1.0;
	h_phi[ny2*nptsx+nx2-1]  = 1.0;
	h_phi[(ny2-1)*nptsx+nx2]  = 1.0;
	h_phi[(ny2-1)*nptsx+nx2-1]  = 1.0;
	return 0;
}

int output_array (float *h_phi, int nptsx, int nptsy)
{
   int i, j, k=0;
   FILE *fp;

   
   fp = fopen("outCUDA.ps","w");
   fprintf(fp,"/picstr %d string def\n",nptsx);
   fprintf(fp,"50 50 translate\n");
   fprintf(fp,"%d %d scale\n",nptsx, nptsy);
   fprintf(fp,"%d %d 8 [%d 0 0 %d 0 %d] \n",nptsx, nptsy, nptsx, nptsy, -nptsx);
   fprintf(fp,"{currentfile 3 200 mul string readhexstring pop} bind false 3 colorimage\n");

   for(j=0;j<nptsy;j++){
		for(i=0;i<nptsx;i++,k++){
			 fprintf(fp,"%06x",RGBval(h_phi[j*nptsx+i]));
			 if((k+1)%10==0) fprintf(fp,"\n");
		}
   }
   fclose(fp);
   return 0;
}

int main (int argc, char *argv[])
{
   float *h_phi;
   float *h_oldphi;
   int *h_mask;
   int nsize1=sizeof(float)*NPTSX*NPTSY;
   int nsize2=sizeof(int)*NPTSX*NPTSY;

   h_phi = (float *)malloc(nsize1);
   h_oldphi = (float *)malloc(nsize1);
   h_mask = (int *)malloc(nsize2);
   setup_grid (h_oldphi, NPTSX, NPTSY, h_mask);
   performUpdates(h_phi,h_oldphi,h_mask,NPTSX,NPTSY,NSTEPS);
 
   output_array (h_phi, NPTSX, NPTSY);
 
   return 0;
}
