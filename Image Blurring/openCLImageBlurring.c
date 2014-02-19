#include "lodepng.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <CL/cl.h>
#include <time.h>
#define NSTEPS 500
#define TX 16
#define TY 16
#define NPTSX 200
#define NPTSY 200
#define UPDATE_KERNEL "performUpdatesKernel"
#define COPY_KERNEL "doCopyKernel"
#define PROGRAM_FILE "openCLImageBlurring.cl"

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

void blurImage(const char * filename)
{
	cl_device_id device;
	cl_context context;
	cl_command_queue queue;
	cl_program program;
	cl_kernel updates_kernel;
	cl_kernel copy_kernel;
	//cl_event timing_event;
	//cl_ulong time_start, time_end, total_time;
	size_t global_size[2];
	size_t local_size[2];
	cl_ulong mem_size;
	//cl_int i, j, k, err, check;
	cl_int err;
	
	   
	//cl_mem phi_buffer, oldphi_buffer, dmask_buffer;
	
  	unsigned error;
	unsigned char* image;
	unsigned width, height;
	
	error = lodepng_decode24_file(&image, &width, &height, filename);
	if(error) printf("error %u: %s\n", error, lodepng_error_text(error));
	
  	int rowsize = height;
  	int colsize = width;
	
  	int *R= (int *)malloc(sizeof(int) * height * width);
  	int *G= (int *)malloc(sizeof(int) * height * width);
  	int *B= (int *)malloc(sizeof(int) * height * width);
	
  	int *h_Rnew= (int *)malloc(sizeof(int) * height * width);
	int *h_Gnew= (int *)malloc(sizeof(int) * height * width);
	int *h_Bnew= (int *)malloc(sizeof(int) * height * width);	
//  int it = 0;
	
  	unsigned char *newImage = (unsigned char *)malloc(sizeof(unsigned char) * width * height * 3);
	
  	int row = 0; 
  	int col = 0;
  	//int i =0, j=0;
  	int nblurs;
	
  	char *first = (char *)malloc(sizeof(char));
  	char *second = (char *)malloc(sizeof(char));
  	char *third = (char *)malloc(sizeof(char));
  	char *firstPointer = first;
  	char *secondPointer = second;
  	char *thirdPointer = third;
	int n, m;
  	for(n=0,m=0;n<3*width*height;n+=3,m++)
  	{
		sprintf(first, "%02x", *(image + n));
		sprintf(second, "%02x", *(image + n+1));
		sprintf(third, "%02x", *(image + n+2));
		*(R+m) = strtol(first, NULL, 16);
		*(G+m) = strtol(second, NULL, 16);
		*(B+m) = strtol(third, NULL, 16);
		first = firstPointer;
		second = secondPointer;
		third = thirdPointer;
    }
	int size = rowsize * colsize; 
  	cl_mem d_R, d_G, d_B, d_Rnew, d_Gnew, d_Bnew;
  	//h_Rnew = (int *)malloc(sizeof(int) * size);
  	//h_Gnew = (int *)malloc(sizeof(int) * size);
  	//h_Bnew = (int *)malloc(sizeof(int) * size);  	
	device = create_device();
	context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
	if(err < 0) {
		perror("Couldn't create a context");
		exit(1);   
	}

 	nblurs = 10;
 	printf("\nGive the number of times to blur the image\n");
  	int icheck = scanf ("%d", &nblurs);

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
	d_R = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(int)*size, R, &err);
	d_G = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(int)*size, G, &err);
	d_B = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(int)*size, B, &err);
	d_Rnew = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(int)*size, h_Rnew, &err);
	d_Gnew = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(int)*size, h_Gnew, &err);
	d_Bnew = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(int)*size, h_Bnew, &err);
	
	if(err < 0) {
		perror("Couldn't create a buffer");
		exit(1);   
	}
   
	queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
	if(err < 0) {
		perror("Couldn't create a command queue");
		exit(1);   
	}
	
	err = clSetKernelArg(updates_kernel, 0, sizeof(cl_mem), &d_R);
	err |= clSetKernelArg(updates_kernel, 1, sizeof(cl_mem), &d_G);
	err |= clSetKernelArg(updates_kernel, 2, sizeof(cl_mem), &d_B);
	err |= clSetKernelArg(updates_kernel, 3, sizeof(cl_mem), &d_Rnew);
	err |= clSetKernelArg(updates_kernel, 4, sizeof(cl_mem), &d_Gnew);
	err |= clSetKernelArg(updates_kernel, 5, sizeof(cl_mem), &d_Bnew);
	err |= clSetKernelArg(updates_kernel, 6, sizeof(int), &rowsize);
	err |= clSetKernelArg(updates_kernel, 7, sizeof(int), &colsize);
	if(err < 0) {
		printf("Couldn't set an argument for the transpose kernel");
		exit(1);   
	}
	
	err = clSetKernelArg(copy_kernel, 0, sizeof(cl_mem), &d_R);
	err |= clSetKernelArg(copy_kernel, 1, sizeof(cl_mem), &d_G);
	err |= clSetKernelArg(copy_kernel, 2, sizeof(cl_mem), &d_B);
	err |= clSetKernelArg(copy_kernel, 3, sizeof(cl_mem), &d_Rnew);
	err |= clSetKernelArg(copy_kernel, 4, sizeof(cl_mem), &d_Gnew);
	err |= clSetKernelArg(copy_kernel, 5, sizeof(cl_mem), &d_Bnew);
	err |= clSetKernelArg(copy_kernel, 6, sizeof(int), &rowsize);
	err |= clSetKernelArg(copy_kernel, 7, sizeof(int), &colsize);
	if(err < 0) {
		printf("Couldn't set an argument for the transpose kernel");
		exit(1);   
	}
	int k;
	global_size[0] = ceil(colsize/(float)TX) * TX;
	global_size[1] = ceil(rowsize/(float)TY) * TY;
	local_size[0] = TX;
	local_size[1] = TY;
	long int start = get_tick();
	for(k=0;k<nblurs;k++)
	{
		err = clEnqueueNDRangeKernel(queue, updates_kernel, 2, NULL, global_size, local_size, 0, NULL, NULL); 		
		err = clEnqueueNDRangeKernel(queue, copy_kernel, 2, NULL, global_size, local_size, 0, NULL, NULL);
	} 
	clFinish(queue);
	long int end = get_tick();
	long double elapsedTime = (end - start)/ (float) 1000;
	printf("%Lf seconds elapsed\n", elapsedTime);
	err = clEnqueueReadBuffer(queue, d_Rnew, CL_TRUE, 0, sizeof(int)*size, h_Rnew, 0, NULL, NULL);
	err |= clEnqueueReadBuffer(queue, d_Gnew, CL_TRUE, 0, sizeof(int)*size, h_Gnew, 0, NULL, NULL);
	err |= clEnqueueReadBuffer(queue, d_Bnew, CL_TRUE, 0, sizeof(int)*size, h_Bnew, 0, NULL, NULL);
	if(err < 0) 
	{
		printf("Error: %d\n", err);
		perror("Couldn't read the buffer");
		exit(1);   
	} 

 	int kk = 0;
  	int mm = 0;
  	for (kk = 0; kk < rowsize; kk++)
  	{
		for (mm=0; mm<colsize; mm++)
		{
	  		R[kk*colsize+mm] = h_Rnew[colsize*kk +mm];
	  		G[kk*colsize + mm] = h_Gnew[colsize*kk +mm];
	  		B[kk*colsize +mm] = h_Bnew[colsize*kk +mm];  
		}
  	}
	
	unsigned char *newImage2 = newImage;
	char *convertMe = (char *)malloc(sizeof(char) * 1);
	char *convertMePointer = convertMe;
	for(row=0;row<rowsize;row++)
	{
		for (col=0;col<colsize;col++)
		{
			sprintf(convertMe  ,"%02x",*(R+ (row * colsize) + col));
	  		*newImage2 = strtol(convertMe, NULL, 16);
	  		convertMe = convertMePointer;
	  		newImage2+=1;
	  		sprintf(convertMe, "%02x", *(G + (row * colsize) + col));
	  		*newImage2 = strtol(convertMe, NULL, 16);
	  		convertMe = convertMePointer;
	  		newImage2+=1;
	  		sprintf(convertMe, "%02x", *(B + (row * colsize) + col));
	  		*newImage2 = strtol(convertMe, NULL, 16);
	  		newImage2+=1;
	  	}
 	}

  	unsigned error2 = lodepng_encode24_file("file.png",newImage, width, height);
  	free(image);
  	free(newImage);
  	free(R);
  	free(G);
  	free(B);
	
}
   


int main (int argc, char *argv[])
{
  blurImage("Laplace.png");
  
  return 0;
}
