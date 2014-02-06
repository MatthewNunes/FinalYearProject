#define PROGRAM_FILE "shMatMulOpenCL.cl"
#define MAT_MUL_KERNEL "matMulKernel"
#define BLOCK_WIDTH 16
#define TILE_WIDTH 16
#define WIDTH 512

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <CL/cl.h>


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

/**
void matMulDevice(float *h_M, float *h_N, float *h_P, int Width)
{
	int size = Width * Width * sizeof(float); 
	float *d_M, *d_N, *d_P;
// Step 1: Allocate and Load M, N to device memory 
	cudaMalloc((void **)&d_M, size);
	cudaMemcpy(d_M, h_M, size, cudaMemcpyHostToDevice);
	cudaMalloc((void **)&d_N, size);
	cudaMemcpy(d_N, h_N, size, cudaMemcpyHostToDevice);
// Step 2: Allocate P on the device
	cudaMalloc((void **)&d_P, size);
// Step 3a: Set up execution configuration
   int numBlocks = ceil(Width/(float)BLOCK_WIDTH);
   dim3 dimGrid(numBlocks,numBlocks);
   dim3 dimBlock(BLOCK_WIDTH, BLOCK_WIDTH);
// Step 3b: Launch the device computation threads!
   matMulKernel<<<dimGrid, dimBlock>>>(d_M, d_N, d_P, Width);
// Step 4: Copy back result, and free memory on device
   cudaMemcpy(h_P, d_P, size, cudaMemcpyDeviceToHost);
   cudaFree(d_M); cudaFree(d_N); cudaFree(d_P);
}
*/

void matMulDevice(float *h_M, float *h_N, float *h_P, int width)
{
	   /* Host/device data structures */
   cl_device_id device;
   cl_context context;
   cl_command_queue queue;
   cl_program program;
   cl_kernel mult_kernel;
   size_t global_size[2];
   size_t local_size[2];
   cl_ulong mem_size;
   cl_int i, j, k, err, check;
   int twidth = TILE_WIDTH;

   /* Data and buffers */
   cl_uint matrix_dim;
 //  float *h_M[width][width];
 //  float *h_N[width][width];
 //  float *h_P[width][width];
   cl_mem m_buffer, n_buffer, p_buffer;
   void* mapped_memory;
   int doubleWidth = width * width;

   /* Create a device and context */
   device = create_device();
   context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
   if(err < 0) {
	  perror("Couldn't create a context");
	  exit(1);   
   }

   /* Build the program */
   program = build_program(context, device, PROGRAM_FILE);


   /* Create a kernel for the multiplication function */
   mult_kernel = clCreateKernel(program, MAT_MUL_KERNEL, &err);
   if(err < 0) {
	  perror("Couldn't create a kernel");
	  exit(1);
   }

   /* Create buffers */
   m_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * width * width, h_M, &err);
   if(err < 0) {
	  perror("Couldn't create a buffer");
	  exit(1);   
   }
   n_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * width * width, h_N, &err);
   p_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * width * width, h_P, &err);

   /* Create a command queue */
   queue = clCreateCommandQueue(context, device, 0, &err);
   if(err < 0) {
	  perror("Couldn't create a command queue");
	  exit(1);   
   }

   /* Create arguments for multiplication kernel */
   err = clSetKernelArg(mult_kernel, 0, sizeof(cl_mem), &m_buffer);
   err |= clSetKernelArg(mult_kernel, 1, sizeof(cl_mem), &n_buffer);
   err |= clSetKernelArg(mult_kernel, 2, TILE_WIDTH * TILE_WIDTH, NULL);
   err |= clSetKernelArg(mult_kernel, 3, TILE_WIDTH * TILE_WIDTH, NULL);
   err |= clSetKernelArg(mult_kernel, 4, sizeof(cl_mem), &p_buffer);
   err |= clSetKernelArg(mult_kernel, 5, sizeof(uint), &width);
   err |= clSetKernelArg(mult_kernel, 6, sizeof(uint), &twidth);
   if(err < 0) {
	  printf("Couldn't set an argument for the transpose kernel");
	  exit(1);   
   }
   

   
	/* Enqueue command to map buffer to host memory */
   /**
   mapped_memory = clEnqueueMapBuffer(queue, p_buffer, CL_TRUE, CL_MAP_READ, 0, sizeof(float) * width * width, 0, NULL, NULL, &err);
   if(err < 0) {
	  perror("Couldn't map the buffer to host memory");
	  exit(1);   
   }
   */
   
   /* Enqueue multiplication kernel */
 //  global_size[0] = ceil(width/BLOCK_WIDTH);
//   global_size[1] = ceil(width/BLOCK_WIDTH);
   global_size[0] =width;
   global_size[1] = width;
   local_size[0] = BLOCK_WIDTH;
   local_size[1] = BLOCK_WIDTH;
   
   size_t global_test = doubleWidth / 8;
   size_t local_test = 8;
   size_t max;
   clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(max), &max, NULL);
   printf("Max: %d\n", max);
   err = clEnqueueNDRangeKernel(queue, mult_kernel, 2, NULL, global_size, local_size, 0, NULL, NULL);
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
	  perror("Couldn't enqueue the multiplication kernel");
	  exit(1);    
   } 
   clFinish(queue);
   /* Read output buffer */
   //memcpy(h_P, mapped_memory, sizeof(float) * width * width); 
	   err = clEnqueueReadBuffer(queue, p_buffer, CL_TRUE, 0, sizeof(float)*width*width, h_P, 0, NULL, NULL);
	printf("----------------------\n");
	printf("CL_INVALID_COMMAND_QUEUE: %d\n", CL_INVALID_COMMAND_QUEUE);
	printf("CL_INVALID_CONTEXT: %d\n", CL_INVALID_CONTEXT);
	printf("CL_INVALID_MEM_OBJECT: %d\n", CL_INVALID_MEM_OBJECT);
	printf("CL_INVALID_VALUE: %d\n", CL_INVALID_VALUE);
	printf("CL_INVALID_EVENT_WAIT_LIST: %d\n", CL_INVALID_EVENT_WAIT_LIST);
	printf("CL_MEM_OBJECT_ALLOCATION_FAILURE: %d\n", CL_MEM_OBJECT_ALLOCATION_FAILURE);
	printf("CL_OUT_OF_HOST_MEMORY: %d\n", CL_OUT_OF_HOST_MEMORY);
   if(err < 0) {
	  printf("Error: %d\n", err);
	  perror("Couldn't read the buffer");
	  exit(1);   
   } 
   /* Unmap memory */ 
   int ind = 0;
   int in =0;
   for(ind = 0; (ind+ 8) < (width*width); ind+=8)
   {
	  in +=1;
	//printf("hi");
	//printf("%f\n", h_P[ind]);
	 printf("%d: %.0f, %.0f, %.0f, %.0f, %.0f, %.0f, %.0f, %.0f\n",in, h_P[ind], h_P[ind + 1], h_P[ind + 2], h_P[ind + 3], h_P[ind + 4], h_P[ind + 5], h_P[ind + 6], h_P[ind + 7]);
   }
   in +=1;
   printf("%d: %.0f, %.0f, %.0f, %.0f, %.0f, %.0f, %.0f, %.0f\n",in, *(h_P+ind), *(h_P+ind+1), *(h_P+ind+2), *(h_P+ind+3), *(h_P+ind+4), *(h_P+ind+5), *(h_P+ind+6), *(h_P+ind+7));
   //err = clEnqueueUnmapMemObject(queue, p_buffer, mapped_memory, 0, NULL, NULL);
   if(err < 0) {
	  perror("Couldn't unmap the buffer");
	  exit(1);   
   }

   /* Deallocate resources */
   clReleaseMemObject(m_buffer);
   clReleaseMemObject(n_buffer);
   clReleaseKernel(mult_kernel);
   clReleaseCommandQueue(queue);
   clReleaseProgram(program);
   clReleaseContext(context);
  // return 0;
}

int main()
{
	float *h_M, *h_N, *h_P;
	int i, n = WIDTH, size=sizeof(float)*n*n;
	h_P = (float *)malloc(size);
	h_M = (float *)malloc(size);
	h_N = (float *)malloc(size);
   int c = 2;
	for(i=0;i<n*n;i++)
	{
		*(h_M+i)=(float)i; 
		*(h_N+i)=(float)i;
	}
	matMulDevice(h_M,h_N,h_P,n);
	return 0;
}