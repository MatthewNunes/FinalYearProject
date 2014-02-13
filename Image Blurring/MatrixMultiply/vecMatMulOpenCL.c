#define PROGRAM_FILE "vecMatMulOpenCL.cl"
#define MAT_MUL_KERNEL "matrix_mult"
#define BLOCK_WIDTH 16
#define WIDTH 512

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <sys/types.h>
#include <sys/time.h>
#include <CL/cl.h>

/* Find a GPU or CPU associated with the first available platform */
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

void matMulDevice(float *h_M, float *h_N, float *h_P, int width)
{
	   /* Host/device data structures */
   struct timeval tim;
   cl_device_id device;
   cl_context context;
   cl_command_queue queue;
   cl_program program;
   cl_kernel mult_kernel;
   cl_event timing_event;
   cl_ulong time_start, time_end, total_time;
   size_t global_size;
   size_t local_size;
   cl_ulong mem_size;
   cl_int i, j, k, err, check;

   /* Data and buffers */
   cl_uint matrix_dim;
 //  float *h_M[width][width];
 //  float *h_N[width][width];
 //  float *h_P[width][width];
   cl_mem m_buffer, n_buffer, p_buffer;
   void* mapped_memory;
   int doubleWidth = width * width;
   float *h_NT = (float *)malloc(sizeof(float) * width * width);
   int i1 = 0;
   int j1= 0;
   for(i1 =0; i1 < width; i1++)
   {
      for (j1 = 0; j1 < width; j1++)
      {
         h_NT[i1 * width + j1] = h_N[width * j1 + i1];
      }
   }

   /* Create a device and context */
   device = create_device();
    cl_uint integerVectorWidth;
    clGetDeviceInfo(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT, sizeof(cl_uint), &integerVectorWidth, NULL);
    
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
   n_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * width * width, h_NT, &err);
   p_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * width * width, h_P, &err);

   /* Create a command queue */
   queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
   if(err < 0) {
	  perror("Couldn't create a command queue");
	  exit(1);   
   }

   /* Create arguments for multiplication kernel */
   err = clSetKernelArg(mult_kernel, 0, sizeof(cl_mem), &m_buffer);
   err |= clSetKernelArg(mult_kernel, 1, sizeof(cl_mem), &n_buffer);
   err |= clSetKernelArg(mult_kernel, 2, sizeof(cl_mem), &p_buffer);
  // err |= clSetKernelArg(mult_kernel, 3, sizeof(uint), &width);
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
   global_size = width;
   //global_size[1] = width/4;
   //local_size = BLOCK_WIDTH;
   //local_size[1] = BLOCK_WIDTH/4;
   
   size_t global_test = doubleWidth / 8;
   size_t local_test = 8;
   size_t max;
   clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(max), &max, NULL);
   printf("Max: %d\n", max);
   err = clEnqueueNDRangeKernel(queue, mult_kernel, 1, NULL, &global_size, NULL, 0, NULL, &timing_event);
   if(err < 0) {
	  printf("Err: %d\n", err);
	  perror("Couldn't enqueue the multiplication kernel");
	  exit(1);    
   } 
   clFinish(queue);
   /* Read output buffer */
   //memcpy(h_P, mapped_memory, sizeof(float) * width * width); 
   clGetEventProfilingInfo(timing_event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
   clGetEventProfilingInfo(timing_event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
   total_time = time_end - time_start;
   double tt;
   tt = total_time / (double) 1000000000;
	err = clEnqueueReadBuffer(queue, p_buffer, CL_TRUE, 0, sizeof(float)*width*width, h_P, 0, NULL, NULL);
   if(err < 0) {
	  printf("Error: %d\n", err);
	  perror("Couldn't read the buffer");
	  exit(1);   
   } 
   /* Unmap memory */ 

   printf("\n%.6lf seconds elapsed\n", tt);

   if(err < 0) {
	  perror("Couldn't unmap the buffer");
	  exit(1);   
   }

   /* Deallocate resources */
   clReleaseMemObject(m_buffer);
   clReleaseMemObject(n_buffer);
   clReleaseMemObject(p_buffer);
   clReleaseKernel(mult_kernel);
   clReleaseCommandQueue(queue);
   clReleaseProgram(program);
   clReleaseContext(context);
  // return 0;
}

int checkP(float *h_P, float *h_PH, int n)
{
    int i;
    int ok = 1;
    for(i=0;i<n*n;i++){
       float diff = fabsf((*(h_P+i))-(*(h_PH+i)))/(*(h_PH+i));
       ok &= (diff<0.00001);
       if(diff>=.00001) printf("%d: %f, %f\n",i,*(h_P+i),*(h_PH+i));
    }
    return (ok);
}

void matMul(float* M, float* N, float* P, int Width) 
{
    int i, j, k;
    for (j = 0; j < Width; ++j)
        for (i = 0; i < Width; ++i) {
            float sum = 0;
            for (k = 0; k < Width; ++k) {
                float a = M[j * Width + k];
                float b = N[k * Width + i];
                sum += a * b;
            }
            P[j * Width + i] = sum;
        }
}

int main() {
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
   float *h_PH = (float *)malloc(size);
   matMul(h_M,h_N,h_PH,n);
   int ok = checkP(h_P,h_PH,n);
   if(ok) printf("Everything worked!\n");
   else printf("Something went wrong!\n");
   return 0;
}
