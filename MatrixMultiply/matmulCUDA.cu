#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/time.h>
#include <cuda.h>
#include <time.h>
#include "matthew_CUDA.h"

#define WIDTH 3000
#define BLOCK_WIDTH 16

__global__ 
void matMulKernel(float *d_M, float *d_N, float *d_P)
{ 
    int Row = blockIdx.y*blockDim.y + threadIdx.y;
    int Col = blockIdx.x*blockDim.x + threadIdx.x;
    int k;
    if ((Row<WIDTH)&&(Col<WIDTH)){
        float Pvalue = 0.0;
        for(k=0;k<WIDTH;k++)
        {
            Pvalue += d_M[Row*WIDTH+k]*d_N[k*WIDTH+Col];
        }
        d_P[Row*WIDTH+Col] = Pvalue;
    }
}

long unsigned int get_tick()
{
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) return (0);
    return ts.tv_sec*(long int)1000 + ts.tv_nsec / (long int) 1000000;
}

void matMulDevice(float *h_M, float *h_N, float *h_P, int Width, float *d_check)
{
   // struct timeval tim;
    int size = Width * Width * sizeof(float); 
    float *d_M, *d_N, *d_P;
// Step 1: Allocate and Load M, N to device memory 
    CUDA_CHECK_RETURN(cudaMalloc((void **)&d_M, size));
    CUDA_CHECK_RETURN(cudaMemcpy(d_M, h_M, size, cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&d_N, size));
    CUDA_CHECK_RETURN(cudaMemcpy(d_N, h_N, size, cudaMemcpyHostToDevice));
// Step 2: Allocate P on the device
    CUDA_CHECK_RETURN(cudaMalloc((void **)&d_P, size));
// Step 3a: Set up execution configuration
   int numBlocks = ceil(Width/(float)BLOCK_WIDTH);
   dim3 dimGrid(numBlocks,numBlocks);
   dim3 dimBlock(BLOCK_WIDTH, BLOCK_WIDTH);
// Step 3b: Launch the device computation threads!
  //  cudaEvent_t start, stop;
  //  cudaEventCreate(&start);
  //  cudaEventCreate(&stop);
     
   //  cudaEventRecord(start, 0);
   //  gettimeofday(&tim, NULL);
  //   double t1=tim.tv_sec+(tim.tv_usec/1000000.0);
   long int start = get_tick();
   matMulKernel<<<dimGrid, dimBlock>>>(d_M, d_N, d_P);
   cudaDeviceSynchronize();
   long int end = get_tick();
  // gettimeofday(&tim, NULL);
  // double t2=tim.tv_sec+(tim.tv_usec/1000000.0);
 //  cudaEventRecord(stop, 0);
 //  cudaEventSynchronize(stop);
   //float elapsedTime;
   //cudaEventElapsedTime(&elapsedTime, start, stop);
   //elapsedTime = elapsedTime / (float) 1000000;
   long double elapsed_time = (end - start)/ (float)1000;
   printf("%Lf seconds elapsed\n", elapsed_time);
// Step 4: Copy back result, and free memory on device
   CUDA_CHECK_RETURN(cudaMemcpy(h_P, d_P, size, cudaMemcpyDeviceToHost));
   cudaFree(d_M); cudaFree(d_N); cudaFree(d_P);
   //printf("%.6f seconds elapsed\n", elapsedTime);
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


int main()
{
    float *h_M, *h_N, *h_P, *check;
    int i, n = WIDTH, size=sizeof(float)*n*n;
    h_P = (float *)malloc(size);
    h_M = (float *)malloc(size);
    h_N = (float *)malloc(size);
    check = (float *)malloc(size);
    for(i=0;i<n*n;i++)
    {
        *(h_M+i)=(float)i; 
        *(h_N+i)=(float)i;
    }
    matMulDevice(h_M,h_N,h_P,n, check);
    /**
    float *h_PH = (float *)malloc(size);
    matMul(h_M,h_N,h_PH,n);
    int ok = checkP(h_P,h_PH,n);
    if(ok) printf("Everything worked!\n");
    else printf("Something went wrong!\n");
    */
}
