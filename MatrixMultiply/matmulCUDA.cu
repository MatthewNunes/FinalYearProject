#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/time.h>
#include <cuda.h>
#define WIDTH 512
#define BLOCK_WIDTH 16

__global__ 
void matMulKernel(float *d_M, float *d_N, float *d_P, int Width)
{ 
    int Row = blockIdx.y*blockDim.y + threadIdx.y;
    int Col = blockIdx.x*blockDim.x + threadIdx.x;
    int k;
    if ((Row<Width)&&(Col<Width)){
        float Pvalue = 0.0;
        for(k=0;k<Width;k++)
        {
            Pvalue += d_M[Row*Width+k]*d_N[k*Width+Col];
        }
        d_P[Row*Width+Col] = Pvalue;
    }
}

void matMulDevice(float *h_M, float *h_N, float *h_P, int Width, float *d_check)
{
    struct timeval tim;
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
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
     
     cudaEventRecord(start, 0);
     gettimeofday(&tim, NULL);
     double t1=tim.tv_sec+(tim.tv_usec/1000000.0);
   matMulKernel<<<dimGrid, dimBlock>>>(d_M, d_N, d_P, Width);
   gettimeofday(&tim, NULL);
   double t2=tim.tv_sec+(tim.tv_usec/1000000.0);
   cudaEventRecord(stop, 0);
   cudaEventSynchronize(stop);
   float elapsedTime;
   cudaEventElapsedTime(&elapsedTime, start, stop);
   elapsedTime = elapsedTime / (float) 1000000;
    //printf("%.6f seconds elapsed\n", t2-t1);
// Step 4: Copy back result, and free memory on device
   cudaMemcpy(h_P, d_P, size, cudaMemcpyDeviceToHost);
   cudaFree(d_M); cudaFree(d_N); cudaFree(d_P);
   printf("%.6f seconds elapsed\n", elapsedTime);
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
    int j;
    int k;
    float *h_MTemp = h_M;
    float *h_NTemp = h_N;
    float sum = 0.0;
    for (i = 0; i < n; i++)
    {
         
        for (j = 0; j < n; j++)
        {
            sum = 0.0;
            for (k =0; k < n; k++)
            {
                sum += h_MTemp[i * n + k] * h_NTemp[k*n + j];
            }
            check[i * n + j] = sum;
        }
        
    }
    /**
    for (i = 0; i < n; i++)
    {
        for (j =0; j + 8 < n; j+=8)
        {
            printf("%f, %f, %f, %f, %f, %f, %f, %f\n", check[i*n + j], check[i*n + j + 1], check[i*n + j + 2], check[i*n + j + 3], check[i*n + j + 4], check[i*n + j + 5], check[i*n + j + 6], check[i*n + j + 7]); 
        }
        printf("%f, %f, %f, %f, %f, %f, %f, %f\n", check[i*n + j], check[i*n + j + 1], check[i*n + j + 2], check[i*n + j + 3], check[i*n + j + 4], check[i*n + j + 5], check[i*n + j + 6], check[i*n + j + 7]);
    }
    */
    matMulDevice(h_M,h_N,h_P,n, check);
}
