#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#define WIDTH 3000
#define BLOCK_WIDTH 16
#define TILE_WIDTH 16
#include <sys/types.h>
#include <sys/time.h>
#include <time.h>

__global__ 
void matMulTiledKernel(float *d_M, float *d_N, float *d_P, int Width)
{ 
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];
    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;
    int Row = by*TILE_WIDTH + ty;
    int Col = bx*TILE_WIDTH + tx;
    int k, m;
    float Pvalue = 0.0;
    for(m=0;m<Width/TILE_WIDTH;++m){
        Mds[ty][tx] = d_M[Row*Width+m*TILE_WIDTH+tx];
        Nds[ty][tx] = d_N[(m*TILE_WIDTH+ty)*Width+Col];
        __syncthreads();
        for(k=0;k<TILE_WIDTH;k++) Pvalue += Mds[ty][k]*Nds[k][tx];
        __syncthreads();
     }
     d_P[Row*Width+Col] = Pvalue;
}

long unsigned int get_tick()
{
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) return (0);
    return ts.tv_sec*(long int)1000 + ts.tv_nsec / (long int) 1000000;
}

void matMulDevice(float *h_M, float *h_N, float *h_P, int Width)
{
   // struct timeval tim;
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
   //cudaEvent_t start, stop;
   //cudaEventCreate(&start);
   //cudaEventCreate(&stop);
   //cudaEventRecord(start, 0);
   long int start = get_tick();
   matMulTiledKernel<<<dimGrid, dimBlock>>>(d_M, d_N, d_P, Width);
   cudaDeviceSynchronize();
   long int end = get_tick();
   long double elapsedTime = (end - start) / (float) 1000;
   printf("%Lf seconds elapsed\n", elapsedTime);
   //cudaEventRecord(stop, 0);
   //cudaEventSynchronize(stop);
   //float elapsedTime;
   //cudaEventElapsedTime(&elapsedTime, start, stop);
   //elapsedTime = elapsedTime / (float) 1000000;
// Step 4: Copy back result, and free memory on device
   cudaMemcpy(h_P, d_P, size, cudaMemcpyDeviceToHost);
   //printf("%.6Lf seconds elapsed\n", elapsedTime);
   cudaFree(d_M); cudaFree(d_N); cudaFree(d_P);
}

int main()
{
    float *h_M, *h_N, *h_P;
    int i, n = WIDTH, size=sizeof(float)*n*n;
    h_P = (float *)malloc(size);
    h_M = (float *)malloc(size);
    h_N = (float *)malloc(size);
    for(i=0;i<n*n;i++){*(h_M+i)=(float)i; 
                     *(h_N+i)=(float)i;}
    matMulDevice(h_M,h_N,h_P,n);
}
