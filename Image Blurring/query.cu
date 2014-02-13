#include <cuda.h>
#include <stdio.h>

int main (int argc, char **argv){
    int ndev, maxtpb;
    cudaGetDeviceCount(&ndev);
    printf("Number of GPUs = %4d\n",ndev);
    for(int i=0;i<ndev;i++){
	cudaDeviceProp deviceProps;
  	cudaGetDeviceProperties(&deviceProps, i);
        maxtpb = deviceProps.maxThreadsPerBlock;
  	printf("GPU device %4d:\n\tName: %s:\n",i,deviceProps.name);
  	printf("\tCompute capabilities: SM %d.%d\n",
            deviceProps.major, deviceProps.minor);
        printf("\tMaximum number of threads per block: %4d\n",maxtpb);
        printf("\tMaximum number of threads per SM: %4d\n",
            deviceProps.maxThreadsPerMultiProcessor);
        printf("\tNumber of streaming multiprocessors: %4d\n",
            deviceProps.multiProcessorCount);
        printf("\tClock rate: %d KHz\n",deviceProps.clockRate);
        printf("\tGlobal memory: %lu bytes\n",deviceProps.totalGlobalMem);
    }
    cudaSetDevice(0);
}
