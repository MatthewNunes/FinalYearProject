#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <time.h>
#define NSTEPS 500
#define TX 16
#define TY 32
#define NPTSX 300
#define NPTSY 300

__global__ 
void performUpdatesKernel(float *d_phi, float *d_oldphi, int *d_mask, int nptsx, int nptsy)
{
    int Row = blockIdx.y*blockDim.y+threadIdx.y;
    int Col = blockIdx.x*blockDim.x+threadIdx.x;
    int x = Row*nptsx+Col;
    int xm = x-nptsx;
    int xp = x+nptsx;

    if(Col<nptsx && Row<nptsy)
        if (d_mask[x]) d_phi[x] = 0.25f*(d_oldphi[x+1]+d_oldphi[x-1]+d_oldphi[xp]+d_oldphi[xm]);
}
__global__
void doCopyKernel(float *d_phi, float *d_oldphi, int *d_mask, int nptsx, int nptsy)
{
    int Row = blockIdx.y*blockDim.y+threadIdx.y;
    int Col = blockIdx.x*blockDim.x+threadIdx.x;
    int x = Row*nptsx+Col;

    if(Col<nptsx && Row<nptsy)
        if (d_mask[x]) d_oldphi[x] = d_phi[x];
}

long unsigned int get_tick()
{
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) return (0);
    return ts.tv_sec*(long int)1000 + ts.tv_nsec / (long int) 1000000;
}


void performUpdates(float *h_phi, float * h_oldphi, int *h_mask, int nptsx, int nptsy, int nsteps)
{
    float *d_phi, *d_oldphi;
    int *d_mask;
    int k;
    int sizef = sizeof(float)*nptsx*nptsy;
    int sizei = sizeof(int)*nptsx*nptsy;
    cudaMalloc((void **)&d_phi,sizef);
    cudaMalloc((void **)&d_oldphi,sizef);
    cudaMalloc((void **)&d_mask,sizei);
    cudaMemcpy(d_oldphi,h_oldphi,sizef,cudaMemcpyHostToDevice);
    cudaMemcpy(d_mask,h_mask,sizei,cudaMemcpyHostToDevice);
    dim3 dimGrid(ceil(nptsx/(float)TX),ceil(nptsy/(float)TY),1);
    dim3 dimBlock(TX,TY,1);
    long int start = get_tick();
    for(k=0;k<nsteps;++k){
        performUpdatesKernel<<<dimGrid,dimBlock>>>(d_phi,d_oldphi,d_mask,nptsx,nptsy);
        doCopyKernel<<<dimGrid,dimBlock>>>(d_phi,d_oldphi,d_mask,nptsx,nptsy);
    } 
    cudaDeviceSynchronize();
    long int end = get_tick();
    long double elapsedTime = (end - start)/ (float) 1000;
    printf("%Lf seconds elapsed\n", elapsedTime);
    cudaMemcpy(h_phi,d_oldphi,sizef,cudaMemcpyDeviceToHost);
    cudaFree(d_phi); cudaFree(d_oldphi); cudaFree(d_mask);
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
 
   //output_array (h_phi, NPTSX, NPTSY);
 
   return 0;
}
