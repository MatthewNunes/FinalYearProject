/*
LodePNG Examples

Copyright (c) 2005-2012 Lode Vandevenne

This software is provided 'as-is', without any express or implied
warranty. In no event will the authors be held liable for any damages
arising from the use of this software.

Permission is granted to anyone to use this software for any purpose,
including commercial applications, and to alter it and redistribute it
freely, subject to the following restrictions:

	1. The origin of this software must not be misrepresented; you must not
	claim that you wrote the original software. If you use this software
	in a product, an acknowledgment in the product documentation would be
	appreciated but is not required.

	2. Altered source versions must be plainly marked as such, and must not be
	misrepresented as being the original software.

	3. This notice may not be removed or altered from any source
	distribution.
*/

#include "lodepng.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/time.h>
#include <cuda.h>
#define ROWSIZE 521
#define COLSIZE 428
#define TX 16
#define TY 32

int *h_R;
int *h_G; 
int *h_B;
int *h_Rnew;
int *h_Gnew;
int *h_Bnew;

__global__
void performUpdatesKernel(int *d_R, int *d_G, int *d_B, int *d_Rnew, int *d_Bnew, int *d_Gnew, int rowsize, int colsize)
{
    int row = blockIdx.y*blockDim.y+threadIdx.y;
    int col = blockIdx.x*blockDim.x+threadIdx.x;
    int x = row*colsize+col;
    int xm = x-colsize;
    int xp = x+colsize;
    
    if ((row < rowsize) && (col < colsize))
    {
      if (row != 0 && row != (rowsize-1) && col != 0 && col != (colsize-1)){
        d_Rnew[x] = (d_R[x+1]+d_R[x-1]+d_R[xm]+d_R[xp])/4;
        d_Gnew[x] = (d_G[x+1]+d_G[x-1]+d_G[xm]+d_G[xp])/4;
        d_Bnew[x] = (d_B[x+1]+d_B[x-1]+d_B[xm]+d_B[xp])/4;
      }
      else if (row == 0 && col != 0 && col != (colsize-1)){
        d_Rnew[x] = (d_R[xp]+d_R[x+1]+d_R[x-1])/3;
        d_Gnew[x] = (d_G[xp]+d_G[x+1]+d_G[x-1])/3;
        d_Bnew[x] = (d_B[xp]+d_B[x+1]+d_B[x-1])/3;
      }
      else if (row == (rowsize-1) && col != 0 && col != (colsize-1)){
        d_Rnew[x] = (d_R[xm]+d_R[x+1]+d_R[x-1])/3;
        d_Gnew[x] = (d_G[xm]+d_G[x+1]+d_G[x-1])/3;
        d_Bnew[x] = (d_B[xm]+d_B[x+1]+d_B[x-1])/3;
      }
      else if (col == 0 && row != 0 && row != (rowsize-1)){
        d_Rnew[x] = (d_R[xp]+d_R[xm]+d_R[x+1])/3;
        d_Gnew[x] = (d_G[xp]+d_G[xm]+d_G[x+1])/3;
        d_Bnew[x] = (d_B[xp]+d_B[xm]+d_B[x+1])/3;
      }
      else if (col == (colsize-1) && row != 0 && row != (rowsize-1)){
        d_Rnew[x] = (d_R[xp]+d_R[xm]+d_R[x-1])/3;
        d_Gnew[x] = (d_G[xp]+d_G[xm]+d_G[x-1])/3;
        d_Bnew[x] = (d_B[xp]+d_B[xm]+d_B[x-1])/3;
      }
      else if (row==0 && col==0){
        d_Rnew[x] = (d_R[x+1]+d_R[xp])/2;
        d_Gnew[x] = (d_G[x+1]+d_G[xp])/2;
        d_Bnew[x] = (d_B[x+1]+d_B[xp])/2;
      }
      else if (row==0 && col==(colsize-1)){
        d_Rnew[x] = (d_R[x-1]+d_R[xp])/2;
        d_Gnew[x] = (d_G[x-1]+d_G[xp])/2;
        d_Bnew[x] = (d_B[x-1]+d_B[xp])/2;
      }
      else if (row==(rowsize-1) && col==0){
        d_Rnew[x] = (d_R[x+1]+d_R[xm])/2;
        d_Gnew[x] = (d_G[x+1]+d_G[xm])/2;
        d_Bnew[x] = (d_B[x+1]+d_B[xm])/2;
      }
      else if (row==(rowsize-1) && col==(colsize-1)){
        d_Rnew[x] = (d_R[x-1]+d_R[xm])/2;
        d_Gnew[x] = (d_G[x-1]+d_G[xm])/2;
        d_Bnew[x] = (d_B[x-1]+d_B[xm])/2;
      }
    }
     
}

__global__
void doCopyKernel(int *d_R, int *d_G, int *d_B, int *d_Rnew, int *d_Bnew, int *d_Gnew, int rowsize, int colsize)
{

    int row = blockIdx.y*blockDim.y+threadIdx.y;
    int col = blockIdx.x*blockDim.x+threadIdx.x;
    int x = row*colsize+col;

    if(col<colsize && row<rowsize)
    {    
      d_R[x] = d_Rnew[x];
      d_G[x] = d_Gnew[x];
      d_B[x] = d_Bnew[x];
    }
    
}

/*
3 ways to decode a PNG from a file to RGBA pixel data (and 2 in-memory ways).
*/

/*
Example 1
Decode from disk to raw pixels with a single function call
*/
void decodeOneStep(const char* filename)
{
  unsigned error;
  unsigned char* image;
  unsigned width, height;

  error = lodepng_decode24_file(&image, &width, &height, filename);
  if(error) printf("error %u: %s\n", error, lodepng_error_text(error));
  printf("I get here!\n");
 // while((*image) != '\0')
  //{
//	printf("%d \n", *image);
//	image++;
//  }
  /*use image here*/
  int count = 0;
  int rowsize = height;
  int colsize = width;
  int nlines = 0;
  //  printf("I get here %d", width);
  int *R= (int *)malloc(sizeof(int) * height * width);
  int *G= (int *)malloc(sizeof(int) * height * width);
  int *B= (int *)malloc(sizeof(int) * height * width);
  int it = 0;

  unsigned char *newImage = (unsigned char *)malloc(sizeof(unsigned char) * width * height * 3);
 
  //printf("I get here 3");
  int  maxlen = width;
  char str[maxlen], lines[height][maxlen];
	unsigned int h1, h2, h3;
  unsigned char *sptr;
  int row = 0; 
  int col = 0;
  int total = width * height;
  int totalReached =0;
  int i =0, j=0;
  int nblurs;

  printf("Width: %d\n", width);
  printf("Height: %d\n", height);

  char *first = (char *)malloc(sizeof(char));
  char *second = (char *)malloc(sizeof(char));
  char *third = (char *)malloc(sizeof(char));
  char *firstPointer = first;
  char *secondPointer = second;
  char *thirdPointer = third;
  for(i=0,j=0;i<3*width*height;i+=3,j++)
  {
    sprintf(first, "%02x", *(image + i));
    sprintf(second, "%02x", *(image + i+1));
    sprintf(third, "%02x", *(image + i+2));
    *(R+j) = strtol(first, NULL, 16);
    *(G+j) = strtol(second, NULL, 16);
    *(B+j) = strtol(third, NULL, 16);
    first = firstPointer;
    second = secondPointer;
    third = thirdPointer;
   }
  
  
///ADD IT HERE
  int size = rowsize * colsize; 
  int *d_R;
  int *d_G;
  int *d_B;
  int *d_Rnew;
  int *d_Gnew;
  int *d_Bnew;
  h_Rnew = (int *)malloc(sizeof(int) * size);
  h_Gnew = (int *)malloc(sizeof(int) * size);
  h_Bnew = (int *)malloc(sizeof(int) * size);

  
  nblurs = 10;
  printf("\nGive the number of times to blur the image\n");
  int icheck = scanf ("%d", &nblurs);
  cudaMalloc((void **)&d_R, size *sizeof(int));
  cudaMalloc((void **)&d_G, size*sizeof(int));
  cudaMalloc((void **)&d_B, size*sizeof(int));
  cudaMalloc((void **)&d_Rnew, size*sizeof(int));
  cudaMalloc((void **)&d_Gnew, size*sizeof(int));
  cudaMalloc((void **)&d_Bnew, size*sizeof(int));
  
  cudaMemcpy(d_R, R, size*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_G, G, size*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B, size*sizeof(int), cudaMemcpyHostToDevice);
  dim3 dimGrid(ceil(colsize/(float)TX),ceil(rowsize/(float)TY),1);
  dim3 dimBlock(TX,TY,1);
  int k;
  for(k=0;k<nblurs;k++){
    performUpdatesKernel<<<dimGrid,dimBlock>>>(d_R, d_G, d_B, d_Rnew, d_Gnew, d_Bnew, rowsize, colsize);
    doCopyKernel<<<dimGrid, dimBlock>>>(d_R, d_G, d_B, d_Rnew, d_Gnew, d_Bnew, rowsize, colsize);
  }
  
  cudaMemcpy(h_Rnew, d_Rnew, size*sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_Gnew, d_Gnew, size*sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_Bnew, d_Bnew, size*sizeof(int), cudaMemcpyDeviceToHost);
  cudaFree(d_R);
  cudaFree(d_G);
  cudaFree(d_B);
  cudaFree(d_Rnew);
  cudaFree(d_Gnew);
  cudaFree(d_Bnew);
  
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

//ENDS HERE

   
  unsigned char *newImage2 = newImage;
  int lineno =0, linelen=200;
  printf("I get here\n");
  char *convertMe = (char *)malloc(sizeof(char) * 1);
  char *convertMePointer = convertMe;
	for(row=0;row<rowsize;row++){
    for (col=0;col<colsize;col++){
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
  printf("I get here 2\n");

  size_t pngsize = width * height * 3;
  printf("I get here 3\n");
  unsigned error2 = lodepng_encode24_file("file.png",newImage, width, height);
  printf("I get here 4\n");
  printf("I get here 5\n");
  free(image);
  free(newImage);
  int f = 0;
  free(R);
  free(G);
  free(B);

}


int main(int argc, char *argv[])
{
  const char* filename = argc > 1 ? argv[1] : "Laplace.png";

  decodeOneStep("Laplace.png");
  
  return 0;
}
