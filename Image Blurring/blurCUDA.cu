#include <stdio.h>
#include <stdlib.h>
#include <math.h>
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


void convertToOneD(int R[ROWSIZE][COLSIZE], int G[ROWSIZE][COLSIZE], int B[ROWSIZE][COLSIZE], int *h_R, int *h_G, int *h_B)
{
    int i =0;
    int j =0;
    //int totalSize = ROWSIZE * COLSIZE;

    for (i =0; i < ROWSIZE; i++)
    {
      for (j=0; j < COLSIZE; j++)
      {
	  	h_R[i*COLSIZE+j] = R[i][j];
		h_G[i*COLSIZE+j] = G[i][j];
		h_B[i*COLSIZE+j] = B[i][j];
      }
    }

}

void getDimensions(char *filename, unsigned int *width, unsigned int *height)
{
	printf("I get here 44\n");
	FILE *ff;
	printf("I get here 55\n");
	unsigned char *buffer;
	printf("I get here 66\n");
	unsigned char widthBuffer[4];
	printf("I get here 77\n");
	unsigned char heightBuffer[4];
	printf("I get here 88\n");
	unsigned char *widthString;
	printf("I get here 99\n");
	unsigned char *heightString;
	printf("I get here 1\n");
	ff = fopen(filename, "rb");
	printf("I get here 2\n");
	buffer = (unsigned char *) malloc (sizeof(unsigned char)*24);
	printf("I get here 3\n");
	widthString = (unsigned char *) malloc(sizeof(unsigned char) * 4);
	printf("I get here 4\n");
	heightString = (unsigned char *)malloc(sizeof(unsigned char) * 4);
	printf("I get here 5\n");
	size_t result = fread (buffer,1,24,ff);
	printf("I get here 6\n");
	int i = 0;
	int j = 0;
	for (i = 16; i < 20; i++)
	{
		widthBuffer[j] = *(buffer+i);
		j++;
	}
	printf("I get here 7\n");
	sprintf((char *)widthString, "%02X%02X%02X%02X", widthBuffer[0], widthBuffer[1], widthBuffer[2], widthBuffer[3]);
	printf("I get here 8\n");
	//printf("String: %s\n", widthString); 
	*width = (int)strtol((const char *)widthString, NULL, 16);
	printf("I get here 9\n");
	//printf("Width: %d\n",width);
	
	j= 0;
	for (i = 20; i < 24; i++)
	{
		heightBuffer[j] = *(buffer+i);
		j++;
	}
	printf("I get here 10\n");
	sprintf((char *)heightString, "%02X%02X%02X%02X", heightBuffer[0], heightBuffer[1], heightBuffer[2], heightBuffer[3]);
	printf("I get here 11\n");
	//printf("String: %s\n", heightString);
	*height = (int)strtol((const char *)heightString, NULL, 16);
	printf("I get here 12\n");
	//printf("Height: %d\n", height);
	fclose(ff);
	printf("I get here 13\n");
	free(buffer);
	printf("I get here 14\n");
	free(widthString);
	printf("I get here 15\n");
	free(heightString);
	printf("I get here 16\n");
	return;
}

int main (int argc, const char * argv[]) {
    // insert code here...
	unsigned int width;
	unsigned int height;
	getDimensions("./test2.png", &width, &height);
	struct timeval tim;
	gettimeofday(&tim, NULL);
	double tTotal1=tim.tv_sec+(tim.tv_usec/1000000.0);
	int size = width * height;
	static int const maxlen = 200, rowsize = height, colsize = width, linelen = 12;
	char str[maxlen], lines[5][maxlen];
	FILE *fp, *fout;
	int nlines = 0;
	unsigned int h1, h2, h3;
	char *sptr;
	int R[rowsize][colsize], G[rowsize][colsize], B[rowsize][colsize];
	//int Rnew[rowsize][colsize], Gnew[rowsize][colsize], Bnew[rowsize][colsize];
	int row = 0, col = 0, nblurs, lineno, k;
	int *d_R;
	int *d_G;
	int *d_B;
	int *d_Rnew;
	int *d_Gnew;
	int *d_Bnew;
	h_R = (int *)malloc(sizeof(int) * size);
	h_G = (int *)malloc(sizeof(int) * size);
	h_B = (int *)malloc(sizeof(int) * size);
	h_Rnew = (int *)malloc(sizeof(int) * size);
	h_Gnew = (int *)malloc(sizeof(int) * size);
	h_Bnew = (int *)malloc(sizeof(int) * size);
	gettimeofday(&tim, NULL);
	double t1=tim.tv_sec+(tim.tv_usec/1000000.0);
	fp = fopen("./test2.png", "r");
	
	while(! feof(fp))
	{
		fscanf(fp, "\n%[^\n]", str);
		if (nlines < 5) {strcpy((char *)lines[nlines++],(char *)str);}
		else{
			for (sptr=&str[0];*sptr != '\0';sptr+=6){
				sscanf(sptr,"%2x",&h1);
				sscanf(sptr+2,"%2x",&h2);
				sscanf(sptr+4,"%2x",&h3);
				
				if (col==colsize){
					col = 0;
					row++;
				}
				if (row < rowsize) {
					R[row][col] = h1;
					G[row][col] = h2;
					B[row][col] = h3;
				}
				col++;
			}
		}
	}
	fclose(fp);
	gettimeofday(&tim, NULL);
	double t2=tim.tv_sec+(tim.tv_usec/1000000.0);
	printf("Input image: %.6lf seconds elapsed\n", t2-t1);
	
	nblurs = 10;
      	printf("\nGive the number of times to blur the image\n");
      	int icheck = scanf ("%d", &nblurs);
	int i;
	int j;
    for (i =0; i < rowsize; i++)
	{
	  for (j=0; j < colsize; j++)
	  {
		h_R[i*colsize+j] = R[i][j];
		h_G[i*colsize+j] = G[i][j];
		h_B[i*colsize+j] = B[i][j];
	  }
	}
	//convertToOneD(R, G, B, h_R, h_B, h_G);
	
	gettimeofday(&tim, NULL);
	double t3=tim.tv_sec+(tim.tv_usec/1000000.0);
	cudaMalloc((void **)&d_R, size *sizeof(int));
	cudaMalloc((void **)&d_G, size*sizeof(int));
	cudaMalloc((void **)&d_B, size*sizeof(int));
	cudaMalloc((void **)&d_Rnew, size*sizeof(int));
	cudaMalloc((void **)&d_Gnew, size*sizeof(int));
	cudaMalloc((void **)&d_Bnew, size*sizeof(int));
	gettimeofday(&tim, NULL);
	double t4=tim.tv_sec+(tim.tv_usec/1000000.0);
	printf("Allocate Memory for image: %.6lf seconds elapsed\n", t4-t3);
	
	cudaMemcpy(d_R, h_R, size*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_G, h_G, size*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, size*sizeof(int), cudaMemcpyHostToDevice);
	dim3 dimGrid(ceil(colsize/(float)TX),ceil(rowsize/(float)TY),1);
	dim3 dimBlock(TX,TY,1);
	
	//gettimeofday(&tim, NULL);
	//double t1=tim.tv_sec+(tim.tv_usec/1000000.0);
	gettimeofday(&tim, NULL);
	double t5=tim.tv_sec+(tim.tv_usec/1000000.0);
	for(k=0;k<nblurs;k++){
	  performUpdatesKernel<<<dimGrid,dimBlock>>>(d_R, d_G, d_B, d_Rnew, d_Gnew, d_Bnew, rowsize, colsize);
	  doCopyKernel<<<dimGrid, dimBlock>>>(d_R, d_G, d_B, d_Rnew, d_Gnew, d_Bnew, rowsize, colsize);
	}
	gettimeofday(&tim, NULL);
	double t6=tim.tv_sec+(tim.tv_usec/1000000.0);
	printf("Blur image: %.6lf seconds elapsed\n", t6-t5);
	
	//gettimeofday(&tim, NULL);
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
	    R[kk][mm] = h_Rnew[colsize*kk +mm];
	    //printf(" %d ", h_Rnew[mm*kk +mm]);
	    G[kk][mm] = h_Gnew[colsize*kk +mm];
	    B[kk][mm] = h_Bnew[colsize*kk +mm];
	    
	  }
	}
	
	//double t2=tim.tv_sec+(tim.tv_usec/1000000.0);
	
	gettimeofday(&tim, NULL);
	double t7=tim.tv_sec+(tim.tv_usec/1000000.0);
	fout= fopen("./DavidBlur.png", "w");
	for (k=0;k<nlines;k++) fprintf(fout,"\n%s", lines[k]);
	fprintf(fout,"\n");
	for(row=0;row<rowsize;row++){
		for (col=0;col<colsize;col++){
			fprintf(fout,"%02x%02x%02x",R[row][col],G[row][col],B[row][col]);
			lineno++;
			if (lineno==linelen){
				fprintf(fout,"\n");
				lineno = 0;
			}
		}
	}
	fclose(fout);
	gettimeofday(&tim, NULL);
	double t8=tim.tv_sec+(tim.tv_usec/1000000.0);
	printf("Output image: %.6lf seconds elapsed\n", t8-t7);
	
	gettimeofday(&tim, NULL);
	double tTotal2=tim.tv_sec+(tim.tv_usec/1000000.0);
	printf("Total Time: %.6lf seconds elapsed\n", tTotal2-tTotal1);
    return 0;
}
