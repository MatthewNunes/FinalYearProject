#include <stdio.h>
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

int main (int argc, const char * argv[]) {
    // insert code here...
	struct timeval tim;
	gettimeofday(&tim, NULL);
	double tTotal1=tim.tv_sec+(tim.tv_usec/1000000.0);
	int size = ROWSIZE * COLSIZE;
	static int const maxlen = 200, rowsize = 521, colsize = 428, linelen = 12;
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
	fp = fopen("./David.ps", "r");
	
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
	
	convertToOneD(R, G, B, h_R, h_B, h_G);
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
	dim3 dimGrid(ceil(COLSIZE/(float)TX),ceil(ROWSIZE/(float)TY),1);
	dim3 dimBlock(TX,TY,1);
	
	//gettimeofday(&tim, NULL);
	//double t1=tim.tv_sec+(tim.tv_usec/1000000.0);
	gettimeofday(&tim, NULL);
	double t5=tim.tv_sec+(tim.tv_usec/1000000.0);
	for(k=0;k<nblurs;k++){
	  performUpdatesKernel<<<dimGrid,dimBlock>>>(d_R, d_G, d_B, d_Rnew, d_Gnew, d_Bnew, ROWSIZE, COLSIZE);
	  doCopyKernel<<<dimGrid, dimBlock>>>(d_R, d_G, d_B, d_Rnew, d_Gnew, d_Bnew, ROWSIZE, COLSIZE);
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
	for (kk = 0; kk < ROWSIZE; kk++)
	{
	  for (mm=0; mm<COLSIZE; mm++)
	  {
	    R[kk][mm] = h_Rnew[COLSIZE*kk +mm];
	    //printf(" %d ", h_Rnew[mm*kk +mm]);
	    G[kk][mm] = h_Gnew[COLSIZE*kk +mm];
	    B[kk][mm] = h_Bnew[COLSIZE*kk +mm];
	    
	  }
	}
	
	//double t2=tim.tv_sec+(tim.tv_usec/1000000.0);
	
	gettimeofday(&tim, NULL);
	double t7=tim.tv_sec+(tim.tv_usec/1000000.0);
	fout= fopen("./DavidBlur.ps", "w");
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
