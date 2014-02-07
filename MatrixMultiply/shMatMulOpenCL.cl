__kernel void matMulKernel(__global float *d_M, __global float *d_N, __local float *d_Mds, __local float *d_Nds, __global float *d_P, const uint wwidth, const uint twidth)
{
	__private int bx = get_group_id(0); 
	__private int by = get_group_id(1);
	__private int tx = get_local_id(0);
	__private int ty = get_local_id(1);
	__private int row = by * twidth + ty;
	__private int col = bx * twidth + tx;
	__private int k, m;
	__private int width = wwidth;
	__private float pValue = 0.0;
	
	for(m=0; m<wwidth/twidth; ++m)
	{
		d_Mds[ty * twidth + tx] = d_M[row*wwidth+m*twidth+tx];
		d_Nds[ty * twidth + tx] = d_N[(m*twidth+ty)* wwidth+col];
		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
		
		for (k  =0; k < twidth; k++)	pValue += d_Mds[ty*twidth+k] * d_Nds[k*twidth+tx];
		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
	}
	d_P[row*width+col] = pValue;
}

/**
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
*/