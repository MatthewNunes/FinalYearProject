#define TILE_WIDTH 16
#define WIDTH 3000

__kernel void matMulKernel(__global float *restrict d_M, __global float *restrict d_N, __global float *restrict d_P)
{
	__private int bx = get_group_id(0); 
	__private int by = get_group_id(1);
	__private int tx = get_local_id(0);
	__private int ty = get_local_id(1);
	__private int row = get_group_id(1) * TILE_WIDTH + ty;
	__private int col = get_group_id(0) * TILE_WIDTH + tx;
	__private int k, m;
	
	__local float d_Mds[TILE_WIDTH][TILE_WIDTH];
	__local float d_Nds[TILE_WIDTH][TILE_WIDTH];
	//__private int width = wwidth;
	__private float pValue = 0.0;
	
	for(m=0; m<WIDTH/TILE_WIDTH; ++m)
	{
		d_Mds[ty][tx] = d_M[row*WIDTH+m*TILE_WIDTH+tx];
		d_Nds[ty][tx] = d_N[(m*TILE_WIDTH+ty)* WIDTH+col];
		barrier(CLK_LOCAL_MEM_FENCE);
		
		for (k  =0; k < TILE_WIDTH; k++)	pValue += d_Mds[ty][k] * d_Nds[k][tx];
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	d_P[row*WIDTH+col] = pValue;
}

