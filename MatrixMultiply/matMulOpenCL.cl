#define WIDTH 3000

__kernel void matMulKernel(__global float *restrict d_M, __global float *restrict d_N, __global float *restrict d_P)
{
	__private int row = get_global_id(1);
	__private int col = get_global_id(0);
	__private int k;
	//__private int width = wwidth;
	if ((row < WIDTH)&&(col<WIDTH))
	{
		__private float pValue = 0.0;
		for (k  =0; k < WIDTH; k++)
		{
			pValue += d_M[row*WIDTH+k] * d_N[k*WIDTH+col];
		}
		d_P[row*WIDTH+col] = pValue;
	}
}

/**
    int Row = blockIdx.y*blockDim.y + threadIdx.y;
	int Col = blockIdx.x*blockDim.x + threadIdx.x;
	int k;
	if ((Row<Width)&&(Col<Width)){
		float Pvalue = 0.0;
		for(k=0;k<Width;k++)
			Pvalue += d_M[Row*Width+k]*d_N[k*Width+Col];
		d_P[Row*Width+Col] = Pvalue;
	}
*/
