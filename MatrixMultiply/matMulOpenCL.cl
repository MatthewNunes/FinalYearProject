__kernel void matMulKernel(__global float *d_M, __global float *d_N, __global float *d_P, const uint wwidth)
{
	int row = get_group_id(0) * get_num_groups(0) * + get_local_id(0);
	int col = get_group_id(1) *	get_num_groups(1) * + get_local_id(1);
	int k;
	int width = wwidth;
	if ((row < width)&&(col<width))
	{
		float pValue = 0.0;
		for (k  =0; k < width; k++)
		{
			pValue += d_M[row*width+k] * d_N[k*width+col];
		}
		d_P[row*width+col] = pValue;
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