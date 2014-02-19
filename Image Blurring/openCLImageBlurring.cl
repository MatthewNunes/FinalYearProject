__kernel void performUpdatesKernel(__global int *restrict d_R, __global int *restrict d_G, __global int *restrict d_B, __global int *restrict d_Rnew, __global int *restrict d_Bnew, __global int *restrict d_Gnew, __private int rowsize, __private int colsize)
{
	__private int row = get_global_id(1);
	__private int col = get_global_id(0);
	__private int x = row*colsize+col;
	__private int xm = x-colsize;
	__private int xp = x+colsize;
	
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

__kernel void doCopyKernel( __global int *restrict d_R, __global int *restrict d_G, __global int *restrict d_B, __global int *restrict d_Rnew, __global int *restrict d_Bnew, __global int *restrict d_Gnew, __private int rowsize, __private int colsize)
{

	__private int row = get_global_id(1);
	__private int col = get_global_id(0);
	__private int x = row*colsize+col;

	if(col<colsize && row<rowsize)
	{    
	  d_R[x] = d_Rnew[x];
	  d_G[x] = d_Gnew[x];
	  d_B[x] = d_Bnew[x];
	}
	
}
