__kernel void performUpdatesKernel(__global float *d_phi, __global float *d_oldphi, __global int *d_mask, __private int nptsx, __private int nptsy)
{
	__private int Row = get_global_id(1);
	__private int Col = get_global_id(0);
	__private int x = Row*nptsx+Col;
	__private int xm = x-nptsx;
	__private int xp = x+nptsx;

	if(Col<nptsx && Row<nptsy)
		if (d_mask[x]) d_phi[x] = 0.25f*(d_oldphi[x+1]+d_oldphi[x-1]+d_oldphi[xp]+d_oldphi[xm]);
}

__kernel void doCopyKernel(__global float *d_phi, __global float *d_oldphi, __global int *d_mask, __private int nptsx, __private int nptsy)
{
	__private int Row = get_global_id(1);
	__private int Col = get_global_id(0);
	__private int x = Row*nptsx+Col;

	if(Col<nptsx && Row<nptsy)
		if (d_mask[x]) d_oldphi[x] = d_phi[x];
}