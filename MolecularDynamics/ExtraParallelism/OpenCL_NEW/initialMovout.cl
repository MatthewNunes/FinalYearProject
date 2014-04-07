#define BLOCK_WIDTH 512
__kernel void initialMovout(__global float *rx, __global float *ry, __global float *rz, __private int natoms)
{
	__private int element = get_global_id(0);
	if (element < natoms)
	{
		if(rx[element] < -0.5){ rx[element] += 1.0;}
		if(rx[element] >  0.5){ rx[element] -= 1.0;}
		if(ry[element] < -0.5){ ry[element] += 1.0;}
		if(ry[element] >  0.5){ ry[element] -= 1.0;}
		if(rz[element] < -0.5){ rz[element] += 1.0;}
		if(rz[element] >  0.5){ rz[element] -= 1.0;}
	}
}