#define BLOCK_WIDTH 512
__kernel void moveb (__global float *kineticArray, __global float *vx, __global float *vy, __global float *vz, __global float *fx, __global float *fy, __global float *fz, __private float dt, __private int natoms)
{
   __private float dt2;
   __private int element = get_global_id(0);
   dt2 = dt*0.5;
   __local float kinetic[BLOCK_WIDTH];
   kinetic[get_local_id(0)] = 0;
   __private float vxElement;
   __private float vyElement;
   __private float vzElement;
   if (element < natoms){
      vxElement = vx[element];
      vyElement = vy[element];
      vzElement = vz[element];
      vxElement = vxElement + dt2*fx[element];
      vyElement = vyElement + dt2*fy[element];
      vzElement = vzElement + dt2*fz[element];
      kinetic[get_local_id(0)] += vxElement*vxElement + vyElement*vyElement + vzElement*vzElement;
      vx[element] = vxElement;
      vy[element] = vyElement;
      vz[element] = vzElement;
   }
   int stride;
   for (stride = get_local_size(0)/2; stride > 0; stride>>=1)
   {
      barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
      if (get_local_id(0) < stride)
      {
         kinetic[get_local_id(0)] += kinetic[get_local_id(0) + stride];   
      }
      
   }
   barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
   if (get_local_id(0) == 0)
   {
      kineticArray[get_group_id(0)] = kinetic[0];
   }
}
