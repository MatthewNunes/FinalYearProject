#define BLOCK_WIDTH 512
__kernel void movea (__global float *rx, __global float *ry, __global float *rz, __global float *vx, __global float *vy, __global float *vz, __global float *fx, __global float *fy, __global float *fz, __private float dt, __private int natoms)
{
   __private float dt2, dtsq2;
   __private int element = get_global_id(0);
   dt2 = dt*0.5;
   dtsq2 = dt*dt2;

   if(element < natoms){
      rx[element] = rx[element] + dt*vx[element] + dtsq2*fx[element];
      ry[element] = ry[element] + dt*vy[element] + dtsq2*fy[element];
      rz[element] = rz[element] + dt*vz[element] + dtsq2*fz[element];
      vx[element] = vx[element] + dt2*fx[element];
      vy[element] = vy[element] + dt2*fy[element];
      vz[element] = vz[element] + dt2*fz[element];
   }
}
