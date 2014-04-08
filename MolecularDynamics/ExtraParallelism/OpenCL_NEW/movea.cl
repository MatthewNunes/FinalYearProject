#define BLOCK_WIDTH 512
__kernel void movea (__global float *rx, __global float *ry, __global float *rz, __global float *vx, __global float *vy, __global float *vz, __global float *fx, __global float *fy, __global float *fz, __private float dt, __private int natoms)
{
   __private float dt2, dtsq2;
   __private int element = get_global_id(0);
   dt2 = dt*0.5;
   dtsq2 = dt*dt2;
   __private float fxElement;
   __private float fyElement;
   __private float fzElement;
   __private float vxElement;
   __private float vyElement;
   __private float vzElement;
   if(element < natoms){
      fxElement = fx[element];
      fyElement = fy[element];
      fzElement = fz[element];
      vxElement = vx[element];
      vyElement = vy[element];
      vzElement = vz[element];
      rx[element] = rx[element] + dt*vxElement + dtsq2*fxElement;
      ry[element] = ry[element] + dt*vyElement + dtsq2*fyElement;
      rz[element] = rz[element] + dt*vzElement + dtsq2*fzElement;
      vx[element] = vxElement + dt2*fxElement;
      vy[element] = vyElement + dt2*fyElement;
      vz[element] = vzElement + dt2*fzElement;
   }
}
