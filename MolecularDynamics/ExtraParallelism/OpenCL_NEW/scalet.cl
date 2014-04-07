#define BLOCK_WIDTH 512
__kernel void scalet (__global float *vx, __global float *vy, __global float *vz, __private float kinetic, __private float eqtemp, __private float tmpx, __private int iscale, __private int natoms, __private int step)
{
   __private float scalef;
   __private int element = get_global_id(0);
   if (step%iscale==0) scalef = sqrt((float)(eqtemp/tmpx));
   else scalef = sqrt ((float)(eqtemp/(2.0*kinetic/(3.0*(float)(natoms-1)))));

   if(element < natoms){
      vx[element] *= scalef;
      vy[element] *= scalef;
      vz[element] *= scalef;
   }
}
