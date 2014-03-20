#include <stdio.h>

void check_cells(float *rx, float *ry, float *rz, int *head, int *list, int mx, int my, int mz, int natoms, int step, int pstep)
{
   int i, icell, ix, iy, iz, icount;

   icount = 0;
   for(icell=0;icell<(mx+2)*(my+2)*(mz+2);icell++){
       ix = icell%(mx+2);
       iy = (icell/(mx+2))%(my+2);
       iz = icell/((mx+2)*(my+2));
       i = head[icell];
       if(step==pstep) printf("\nCell number %d at (%d,%d,%d) contains particles:",icell,ix,iy,iz);
       while(i>=0){
           if(ix>0&&ix<(mx+1)&&iy>0&&iy<(my+1)&&iz>0&&iz<(mz+1)) icount++;
//	       if(step==pstep) {
//	           if(rx[i] < -0.5 || rx[i] > 0.5 || ry[i] < -0.5 || ry[i] > 0.5 || rz[i] < -0.5 || rz[i] > 0.5) printf("%d, r = (%f,%f,%f) ",i,rx[i],ry[i],rz[i]);
 //              }
            printf(" (%f,%f,%f);",rx[i],ry[i],rz[i]);
	    i = list[i];
	}
    }
    if (icount != natoms) printf("\nNumber of particles in cells = %d\n",icount);
}
