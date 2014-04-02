#include <stdio.h>

void check_cells(float *rx, float *ry, float *rz, int *head, int *list, int mx, int my, int mz, int natoms, int step, int pstep, int gtx, int gty, int gtz)
{
   int i, icell, ix, iy, iz, icount;

   icount = 0;
   for(icell=0;icell<gtx*gty*gtz;icell++){
       ix = icell%gtx;
       iy = (icell/gtx)%gty;
       iz = icell/(gtx*gty);
       i = head[icell];
       if(((ix<0 || ix>mx)||(ix<0 || ix>mx)||(ix<0 || ix>mx))&&i!=-1){
           printf("Cell %d at (%d,%d,%d) is not empty\n",icell,ix,iy,iz);
           fflush(stdout);
       }
//       if(step==pstep) printf("\nCell number %d at (%d,%d,%d) contains particles:",icell,ix,iy,iz);
       while(i>=0){
           if(ix>0&&ix<(mx+1)&&iy>0&&iy<(my+1)&&iz>0&&iz<(mz+1)) icount++;
//	       if(step==pstep) {
//	           if(rx[i] < -0.5 || rx[i] > 0.5 || ry[i] < -0.5 || ry[i] > 0.5 || rz[i] < -0.5 || rz[i] > 0.5) printf("%d, r = (%f,%f,%f) ",i,rx[i],ry[i],rz[i]);
 //              }
//            printf(" (%f,%f,%f);",rx[i],ry[i],rz[i]);
	    i = list[i];
	}
    }
    if (icount != natoms) printf("\nNumber of particles in cells = %d\n",icount);
}
