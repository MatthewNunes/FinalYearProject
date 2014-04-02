#include <stdio.h>
#include "moldyn.h"

int input_parameters (float *sigma, float *rcut, float *dt, float *eqtemp, float *dens, float *boxlx, float *boxly, float *boxlz, float *sfx, float *sfy, float *sfz, float *sr6, float *vrcut, float *dvrcut, float *dvrc12, float *freex, int *nstep, int *nequil, int *iscale, int *nc, int *natoms, int *mx, int *my, int *mz, int *iprint, int*tbx, int *tby, int *tbz)
{
   int abort;

   abort = read_input (sigma, rcut, dt, eqtemp, dens, nstep, nequil, iscale, iprint, nc, tbx, tby, tbz);
//   printf ("\nReturned from read_input, nc = %d\n", *nc);

   if (abort!=0) return (abort);
   
   *natoms = 4*(*nc)*(*nc)*(*nc);
   *freex  = (float)(3*(*natoms-1));
   *boxlx = 1.0;
   *boxly = 1.0;
   *boxlz = 1.0;
   *mx = (int)(*boxlx/(*rcut));
   *my = (int)(*boxly/(*rcut));
   *mz = (int)(*boxlz/(*rcut));
   *sfx = *boxlx/(float)(*mx);
   *sfy = *boxly/(float)(*my);
   *sfz = *boxlz/(float)(*mz);

   //set_nearest_cells (map, mx, my, mz);

   *sr6    = (*sigma/(*rcut))*(*sigma/(*rcut));
   *sr6    = (*sr6)*(*sr6)*(*sr6);
   *vrcut  = (*sr6)*((*sr6)-1.0);
   *dvrcut = -1.0*(*sr6)*((*sr6)-0.5)/(*rcut);
   *dvrc12 = 12.0*(*dvrcut);

   printf ("\n NUMBER OF CELLS                    (%4d, %4d, %4d)",*mx,*my,*mz);
   printf ("\n NUMBER OF THREADS PER BLOCK        (%4d, %4d, %4d)",*tbx,*tby,*tbz);
   
   return (abort);
}
