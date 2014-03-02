#ifndef SEQ
#include "mpi.h"
#endif

void sum_energies (v, k, w, fparam)
float   v;
float   k;
float   w;
float *fparam;
{
   float vg, wg, kg;
   float work1[3], work2[3];
      
   work1[0] = v;
   work1[1] = k;
   work1[2] = w;

#ifndef SEQ
   MPI_Allreduce (work1, work2, 3, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
#else
   work2[0] = work1[0];
   work2[1] = work1[1];
   work2[2] = work1[2];
#endif

   vg = work2[0];
   kg = work2[1];
   wg = work2[2];

   fparam[19] = vg;
   fparam[20] = wg;
   fparam[21] = kg;
}
