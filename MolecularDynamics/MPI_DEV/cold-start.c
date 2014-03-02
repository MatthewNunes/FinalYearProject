#include <stdio.h>

#ifndef SEQ
#include "mpi.h"
#endif

#include "moldyn.h"

#define kcell(A,B,C) ( (((A)+nprosx)%nprosx)+((((B)+nprosy)%nprosy) + (((C)+nprosz)%nprosz)*nprosy)*nprosx )

int chnksz;
int pperc;
int cperb;

void cold_start(rank, rx, ry, rz, vx, vy, vz, xi, yi, zi, xo, yo, zo, natm,natoms,scrat,iparam,fparam, bigbuf)
int   rank;
float rx[];
float ry[];
float rz[];
float vx[];
float vy[];
float vz[];
int   xi;
int   yi;
int   zi;
float xo;
float yo;
float zo;
int  *natm;
int natoms;
float *scrat;
int *iparam;
float *fparam;
float *bigbuf;
{
   int where, istart;
   float boxlx2, boxly2, boxlz2;
   int east, north, upx;
   int sizbuf, ibuf[2], i, npass, tag, nsum;
   int nchunk, p2c[MAXLST], holist[MAXLST];
#ifndef SEQ
   MPI_Status istatus;
#endif

   int nprosx = iparam[0];
   int nprosy = iparam[1];
   int nprosz = iparam[2];
   int nstart = iparam[7];

   float boxlx = fparam[6];
   float boxly = fparam[7];
   float boxlz = fparam[8];

   east  = kcell (xi+1, yi  , zi);
   north = kcell (xi  , yi+1, zi);
   upx   = kcell (xi  , yi  , zi+1);

   boxlx2 = 0.5*boxlx;
   boxly2 = 0.5*boxly;
   boxlz2 = 0.5*boxlz;

   sizbuf = sizeof(float)*6*natoms;
   if (I_AM_HOST) set_chunk (sizbuf, ibuf, &nchunk, holist, p2c,iparam,fparam, bigbuf);

#ifndef SEQ
   MPI_Bcast (ibuf, 2, MPI_INT, RANKHOST, MPI_COMM_WORLD);
#endif

   istart = ibuf[0];
   where  = ibuf[1];
   iparam[16] = istart;
   iparam[17] = where;

   if (I_AM_HOST) {
      if (nstart==3) 
         initialise_particles(&nchunk, natm, 
                     holist, p2c, rx, ry, rz, vx, vy, vz, xo, yo, zo,iparam,fparam,bigbuf);
/*      else if (nstart==5 || nstart==6) wstart(); */
   }
#ifndef SEQ
   else{
      for(;;){
         MPI_Recv (scrat, 6*natoms, MPI_FLOAT, MPI_ANY_SOURCE,
                          MPI_ANY_TAG, MPI_COMM_WORLD, &istatus);
         tag = istatus.MPI_TAG;
         if (tag==777) {
            npass = (int)(scrat[0]+0.1);
            npass = 6*npass + 1;
            for(i=2;i<=npass;i=i+6){
               rx[*natm] = scrat[i-1] - xo;
               ry[*natm] = scrat[i]   - yo;
               rz[*natm] = scrat[i+1] - zo;
               vx[*natm] = scrat[i+2];
               vy[*natm] = scrat[i+3];
               vz[*natm] = scrat[i+4];
               (*natm)++;
            }
         }
         else if (tag==555) {
            nsum = *natm;
            MPI_Allreduce (natm, &nsum, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
            break;
         }
         else {
            printf ("\n Error sending particles to %d",rank);
            break;
         }
      }
   }
#endif
}
