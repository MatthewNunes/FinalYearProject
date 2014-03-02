/*
FILE  main.c

DOC   This program simulates a three-dimensional Lennard-Jones fluid.

LANG  C plus MPI message passing.

HIS   1) Parallel version originally written by University of
HIS      Southampton.
HIS   2) Adopted to run on the Intel iPSC/2 computer at Daresbury
HIS      Laboratory, and placed in the public domain through the
HIS      CCP5 programme.
HIS   3) Adopted to run on the Intel Paragon and enhanced by David W.
HIS      Walker at Oak Ridge National Laboratory, Tennessess, USA
HIS   4) Converted to use MPI message passing library by David W. Walker
HIS      at University of Wales Cardiff in July 1997.
HIS   5) Converted from Fortran to C in November 1997.

*/

#ifndef SEQ
#include "mpi.h"
#endif

#include <stdio.h>
#include <stdlib.h>
#include "moldyn.h"

#define alpha (10.0)
#define beta (0.2)

main (argc, argv)
int argc;
char *argv[];
{
   int rank;
   int xi, yi, zi;
   float xo, yo, zo;
   float *rx, *ry, *rz, *vx, *vy, *vz;
   int   *head, *list;
   int   natm=0;
   int ierror, input_parameters();
   int jstart, step, itemp;
   float potential, virial, kinetic;
   int i, icell;
   int map[27];
   int nsize2, nsize;
   float *ptemp1, *ptemp2;
   float *scratch;
   int iparam[23];
   float fparam[32];
   int   outfst=0;
   int   outfrm;
   float bigbuf[MYBUFSIZ];

#ifndef SEQ
   MPI_Init (&argc, &argv);
#endif

#ifndef SEQ
   MPI_Comm_rank (MPI_COMM_WORLD, &rank);
#include "moldyn.h"
#else
   rank = RANKHOST;
#endif

   ierror = input_parameters (rank, &xi, &yi, &zi, &xo, &yo, &zo,map,iparam,fparam);

   int nstep = iparam[3];
   int isave = iparam[5];
   int nstart = iparam[7];
   int isvunf = iparam[8];
   int natoms = iparam[11];
   int mx = iparam[13];
   int my = iparam[14];
   int mz = iparam[15];
   iparam[22] = outfst;

//   nsize2 = (int)(natoms*(1.0+beta));
//   nsize = (int)(natoms*(1.0+beta));
   nsize = NMAX;
   nsize2 = NMAX2;
//   head = (int *)malloc(sizeof(int)*mx*my*mz);
   head = (int *)malloc(sizeof(int)*NCELLS);
   list = (int *)malloc(sizeof(int)*nsize2);
   rx = (float *)malloc(sizeof(float)*nsize2);
   ry = (float *)malloc(sizeof(float)*nsize2);
   rz = (float *)malloc(sizeof(float)*nsize2);
   vx = (float *)malloc(sizeof(float)*nsize);
   vy = (float *)malloc(sizeof(float)*nsize);
   vz = (float *)malloc(sizeof(float)*nsize);
   ptemp1 = (float *)malloc(sizeof(float)*nsize);
   ptemp2 = (float *)malloc(sizeof(float)*nsize);
   scratch = (float *)malloc(sizeof(float)*6*nsize);

  
   if (ierror==0){
      if (nstart==1 || nstart==2) 
         printf ("\n Sorry, this option is not available");
      else if (nstart==3)
         cold_start (rank, rx, ry, rz, vx, vy, vz, 
                     xi, yi, zi, xo, yo, zo,&natm,nsize,scratch,iparam,fparam,bigbuf);
   }
   int istart = iparam[16];
   int where = iparam[17];

   loop_initialise (rank,iparam,fparam);

   if (where==1) {
     movout (rx, ry, rz, vx, vy, vz, head, list, xi, yi, zi, &natm,nsize2,scratch,ptemp1,ptemp2,natoms,iparam,fparam);
     force (&potential, &virial, rx, ry, rz, head, list, natm, map, scratch,iparam,fparam);
   }
      //printf ("\nReturned from force: potential = %f, virial = %f, kinetic = %f\n",potential, virial, kinetic);


   jstart = (istart==-1) ? 1 : istart + where;

   for(step=jstart;step<=nstep;step++){
      if ((where==1)||(step>jstart)) movea (rx, ry, rz, vx, vy, vz, natm, scratch, fparam);
      movout (rx, ry, rz, vx, vy, vz, head, list, xi, yi, zi, &natm,nsize2,scratch,ptemp1, ptemp2,natoms,iparam,fparam);
      itemp = abs(isave);
      outfrm = 0;
      if (step%itemp == 0) outfrm++;
      itemp = abs(isvunf);
      if (step%itemp ==0) outfrm += 10;
      iparam[21] = outfrm;
      if (outfrm>0) outcon (step, 0, rank, rx, ry, rz, vx, vy, vz, 
                            xo, yo, zo, natm, scratch,iparam, fparam, bigbuf);
      force (&potential, &virial, rx, ry, rz, head, list, natm, map, scratch,iparam,fparam);
      //printf ("\nReturned from force: potential = %f, virial = %f, kinetic = %f\n",potential, virial, kinetic);
      moveb (&kinetic, vx, vy, vz, natm, scratch, fparam);
      sum_energies (potential, kinetic, virial, fparam);
      hloop (kinetic, step, rank, vx, vy, vz, natm, iparam, fparam);
/*
      i = findit(15935,list,head);
      printf ("\n findit gives %d",i);
*/
   }
/*
   if (rank==0) {
      for(icell=0;icell<mx*my*mz;icell++){
         i = head[icell];
         while (i>0) {
            printf("\n%d %13.6f %13.6f %13.6f %13.6f %13.6f %13.6f",i,rx[i-1],
                    ry[i-1], rz[i-1], vx[i-1], vy[i-1], vz[i-1]);
            i = list[i-1];
         }
      }
   }
*/
      
   outfrm = 0;
   if (isave > 0) outfrm++;
   if (isvunf > 0) outfrm += 10;

   iparam[21] = outfrm;
   if (isave >0 || isvunf > 0) 
      outcon (nstep, 1, rank, rx, ry, rz, vx, vy, vz, xo, yo, zo, natm, scratch,iparam, fparam, bigbuf);

   if (I_AM_HOST) tidyup (iparam, fparam);
#ifndef SEQ
   MPI_Finalize ();
#endif
}
