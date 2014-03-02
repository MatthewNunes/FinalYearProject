#ifndef SEQ
#include "mpi.h"
#endif

#include "moldyn.h"

void outcon (istep, iwhere, rank, rx, ry, rz, vx, vy, vz, xor, yor, zor, natm,scrat,iparam, fparam, bigbuf)
int   istep;
int   iwhere;
int   rank;
float rx[];
float ry[];
float rz[];
float vx[];
float vy[];
float vz[];
float xor;
float yor;
float zor;
int   natm;
float *scrat;
int *iparam;
float *fparam;
float *bigbuf;
{
   int i, j, npass, ipack, npack, pacsiz;
#ifndef SEQ
   MPI_Status istatus;
#endif

   if (I_AM_HOST) 
      savcon (istep, iwhere, natm, rx, ry, rz, vx, vy, vz, xor, yor, zor,iparam, fparam,bigbuf);
#ifndef SEQ
   else {
      MPI_Recv (&npack, 1, MPI_INT, RANKHOST, rank, MPI_COMM_WORLD, &istatus);
      pacsiz = (MAXPAS-1)/6;
      npack  = (natm-1)/pacsiz + 1;

      MPI_Send (&npack, 1, MPI_INT, RANKHOST, rank, MPI_COMM_WORLD);

      ipack = 0;
      npass = 0;
      j     = 0;

      for (i=1;i<=natm;++i){
         npass += 6;
         scrat[npass-5] = rx[j] + xor;
         scrat[npass-4] = ry[j] + yor;
         scrat[npass-3] = rz[j] + zor;
         scrat[npass-2] = vx[j];
         scrat[npass-1] = vy[j];
         scrat[npass]   = vz[j];
         j++;
         if (npass/6 == pacsiz) {
            ipack++;
            scrat[0] = (float)(npass/6);
            MPI_Send (scrat, npass+1, MPI_FLOAT, RANKHOST, rank, MPI_COMM_WORLD);
            npass = 0;
         }
      }
      if (ipack < npack) {
         scrat[0] = (float)(npass/6);
         MPI_Send (scrat, npass+1, MPI_FLOAT, RANKHOST, rank, MPI_COMM_WORLD);
     }
   }
#endif
}
