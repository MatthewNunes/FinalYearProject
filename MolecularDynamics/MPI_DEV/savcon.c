#include <stdio.h>

#ifndef SEQ
#include "mpi.h"
#endif

#include "moldyn.h"

void savcon (istep, iwhere, natm, rx, ry, rz, vx, vy, vz, xor, yor, zor,iparam, fparam,bigbuf)
int istep;
int iwhere;
int natm;
float rx[];
float ry[];
float rz[];
float vx[];
float vy[];
float vz[];
float xor;
float yor;
float zor;
int *iparam;
float *fparam;
float *bigbuf;
{
   int npack, nrecv, pacsiz, npass, start, end;
   int i, j, k, iptr, kk;
   FILE *checkpoint;
   float posvel[INREAL];
#ifndef SEQ
   MPI_Status istatus;
#endif

   int nprosx = iparam[0];
   int nprosy = iparam[1];
   int nprosz = iparam[2];
   int natoms = iparam[11];
   int outfrm = iparam[21];
   int outfst = iparam[22];

   float ace   = fparam[23];
   float acv   = fparam[24];
   float ack   = fparam[25];
   float acp   = fparam[26];
   float acesq = fparam[27];
   float acvsq = fparam[28];
   float acksq = fparam[29];
   float acpsq = fparam[30];
   
   if (outfrm%10==1 && outfst%10==0){
      checkpoint = fopen ("checkpointseq.for","w"); 
      outfst++;
   }

   if ((outfrm/10)%10==1 && (outfst/10)%10==0) {
      printf ("\n Sorry - unformatted checkpointing not implemented yet");
      outfst += 10;
   }
   iparam[22] = outfst;

   if (outfrm%10==1){
      fprintf (checkpoint, "%8d%8d%8d",natoms,istep,iwhere);
      fprintf (checkpoint, "\n%18e%18e%18e%18e\n%18e%18e%18e%18e",
           ace, acv, ack,acp, acesq, acvsq, acksq, acpsq);
   }

   iptr = 0;
   for (i=0;i<nprosx*nprosy*nprosz;++i){
      if (i==RANKHOST) {
         pacsiz = (MAXPAS-1)/6;
         npack  = (natm-1)/pacsiz + 1;
      }
#ifndef SEQ
      else {
         MPI_Send (&i, 1, MPI_INT, i, i, MPI_COMM_WORLD);
         MPI_Recv (&npack, 1, MPI_INT, i, i, MPI_COMM_WORLD, &istatus);
      }
#endif
      for (j=1;j<=npack;++j){
         if (i==RANKHOST){
            start = (j-1)*pacsiz + 1;
            end   = min (start+pacsiz, natm);
            npass = 0;
            for (k=start-1;k<end;++k){
               npass += 6;
               bigbuf[npass-5] = rx[k] + xor;
               bigbuf[npass-4] = ry[k] + yor;
               bigbuf[npass-3] = rz[k] + zor;
               bigbuf[npass-2] = vx[k];
               bigbuf[npass-1] = vy[k];
               bigbuf[npass]   = vz[k];
            }
            bigbuf[0] = (float)(end-start+1);
         }
#ifndef SEQ
         else{
            MPI_Recv (bigbuf, MYBUFSIZ, MPI_FLOAT, i, i, MPI_COMM_WORLD, &istatus);

         }
#endif
         nrecv = 6*(int)(bigbuf[0]+0.1) + 1;
 
         for(k=2;k<=nrecv;k=k+6){
            iptr += 6;
            posvel[iptr-6] = bigbuf[k-1];
            posvel[iptr-5] = bigbuf[k];
            posvel[iptr-4] = bigbuf[k+1];
            posvel[iptr-3] = bigbuf[k+2];
            posvel[iptr-2] = bigbuf[k+3];
            posvel[iptr-1] = bigbuf[k+4];
            if (iptr==INREAL) {
               if (outfrm%10==1) {
                  fprintf (checkpoint, "\n");
                  for (kk=0;kk<iptr;++kk) fprintf (checkpoint, "%13.5e%s",posvel[kk],((kk+1)%6==0 && kk!=(iptr-1))? "\n" : " ");
               }
               iptr = 0;
            }
         }
      }
   }

   if (outfrm%10==1 && iptr!=0){
      fprintf (checkpoint, "\n");
      for (kk=0;kk<iptr;++kk) fprintf (checkpoint, "%13.5e%s",posvel[kk],((kk+1)%6==0 && kk!=(iptr-1)) ? "\n" : " ");
   }
}
