#ifndef SEQ
#include "mpi.h"
#endif

#include "moldyn.h"

void flush_buffer (bigbuf, cf, nchunk, p2c, holist, natm, 
                   rx, ry, rz, vx, vy, vz, xor, yor, zor, iparam)
float  bigbuf[];
float  cf;
int   *nchunk;
int    p2c[];
int    holist[];
int   *natm;
float rx[];
float ry[];
float rz[];
float vx[];
float vy[];
float vz[];
float  xor;
float  yor;
float  zor;
int *iparam;
{
   int nclear, nleft, cid, pid, nsend;
   int boff;
   float btemp;
  
   int cperb  = iparam[18];
   int chnksz = iparam[20];

   nclear = (int)((*nchunk-1)*cf+1.1);
   nleft  = *nchunk - nclear;
   cid    = *nchunk;

   for(;;){
      boff = (cid-1)*chnksz + 1;
      btemp = bigbuf[boff-1] + 0.1;
      if (btemp > 0.0) {
         pid = (int)btemp;
         nsend = (int)(bigbuf[boff] + 0.1);
         if (pid==RANKHOST) 
            tohost (&bigbuf[boff+1], nsend, natm, rx, ry, rz, vx, vy, vz, 
                    xor, yor, zor);
#ifndef SEQ
         else 
            MPI_Send (&bigbuf[boff], 6*nsend+1, MPI_FLOAT, pid, 777,
                        MPI_COMM_WORLD);
#endif
         bigbuf[boff-1] = -1.0;
         bigbuf[boff]   =  0.0;
         p2c[pid] = -1;
         *nchunk = *nchunk - 1;
         holist[cperb-*nchunk-1] = cid;
      }
      cid--;
      if (*nchunk <= nleft) break;
   }
}
