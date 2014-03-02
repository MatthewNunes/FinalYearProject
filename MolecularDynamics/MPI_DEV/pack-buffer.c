#ifndef SEQ
#include "mpi.h"
#endif

#include "moldyn.h"

void pack_buffer (bigbuf, cid, natm, nchunk, p2c, holist, x, y, z, vvx, vvy, vvz,
                  rx, ry, rz, vx, vy, vz, xor, yor, zor, iparam)
float *bigbuf;
int    cid;
int   *natm;
int   *nchunk;
int    p2c[];
int    holist[];
float  x;
float  y;
float  z;
float  vvx;
float  vvy;
float  vvz;
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
   int boff, pid, nsend, bptr;

   int cperb  = iparam[18];
   int pperc  = iparam[19];
   int chnksz = iparam[20];

   boff  = (cid-1)*chnksz + 1;
   pid   = (int)(bigbuf[boff-1] + 0.1);
   nsend = (int)(bigbuf[boff] + 0.1);
   bptr  = 6*nsend + boff + 2;
   bigbuf[bptr-1] = x;
   bigbuf[bptr]   = y;
   bigbuf[bptr+1] = z;
   bigbuf[bptr+2] = vvx;
   bigbuf[bptr+3] = vvy;
   bigbuf[bptr+4] = vvz;
   nsend++;
   bigbuf[boff] = (float)nsend;
   if (nsend == pperc) {
      if(pid==RANKHOST) 
         tohost (&bigbuf[boff+1], nsend, natm, rx, ry, rz, vx, vy, vz, 
                 xor, yor, zor);
#ifndef SEQ
      else
         MPI_Send (&bigbuf[boff], 6*nsend+1, MPI_FLOAT, pid, 777, 
                   MPI_COMM_WORLD);
#endif
      bigbuf[boff-1] = -1.0;
      bigbuf[boff]   =  0.0;
      p2c[pid]       = -1;
      *nchunk = *nchunk - 1;
      holist[cperb-*nchunk-1] = cid;
   }
}
