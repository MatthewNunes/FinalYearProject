#ifndef SEQ
#include "mpi.h"
#endif

#include <stdio.h>
#include "moldyn.h"

void initialise_particles (nchunk, natm, holist, p2c, 
                           rx, ry, rz, vx, vy, vz, xor, yor, zor,iparam,fparam,bigbuf)
int  *nchunk;
int  *natm;
int   holist[];
int   p2c[];
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
   int is = 2215;
   int i, lz, ly, lx;
   int nz1, nz2, nz3, nz4, ny1, ny2, ny3, ny4, nx;
   int cid, pid, nsum;
   float rz1, rz2, rz3, rz4, ry1, ry2, ry3, ry4, rx1, vvx, vvy, vvz;

   int nprosx = iparam[0];
   int nprosy = iparam[1];
   int nprosz = iparam[2];
   int nc     = iparam[10];
   int cperb  = iparam[18];
   int chnksz = iparam[20];
   
   float boxlx = fparam[6];
   float boxly = fparam[7];
   float boxlz = fparam[8];
   float cfact = fparam[31];

   for (lz=0;lz<nc;lz++){
      rz1 = -0.5 + (float)lz/(float)nc + 0.000001;
      rz2 = -0.5 + ((float)lz+0.5)/(float)nc - 0.000001;
      rz3 = -0.5 + (float)lz/(float)nc + 0.000001;
      rz4 = -0.5 + ((float)lz+0.5)/(float)nc - 0.000001;
      nz1 = nprosy*(int)((rz1+0.5)/boxlz);
      nz2 = nprosy*(int)((rz2+0.5)/boxlz);
      nz3 = nprosy*(int)((rz3+0.5)/boxlz);
      nz4 = nprosy*(int)((rz4+0.5)/boxlz);
      for(ly=0;ly<nc;ly++){
         ry1 = -0.5 + (float)ly/(float)nc + 0.000001;
         ry2 = -0.5 + (float)ly/(float)nc + 0.000001;
         ry3 = -0.5 + ((float)ly+0.5)/(float)nc - 0.000001;
         ry4 = -0.5 + ((float)ly+0.5)/(float)nc - 0.000001;
         ny1 = nprosx*((int)((ry1+0.5)/boxly)+nz1);
         ny2 = nprosx*((int)((ry2+0.5)/boxly)+nz2);
         ny3 = nprosx*((int)((ry3+0.5)/boxly)+nz3);
         ny4 = nprosx*((int)((ry4+0.5)/boxly)+nz4);
         for(lx=0;lx<nc;lx++){
            rx1 = -0.5 + (float)lx/(float)nc + 0.000001;
            nx = 1 + (int)((rx1+0.5)/boxlx);
            pseudorand (&is, &vvx);
            pseudorand (&is, &vvy);
            pseudorand (&is, &vvz);
            pid = nx + ny1;
            cid = p2c[pid-1];
            if (cid<=0) {
               if (*nchunk==cperb) flush_buffer (bigbuf, cfact, nchunk, p2c, 
                                   holist, natm, rx, ry, rz, vx, vy, vz, 
                                   xor, yor, zor, iparam);
               cid = holist[cperb-*nchunk-1];
               *nchunk = *nchunk + 1;
               bigbuf[(cid-1)*chnksz] = (float)(pid-1);
               p2c[pid-1] = cid;
            }
            pack_buffer (bigbuf, cid, natm, nchunk, p2c, holist, rx1, ry1, rz1, 
                         vvx, vvy, vvz, rx, ry, rz, vx, vy, vz, xor, yor, zor, iparam);
            rx1 = -0.5 + ((float)lx+0.5)/(float)nc - 0.000001;
            nx = 1 + (int)((rx1+0.5)/boxlx);
            pid = nx + ny2;
            cid = p2c[pid-1];
            if (cid<=0) {
               if (*nchunk==cperb) flush_buffer (bigbuf, cfact, nchunk, p2c, 
                                       holist, natm, rx, ry, rz, vx, vy, vz, 
                                       xor, yor, zor, iparam);
               cid = holist[cperb-*nchunk-1];
               *nchunk = *nchunk + 1;
               bigbuf[(cid-1)*chnksz] = (float)(pid-1);
               p2c[pid-1] = cid;
            }
            pack_buffer (bigbuf, cid, natm, nchunk, p2c, holist, rx1, ry2, rz2, 
                       -vvx, -vvy, -vvz, rx, ry, rz, vx, vy, vz, xor, yor, zor, iparam);
            rx1 = -0.5 + ((float)lx+0.5)/(float)nc - 0.000001;
            pseudorand (&is, &vvx);
            pseudorand (&is, &vvy);
            pseudorand (&is, &vvz);
            nx = 1 + (int)((rx1+0.5)/boxlx);
            pid = nx + ny3;
            cid = p2c[pid-1];
            if (cid<=0) {
               if (*nchunk==cperb) flush_buffer (bigbuf, cfact, nchunk, p2c, 
                                   holist, natm, rx, ry, rz, vx, vy, vz, 
                                   xor, yor, zor, iparam);
               cid = holist[cperb-*nchunk-1];
               *nchunk = *nchunk + 1;
               bigbuf[(cid-1)*chnksz] = (float)(pid-1);
               p2c[pid-1] = cid;
            }
            pack_buffer (bigbuf, cid, natm, nchunk, p2c, holist, rx1, ry3, rz3, 
                         vvx, vvy, vvz, rx, ry, rz, vx, vy, vz, xor, yor, zor, iparam);
            rx1 = -0.5 + (float)lx/(float)nc + 0.000001;
            nx = 1 + (int)((rx1+0.5)/boxlx);
            pid = nx + ny4;
            cid = p2c[pid-1];
            if (cid<=0) {
               if (*nchunk==cperb) flush_buffer (bigbuf, cfact, nchunk, p2c, 
                                   holist, natm, rx, ry, rz, vx, vy, vz, 
                                   xor, yor, zor, iparam);
               cid = holist[cperb-*nchunk-1];
               *nchunk = *nchunk + 1;
               bigbuf[(cid-1)*chnksz] = (float)(pid-1);
               p2c[pid-1] = cid;
            }
            pack_buffer (bigbuf, cid, natm, nchunk, p2c, holist, rx1, ry4, rz4, 
                       -vvx, -vvy, -vvz, rx, ry, rz, vx, vy, vz, xor, yor, zor, iparam);
         }
      }
   }
   flush_buffer (bigbuf, 1.0, nchunk, 
                 p2c, holist, natm, rx, ry, rz, vx, vy, vz, xor, yor, zor, iparam);

#ifndef SEQ
   for(i=1;i<=nprosx*nprosy*nprosz-1;i++)
      MPI_Send (&nsum, 1, MPI_INT, i, 555, MPI_COMM_WORLD);
#endif

   nsum = *natm;
#ifndef SEQ
   MPI_Allreduce (natm, &nsum, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
#endif
   if (nsum!=4*nc*nc*nc) 
      printf ("\n Workers have %8d particles; host has %8d\n",nsum, 4*nc*nc*nc);
}
