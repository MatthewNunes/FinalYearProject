#include <stdio.h>
#include "moldyn.h"

void set_chunk (wsize, ibuf, nchunk, holist, p2c,iparam,fparam,bigbuf)
int    wsize;
int    ibuf[];
int   *nchunk;
int    holist[];
int    p2c[];
int *iparam;
float *fparam;
float *bigbuf;
{
   int chnksz, cperb, pperc;
   int istart, where;
   FILE *input_file;
   int i;
   float cfact;

   int nprosx = iparam[0];
   int nprosy = iparam[1];
   int nprosz = iparam[2];
   int nstart = iparam[7];
   int natoms = iparam[11];

   float ace   = fparam[23];
   float acv   = fparam[24];
   float ack   = fparam[25];
   float acp   = fparam[26];
   float acesq = fparam[27];
   float acvsq = fparam[28];
   float acksq = fparam[29];
   float acpsq = fparam[30];

   printf ("\n PLEASE GIVE CHUNK SIZE");
   int chk = scanf  ("%d", &chnksz);
   printf ("\n PLEASE GIVE CLEAR-OUT FACTOR (1.0=ALL)");
   chk = scanf  ("%f", &cfact);
   cfact = min (cfact, 1.0);
   cfact = max (cfact, 0.0);
   fparam[31] = cfact;

   chnksz = min (chnksz, wsize/4+1);
   chnksz = min (chnksz, MYBUFSIZ);
 
   cperb  = min (MYBUFSIZ/chnksz, nprosx*nprosy*nprosz);
   pperc  = (chnksz-2)/6;

   for(i=1;i<=cperb;++i){
      holist[i-1] = i;
      bigbuf[(i-1)*chnksz]   = -1.0;
      bigbuf[(i-1)*chnksz+1] =  0.0;
      p2c[i-1] =  -1;
   }

   *nchunk = 0;

   printf ("\n Chunk size          = %6d", chnksz);
   printf ("\n Chunks in buffer    = %6d", cperb);
   printf ("\n Particles per chunk = %6d", pperc);

   iparam[18] = cperb;
   iparam[19] = pperc;
   iparam[20] = chnksz;

   if (nstart == 3) {
      ibuf[0] = -1;
      ibuf[1] =  1;
      where   = 1;
   }
   else if (nstart == 5) {
      input_file = fopen ("checkpointseq.for", "r");
      chk = fscanf (input_file,"%8d%8d%8d",&natoms, &istart, &where);
      chk = fscanf (input_file,"%18e%18e%18e%18e%18e%18e%18e%18e", 
            &ace, &acv, &ack,&acp, &acesq, &acvsq, &acksq, &acpsq);
      ibuf[0] = istart;
      ibuf[1] = where;
   }
   else if (nstart == 6) {
      printf ("\n Sorry - unformatted warm starts not yet implemented");
   }
}
