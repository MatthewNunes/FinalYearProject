/*
  file: edgadd-worker.f

  The routine edgadd takes particles out of a communication buffer and
  stores them as edge particles in a process.

    NEDGE    An integer giving the index of the last location in the
             particle array into which an edge particle was stored.

    NPASS    An integer array of length 2. The first part of the buffer
             pass contains migrating particles, and the second part
             contains edge particles. npass(1) is the number of reals
             used to pass the migrating particles' information. npass(2)
             is the total number of reals used to pass both migrating and
             edge particles.

    PASS     A real array containing the buffer to be inserted as edge 
             particles.
*/

void edgadd (nedge, npass, pass, rx, ry, rz, list, head, iparam, fparam)
int *nedge;
int npass[2];
float *pass;
float rx[];
float ry[];
float rz[];
int head[];
int list[];
int *iparam;
float *fparam;
{
   int i, icell;

   int mx = iparam[13];
   int my = iparam[14];
   int mz = iparam[15];

   float sfx = fparam[9];
   float sfy = fparam[10];
   float sfz = fparam[11];
   float cellix = fparam[12];
   float celliy = fparam[13];
   float celliz = fparam[14];

   for(i=npass[0]+1;i<=npass[1];i+=3){
      *nedge -= 1;
      rx[*nedge-1] = pass[i-1];
      ry[*nedge-1] = pass[i];
      rz[*nedge-1] = pass[i+1];
      icell       = 1 + (int)((rx[*nedge-1]*sfx+0.5)*cellix) +
                        (int)((ry[*nedge-1]*sfy+0.5)*celliy)*mx +
                        (int)((rz[*nedge-1]*sfz+0.5)*celliz)*mx*my;
      list[*nedge-1] = head[icell-1];
      head[icell-1] = *nedge;
   }
}
