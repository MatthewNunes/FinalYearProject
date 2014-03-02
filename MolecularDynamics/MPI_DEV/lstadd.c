/*
  file: lstadd-worker.f

  The routine lstadd adds the particles in the buffer pass to the local
  particle data structures as regular particles.

  NPASS    An integer giving the number of reals taken up by the particles.
           This is just 6 times the number of particles since both position
           and velocity components in 3 dimensions must be specified.

  PASS     A real array containing the particle information to be inserted
           into the local particle data structures. This array is a buffer
           containing particles that have migrated from a neighboring
           process.
*/

void lstadd (npass, pass, rx, ry, rz, vx, vy, vz, list, head, natm,iparam,fparam)
int npass;
float pass[];
float rx[];
float ry[];
float rz[];
float vx[];
float vy[];
float vz[];
int list[];
int head[];
int *natm;
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
   
   for(i=1;i<=npass;i+=6){
      *natm += 1;
      rx[*natm-1] = pass[i-1];
      ry[*natm-1] = pass[i];
      rz[*natm-1] = pass[i+1];
      vx[*natm-1] = pass[i+2];
      vy[*natm-1] = pass[i+3];
      vz[*natm-1] = pass[i+4];
      icell = 1 + (int)((rx[*natm-1]*sfx+0.5)*cellix) +
                  (int)((ry[*natm-1]*sfy+0.5)*celliy)*mx +
                  (int)((rz[*natm-1]*sfz+0.5)*celliz)*mx*my;
      list[*natm-1]  = head[icell-1];
      head[icell-1] = *natm;
   }
}


