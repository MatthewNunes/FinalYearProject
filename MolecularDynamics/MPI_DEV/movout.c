#ifndef SEQ
#include "mpi.h"
#endif

#include "moldyn.h"
#include <stdio.h>

#define kcell(A,B,C) ( (((A)+nprosx)%nprosx)+((((B)+nprosy)%nprosy) + (((C)+nprosz)%nprosz)*nprosy)*nprosx )

void movout (rx, ry, rz, vx, vy, vz, head, list, xid, yid, zid, natm,nsize,scratch,ptemp1,ptemp2,natoms,iparam,fparam)
float rx[];
float ry[];
float rz[];
float vx[];
float vy[];
float vz[];
int head[];
int list[];
int xid;
int yid;
int zid;
int *natm;
int nsize;
float *scratch;
float *ptemp1;
float *ptemp2;
int natoms;
int *iparam;
float *fparam;
{
   float *passn=&scratch[0];
   float *passs=&scratch[natoms];
   float *passe=&scratch[2*natoms];
   float *passw=&scratch[3*natoms];
   float *passu=&scratch[4*natoms];
   float *passd=&scratch[5*natoms];
   int    npassn[2], npasss[2], npasse[2], npassw[2];
   int    npassu[2], npassd[2], npass1[2], npass2[2];
   int    ntemp1[2], ntemp2[2];
   int    i, j, k, icell, kcell, nedge;
   int    ix, iy, iz, xi, yi, zi;
   int    north, south, east, west, up, down;

   int nprosx = iparam[0];
   int nprosy = iparam[1];
   int nprosz = iparam[2];
   int mx = iparam[13];
   int my = iparam[14];
   int mz = iparam[15];

   float boxly = fparam[7];
   float boxlz = fparam[8];
   float sfx   = fparam[9];
   float sfy   = fparam[10];
   float sfz   = fparam[11];
   float cellix   = fparam[12];
   float celliy   = fparam[13];
   float celliz   = fparam[14];

#ifndef SEQ
   MPI_Status istatus;
   MPI_Request msgn1, msgp1, msgn2, msgp2;
#endif

#ifndef SEQ
   MPI_Irecv (ntemp1, 2, MPI_INT, MPI_ANY_SOURCE, 1410, MPI_COMM_WORLD, &msgn1);
   MPI_Irecv (ptemp1, natoms, MPI_FLOAT, MPI_ANY_SOURCE, 1420, 
              MPI_COMM_WORLD, &msgp1);
#endif

   north = kcell (xid, yid+1, zid);
   south = kcell (xid, yid-1, zid);
   east  = kcell (xid+1, yid, zid);
   west  = kcell (xid-1, yid, zid);
   up    = kcell (xid, yid, zid+1);
   down  = kcell (xid, yid, zid-1);

   npassn[0] = 0;
   npasss[0] = 0;
   npasse[0] = 0;
   npassw[0] = 0;
   npassu[0] = 0;
   npassd[0] = 0;
   npassn[1] = 0;
   npasss[1] = 0;
   npasse[1] = 0;
   npassw[1] = 0;
   npassu[1] = 0;
   npassd[1] = 0;

   for (icell=0;icell<mx*my*mz;++icell) head[icell] = 0;

   nedge = nsize;
   i = 0;

   while (i < *natm) {
      i += 1;
      xi = 1 + (int)(cellix*(rx[i-1]*sfx+0.5));
      yi = 1 + (int)(celliy*(ry[i-1]*sfy+0.5));
      zi = 1 + (int)(celliz*(rz[i-1]*sfz+0.5));
      icell = xi + (yi-1)*mx + (zi-1)*mx*my;


      if (xi==mx) {                         
         migrate_buffer (1, i, icell, &nedge, passe, &npasse[0], 
                         rx, ry, rz, vx, vy, vz, list, head, natm, fparam);
         i -= 1;
      }
      else if (xi==1) {                     
         migrate_buffer (2, i, icell, &nedge, passw, &npassw[0],
                         rx, ry, rz, vx, vy, vz, list, head, natm, fparam);
         i -= 1;
      }
      else if (yi==my) {                    
         migrate_buffer (3, i, icell, &nedge, passn, &npassn[0],
                         rx, ry, rz, vx, vy, vz, list, head, natm, fparam);
         i -= 1;
      }
      else if (yi==1) {                     
         migrate_buffer (4, i, icell, &nedge, passs, &npasss[0],
                         rx, ry, rz, vx, vy, vz, list, head, natm, fparam);
         i -= 1;
      }
      else if (zi==mz) {                    
         migrate_buffer (5, i, icell, &nedge, passu, &npassu[0],
                         rx, ry, rz, vx, vy, vz, list, head, natm, fparam);
         i -= 1;
      }
      else if (zi==1) {                     
         migrate_buffer (6, i, icell, &nedge, passd, &npassd[0],
                         rx, ry, rz, vx, vy, vz, list, head, natm, fparam);
         i -= 1;
      }
      else {                                
         list[i-1]      = head[icell-1];
         head [icell-1] = i;
      }
   }

   npasse[1] = npasse[0];

   edgbuf (1, mx-1, mx-1, 1, my, 1, mz, &npasse[1], passe, 
           rx, ry, rz, list, head, iparam, fparam);

#ifndef SEQ
   MPI_Send (npasse, 2, MPI_INT, east, 1410, MPI_COMM_WORLD);
   MPI_Send (passe, npasse[1], MPI_FLOAT, east, 1420, MPI_COMM_WORLD);

   MPI_Irecv (ntemp2, 2, MPI_INT, MPI_ANY_SOURCE, 1430, MPI_COMM_WORLD, &msgn2);
   MPI_Irecv (ptemp2, natoms, MPI_FLOAT, MPI_ANY_SOURCE, 1440, 
              MPI_COMM_WORLD, &msgp2);
#endif

   npassw[1] = npassw[0];

   edgbuf (2, 2, 2, 1, my, 1, mz, &npassw[1], passw, rx, ry, rz, list, head, iparam, fparam);

#ifndef SEQ
   MPI_Send (npassw, 2, MPI_INT, west, 1430, MPI_COMM_WORLD);
   MPI_Send (passw, npassw[1], MPI_FLOAT, west, 1440, MPI_COMM_WORLD);

   MPI_Wait (&msgn1, &istatus);
   MPI_Wait (&msgp1, &istatus);
#else
   ntemp1[0] = npasse[0];
   ntemp1[1] = npasse[1];
   for (i=0;i<npasse[1];i++) ptemp1[i] = passe[i];
#endif


   for (i=1;i<=ntemp1[0];i+=6){
      zi = 1 + (int)(celliz*(ptemp1[i+1]*sfz+0.5));  
      yi = 1 + (int)(celliy*(ptemp1[i]*sfy+0.5));  


      if (yi==my) {
         icell = 2 + (my-1)*mx + (zi-1)*mx*my;
         pasbuf (i, icell, &nedge, npassn, passn, ptemp1, 0.0, -boxly, 0.0,
                 rx, ry, rz, list, head);
      }
      else if (yi==1) {
         icell = 2 + (zi-1)*mx*my;
         pasbuf (i, icell, &nedge, npasss, passs, ptemp1, 0.0, boxly, 0.0,
                 rx, ry, rz, list, head);
      }
      else if (zi==mz) {
         icell = 2 + (yi-1)*mx + (mz-1)*mx*my;
         pasbuf (i, icell, &nedge, npassu, passu, ptemp1, 0.0, 0.0, -boxlz,
                 rx, ry, rz, list, head);
      }
      else if (zi==1) {
         icell = 2 + (yi-1)*mx;
         pasbuf (i, icell, &nedge, npassd, passd, ptemp1, 0.0, 0.0, boxlz,
                 rx, ry, rz, list, head);
      }
      else lstadd (1, &ptemp1[i-1], rx, ry, rz, vx, vy, vz, list, head, natm,iparam, fparam);
   }

   edgadd (&nedge, ntemp1, ptemp1, rx, ry, rz, list, head, iparam, fparam);

#ifndef SEQ
   MPI_Irecv (ntemp1, 2, MPI_INT, MPI_ANY_SOURCE, 1450, MPI_COMM_WORLD, &msgn1);
   MPI_Irecv (ptemp1, natoms, MPI_FLOAT, MPI_ANY_SOURCE, 1460, 
              MPI_COMM_WORLD, &msgp1);

   MPI_Wait (&msgn2, &istatus);
   MPI_Wait (&msgp2, &istatus);
#else
   ntemp2[0] = npassw[0];
   ntemp2[1] = npassw[1];
   for (i=0;i<npassw[1];i++) ptemp2[i] = passw[i];
#endif

   for (i=1;i<=ntemp2[0];i+=6) {
      zi = 1 + (int)(celliz*(ptemp2[i+1]*sfz+0.5));
      yi = 1 + (int)(celliy*(ptemp2[i]*sfy+0.5));


      if (yi==my) {
         icell = my*mx - 1 + (zi-1)*mx*my;
         pasbuf (i, icell, &nedge, npassn, passn, ptemp2, 0.0, -boxly, 0.0,
                 rx, ry, rz, list, head);
      }
      else if (yi==1) {
         icell = mx - 1 + mx*my*(zi-1);
         pasbuf (i, icell, &nedge, npasss, passs, ptemp2, 0.0, boxly, 0.0,
                 rx, ry, rz, list, head);
      }
      else if (zi==mz) {
         icell = mx - 1 + (yi-1)*mx + (mz-1)*mx*my;
         pasbuf (i, icell, &nedge, npassu, passu, ptemp2, 0.0, 0.0, -boxlz,
                 rx, ry, rz, list, head);
      }
      else if (zi==1) {
         icell = mx - 1 + (yi-1)*mx;
         pasbuf (i, icell, &nedge, npassd, passd, ptemp2, 0.0, 0.0, boxlz,
                 rx, ry, rz, list, head);
      }
      else lstadd (1, &ptemp2[i-1], rx, ry, rz, vx, vy, vz, list, head, natm,iparam, fparam);
   }


   edgadd (&nedge, ntemp2, ptemp2, rx, ry, rz, list, head, iparam, fparam);

#ifndef SEQ
   MPI_Irecv (ntemp2, 2, MPI_INT, MPI_ANY_SOURCE, 1470, MPI_COMM_WORLD, &msgn2);
   MPI_Irecv (ptemp2, natoms, MPI_FLOAT, MPI_ANY_SOURCE, 1480, 
              MPI_COMM_WORLD, &msgp2);
#endif

   npassn[1] = npassn[0];

   edgbuf (3,  1,  1,   my,   my, 1, mz, &npassn[1], passn, 
           rx, ry, rz, list, head, iparam, fparam);
   edgbuf (3, mx, mx,   my,   my, 1, mz, &npassn[1], passn, 
           rx, ry, rz, list, head, iparam, fparam);
   /* edgprint (1, mx, my-1, my-1, 1, mz, rx, ry, rz, list, head,iparam); */
   edgbuf (3,  1, mx, my-1, my-1, 1, mz, &npassn[1], passn, 
           rx, ry, rz, list, head, iparam, fparam);

#ifndef SEQ
   MPI_Send (npassn, 2, MPI_INT, north, 1450, MPI_COMM_WORLD);
   MPI_Send (passn, npassn[1], MPI_FLOAT, north, 1460, MPI_COMM_WORLD);
#endif

   npasss[1] = npasss[0];

   edgbuf (4,  1,  1, 1, 1, 1, mz, &npasss[1], passs, rx, ry, rz, list, head, iparam, fparam);
   edgbuf (4, mx, mx, 1, 1, 1, mz, &npasss[1], passs, rx, ry, rz, list, head, iparam, fparam);
   edgbuf (4,  1, mx, 2, 2, 1, mz, &npasss[1], passs, rx, ry, rz, list, head, iparam, fparam);

#ifndef SEQ
   MPI_Send (npasss, 2, MPI_INT, south, 1470, MPI_COMM_WORLD);
   MPI_Send (passs, npasss[1], MPI_FLOAT, south, 1480, MPI_COMM_WORLD);

   MPI_Wait (&msgn1, &istatus);
   MPI_Wait (&msgp1, &istatus);
#else
   ntemp1[0] = npassn[0];
   ntemp1[1] = npassn[1];
   for (i=0;i<npassn[1];i++) ptemp1[i] = passn[i];
#endif

   for (i=1;i<=ntemp1[0];i+=6) {
      xi = 1 + (int)(cellix*(ptemp1[i-1]*sfx+0.5));
      zi = 1 + (int)(celliz*(ptemp1[i+1]*sfz+0.5));


      if (zi==mz) {
         icell = xi + mx + (mz-1)*mx*my;
         pasbuf (i, icell, &nedge, npassu, passu, ptemp1, 0.0, 0.0, -boxlz,
                 rx, ry, rz, list, head);
      }
      else if (zi==1) {
         icell = xi + mx;
         pasbuf (i, icell, &nedge, npassd, passd, ptemp1, 0.0, 0.0, boxlz,
                 rx, ry, rz, list, head);
      }
      else lstadd (1, &ptemp1[i-1], rx, ry, rz, vx, vy, vz, list, head, natm,iparam, fparam);
   }

   edgadd (&nedge, ntemp1, ptemp1, rx, ry, rz, list, head, iparam, fparam);

#ifndef SEQ
   MPI_Irecv (ntemp1, 2, MPI_INT, MPI_ANY_SOURCE, 1490, MPI_COMM_WORLD, &msgn1);
   MPI_Irecv (ptemp1, natoms, MPI_FLOAT, MPI_ANY_SOURCE, 1500, 
              MPI_COMM_WORLD, &msgp1);

   MPI_Wait (&msgn2, &istatus);
   MPI_Wait (&msgp2, &istatus);
#else
   ntemp2[0] = npasss[0];
   ntemp2[1] = npasss[1];
   for (i=0;i<npasss[1];i++) ptemp2[i] = passs[i];
#endif

   for (i=1;i<=ntemp2[0];i+=6) {
      xi = 1 + (int)(cellix*(ptemp2[i-1]*sfx+0.5));
      zi = 1 + (int)(celliz*(ptemp2[i+1]*sfz+0.5));


      if (zi==mz) {
         icell = xi + mx*(my-2) + (mz-1)*mx*my;
         pasbuf (i, icell, &nedge, npassu, passu, ptemp2, 0.0, 0.0, -boxlz,
                 rx, ry, rz, list, head);
      }
      else if (zi==1) {
         icell = xi + mx*(my-2);
         pasbuf (i, icell, &nedge, npassd, passd, ptemp2, 0.0, 0.0, boxlz,
                 rx, ry, rz, list, head);
      }
      else lstadd (1, &ptemp2[i-1], rx, ry, rz, vx, vy, vz, list, head, natm,iparam, fparam);
   }

   edgadd (&nedge, ntemp2, ptemp2, rx, ry, rz, list, head, iparam, fparam);

#ifndef SEQ
   MPI_Irecv (ntemp2, 2, MPI_INT, MPI_ANY_SOURCE, 1510, MPI_COMM_WORLD, &msgn2);
   MPI_Irecv (ptemp2, natoms, MPI_FLOAT, MPI_ANY_SOURCE, 1520,
              MPI_COMM_WORLD, &msgp2);
#endif

   npassu[1] = npassu[0];

   edgbuf (5,  1, mx,  1,    1,   mz,   mz, &npassu[1], passu, 
           rx, ry, rz, list, head, iparam, fparam);
   edgbuf (5,  1, mx, my,   my,   mz,   mz, &npassu[1], passu, 
           rx, ry, rz, list, head, iparam, fparam);
   edgbuf (5,  1,  1,  2, my-1,   mz,   mz, &npassu[1], passu, 
           rx, ry, rz, list, head, iparam, fparam);
   edgbuf (5, mx, mx,  2, my-1,   mz,   mz, &npassu[1], passu, 
           rx, ry, rz, list, head, iparam, fparam);
   edgbuf (5,  1, mx,  1,   my, mz-1, mz-1, &npassu[1], passu, 
           rx, ry, rz, list, head, iparam, fparam);

#ifndef SEQ
   MPI_Send (npassu, 2, MPI_INT, up, 1490, MPI_COMM_WORLD);
   MPI_Send (passu, npassu[1], MPI_FLOAT, up, 1500, MPI_COMM_WORLD);
#endif

   npassd[1] = npassd[0];

   edgbuf (6,  1, mx,  1,    1, 1, 1, &npassd[1], passd, rx, ry, rz, list, head, iparam, fparam);
   edgbuf (6,  1, mx, my,   my, 1, 1, &npassd[1], passd, rx, ry, rz, list, head, iparam, fparam);
   edgbuf (6,  1,  1,  2, my-1, 1, 1, &npassd[1], passd, rx, ry, rz, list, head, iparam, fparam);
   edgbuf (6, mx, mx,  2, my-1, 1, 1, &npassd[1], passd, rx, ry, rz, list, head, iparam, fparam);
   edgbuf (6,  1, mx,  1,   my, 2, 2, &npassd[1], passd, rx, ry, rz, list, head, iparam, fparam);

#ifndef SEQ
   MPI_Send (npassd, 2, MPI_INT, down, 1510, MPI_COMM_WORLD);
   MPI_Send (passd, npassd[1], MPI_FLOAT, down, 1520, MPI_COMM_WORLD);

   MPI_Wait (&msgn1, &istatus);
   MPI_Wait (&msgp1, &istatus);
#else
   ntemp1[0] = npassu[0];
   ntemp1[1] = npassu[1];
   for (i=0;i<npassu[1];i++) ptemp1[i] = passu[i];
#endif

   lstadd (ntemp1[0], ptemp1, rx, ry, rz, vx, vy, vz, list, head, natm,iparam, fparam);

   edgadd (&nedge, ntemp1, ptemp1, rx, ry, rz, list, head, iparam, fparam);

#ifndef SEQ
   MPI_Wait (&msgn2, &istatus);
   MPI_Wait (&msgp2, &istatus);
#else
   ntemp2[0] = npassd[0];
   ntemp2[1] = npassd[1];
   for (i=0;i<npassd[1];i++) ptemp2[i] = passd[i];
#endif

   lstadd (ntemp2[0], ptemp2, rx, ry, rz, vx, vy, vz, list, head, natm,iparam, fparam);

   edgadd (&nedge, ntemp2, ptemp2, rx, ry, rz, list, head, iparam, fparam);


   if (nedge<*natm) printf("\n linked list recursive ");

}









