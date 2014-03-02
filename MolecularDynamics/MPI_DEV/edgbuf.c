/*
  file: edgbuf-worker.f

  This routine packs edge particles from a specified 3D rectangle of
  cells into a communication buffer.

    DIR       integer giving the direction in which edge particles
              are to be moved.
                   1 = east
                   2 = west
                   3 = north
                   4 = south
                   5 = up
                   6 = down

    ILO        lower coordinate of 3D block of cells in the first
               coordinate direction (east-west).

    IHI        upper coordinate of 3D block of cells in the first
               coordinate direction (east-west).

    JLO        lower coordinate of 3D block of cells in the second
               coordinate direction (north-south).

    JHI        upper coordinate of 3D block of cells in the second
               coordinate direction (north-south).

    KLO        lower coordinate of 3D block of cells in the third
               coordinate direction (up-down).

    KHI        upper coordinate of 3D block of cells in the third
               coordinate direction (up-down).

    NPASS      the number of reals placed into the communication 
               buffer.

    PASS       the communication buffer.
*/
void edgbuf (dir, ilo, ihi, jlo, jhi, klo, khi, npass, pass,
             rx, ry, rz, list, head,iparam, fparam)
int dir;
int ilo;
int ihi;
int jlo;
int jhi;
int klo;
int khi;
int *npass;
float *pass;
float rx[];
float ry[];
float rz[];
int head[];
int list[];
int *iparam;
float *fparam;
{
   int i, j, k, indx, ipart, kadd, jadd;
   float addx, addy, addz;

   int mx = iparam[13];
   int my = iparam[14];
   int mz = iparam[15];

   float boxlx = fparam[6];
   float boxly = fparam[7];
   float boxlz = fparam[8];

/* Figure out what we need to add to  the coordinates to transform to
   target process coordinate system. */

   addx = 0.0;
   addy = 0.0;
   addz = 0.0;
   switch (dir) {
      case 1:
          addx = -boxlx;
          break;
      case 2:
          addx =  boxlx;
          break;
      case 3:
          addy = -boxly;
          break;
      case 4:
          addy =  boxly;
          break;
      case 5:
          addz = -boxlz;
          break;
      case 6:
          addz =  boxlz;
          break;
   }

/* Loop over the 3D block of cells, and extract particles from each 
   cell by following linked list.  */
   for(k=klo;k<=khi;k++){
      kadd = (k-1)*my;
      for(j=jlo;j<=jhi;j++){
         jadd = (j-1+kadd)*mx;
         for(i=ilo;i<=ihi;i++){
            indx  = i + jadd;
            ipart = head[indx-1];
            while (ipart != 0){
               *npass += 3;
               pass[*npass-3] = rx[ipart-1] + addx;
               pass[*npass-2] = ry[ipart-1] + addy;
               pass[*npass-1] = rz[ipart-1] + addz;
               ipart = list[ipart-1];
            }
         }
      }
   }
}
