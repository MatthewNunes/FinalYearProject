/*
  file: pasbuf-worker.f

  The routine pasbuf takes a particle in the buffer passi, and makes it
  an edge particle and also puts it into the buffer passo to be passed
  on to another process.

    ICELL    An integer giving the cell number of the particle.

    NEDGE    An integer giving the index into the particle array of
             the edge particle most recently placed in the array.

    NPASS    An integer giving the location in the buffer passo at which
             the particle data are to be stored.

    PASSO    A real array containing the buffer written to. This buffer
             will be passed to another process in a subsequent
             communication phase.

    PASSI    A real array containing the input buffer. This is a buffer
             that has been received from another processs.

    ADDX     A real giving the amount added to the x-coordinate when
             transforming to the local coordinates of the destination
             process. Either 0, -boxlx, or +boxlx.

    ADDY     A real giving the amount added to the y-coordinate when
             transforming to the local coordinates of the destination
             process. Either 0, -boxly, or +boxly.

    ADDZ     A real giving the amount added to the z-coordinate when
             transforming to the local coordinates of the destination
             process. Either 0, -boxlz, or +boxlz.
*/

void pasbuf(i, icell, nedge, npass, passo, passi, addx, addy, addz,
            rx, ry, rz, list, head)
int i;
int icell;
int *nedge;
int npass[2];
float passo[];
float passi[];
float addx;
float addy;
float addz;
float rx[];
float ry[];
float rz[];
int list[];
int head[];
{
   *nedge -= 1;
   rx[*nedge-1] = passi[i-1];
   ry[*nedge-1] = passi[i];
   rz[*nedge-1] = passi[i+1];
   list[*nedge-1] = head[icell-1];
   head[icell-1] = *nedge;

   npass[0] += 6;
   passo[npass[0]-6] = passi[i-1] + addx;
   passo[npass[0]-5] = passi[i]   + addy;
   passo[npass[0]-4] = passi[i+1] + addz;
   passo[npass[0]-3] = passi[i+2];
   passo[npass[0]-2] = passi[i+3];
   passo[npass[0]-1] = passi[i+4];
}
