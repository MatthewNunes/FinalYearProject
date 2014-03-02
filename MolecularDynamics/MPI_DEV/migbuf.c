/*
  file: migbuf-worker.f

  The subroutine migbuf does buffering needed to migrate particles
  across a process boundary in a given direction. The migrating
  particles are put into a communication buffer. In addition, they
  are removed from the local particle array, and stored as edge
  particles since in the next phase they will be boundary particles
  in the process to which they are sent.

    DIR       integer giving the direction in which particles are
              to be migrated.
                   1 = east
                   2 = west
                   3 = north
                   4 = south
                   5 = up
                   6 = down

    I         index in particle array of particle to be migrated

    ICELL     index of cell that particle I is in

    NEDGE     integer giving the number of edge particles

    PASS      real array used to buffer information for particles
              to be communicated

    NPASS     integer giving the number of reals in the PASS
              array
*/

void migrate_buffer (dir, i, icell, nedge, pass, npass, 
                     rx, ry, rz, vx, vy, vz, list, head, natm, fparam)
int dir;
int i;
int icell;
int *nedge;
int *npass;
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
float *fparam;
{
       float boxlx = fparam[6];
       float boxly = fparam[7];
       float boxlz = fparam[8];

/* We are going to put another 6 reals into the PASS array. */
       *npass += 6;

/* Place position and velocity into PASS array. */
       pass[*npass-6] = rx[i-1];
       pass[*npass-5] = ry[i-1];
       pass[*npass-4] = rz[i-1];
       pass[*npass-3] = vx[i-1];
       pass[*npass-2] = vy[i-1];
       pass[*npass-1] = vz[i-1];

/* Shift coordinates to those of the process to which particle is
   migrating. */
       switch (dir) {
          case 1:
	     pass[*npass-6] = rx[i-1] - boxlx;
             break;
          case 2:
	     pass[*npass-6] = rx[i-1] + boxlx;
             break;
          case 3:
	     pass[*npass-5] = ry[i-1] - boxly;
             break;
          case 4:
	     pass[*npass-5] = ry[i-1] + boxly;
             break;
          case 5:
	     pass[*npass-4] = rz[i-1] - boxlz;
             break;
          case 6:
	     pass[*npass-4] = rz[i-1] + boxlz;
             break;
       }

/* Add to linked list as an edge particle. */
       *nedge = *nedge - 1;
       rx[*nedge-1] = rx[i-1];
       ry[*nedge-1] = ry[i-1];
       rz[*nedge-1] = rz[i-1];
       list[*nedge-1] = head[icell-1];
       head[icell-1] = *nedge;

/* Fill empty space of migrated particle with real particle. */
       rx[i-1] = rx[*natm-1];
       ry[i-1] = ry[*natm-1];
       rz[i-1] = rz[*natm-1];
       vx[i-1] = vx[*natm-1];
       vy[i-1] = vy[*natm-1];
       vz[i-1] = vz[*natm-1];
       *natm = *natm - 1;
}
