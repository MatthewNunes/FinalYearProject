#ifndef SEQ
#include "mpi.h"
#endif

#include "moldyn.h"

int input_parameters (rank, xid, yid, zid, xor, yor, zor, map,iparam,fparam)
int    rank; 
int   *xid;
int   *yid;
int   *zid;
float *xor;
float *yor;
float *zor;
int *map;
int *iparam;
float *fparam;
{
   float sigma, rcut, dt, dens; 
   float vrcut, dvrcut, dvrc12;
   float sr6, sfx, sfy, sfz, cellix, celliy, celliz;
   float freex, eqtemp;
   float boxlx, boxly, boxlz;
   int nstep, nequil, nstart;
   int natoms, iprint;
   int nprosx, nprosy, nprosz, mx, my, mz, iscale, isave, isvunf, isvpfs, nc;
   
   float pass[20];
   int abort;

   if (I_AM_HOST) abort = read_input (pass, &iprint);

#ifndef SEQ
   MPI_Bcast (&abort, 1, MPI_INT,  RANKHOST, MPI_COMM_WORLD);
#endif

   if (abort==0){
#ifndef SEQ
      MPI_Bcast (pass, 16, MPI_REAL, RANKHOST, MPI_COMM_WORLD);
#endif

      sigma  = pass[0];
      rcut   = pass[1];
      dt     = pass[2];
      nprosx = (int)(pass[3]+0.1);
      nprosy = (int)(pass[4]+0.1);
      nprosz = (int)(pass[5]+0.1);
      nstep  = (int)(pass[6]+0.1);
      nequil = (int)(pass[7]+0.1);
      
      if (pass[8] < 0.0) 
         isave = (int)(pass[8]-0.1);
      else
         isave = (int)(pass[8]+0.1);

      iscale = (int)(pass[9]+0.1);
      eqtemp = pass[10];
      nstart = (int)(pass[11]+0.1);
      
      if (pass[12] < 0.0)
         isvunf = (int)(pass[12]-0.1);
      else
         isvunf = (int)(pass[12]+0.1);
      
      if (pass[13] < 0.0)
         isvpfs = (int)(pass[13]-0.1);
      else
         isvpfs = (int)(pass[13]+0.1);

      nc    = (int)(pass[14]+0.1);
      freex = pass[15];
      dens= pass[16];

      natoms = 4*nc*nc*nc;

      *xid = rank%nprosx;           // (xid,yid,zid) is position in 
      *yid = (rank/nprosx)%nprosy;  // processor mesh
      *zid = rank/(nprosx*nprosy);

      *xor = ((float)(*xid)+0.5)/(float)nprosx - 0.5; // (xor,yor,zor) is the
      *yor = ((float)(*yid)+0.5)/(float)nprosy - 0.5; // centre of a processor's
      *zor = ((float)(*zid)+0.5)/(float)nprosz - 0.5; // sub-domain

      boxlx = 1.0/(float)nprosx; // (boxlx,boxly,boxlz) is the size of a
      boxly = 1.0/(float)nprosy; // processor's sub-domain.
      boxlz = 1.0/(float)nprosz;

      mx = (int)(boxlx/rcut)+2; // (mx,my,mz) is the number of cells
      my = (int)(boxly/rcut)+2; // in each processor's sub-domain,
      mz = (int)(boxlz/rcut)+2; // including ghost cells.

      set_nearest_cells (map);

      sfx = ((float)mx - 2.0)/((float)mx*boxlx); // (sfx,sfy,sfz) 
      sfy = ((float)my - 2.0)/((float)my*boxly);
      sfz = ((float)mz - 2.0)/((float)mz*boxlz);

      cellix = (float)mx;
      celliy = (float)my;
      celliz = (float)mz;

      sr6    = (sigma/rcut)*(sigma/rcut);
      sr6    = sr6*sr6*sr6;
      vrcut  = sr6*(sr6-1.0);
      dvrcut = -1.0*sr6*(sr6-0.5)/rcut;
      dvrc12 = 12.0*dvrcut;
      iparam[0] = nprosx;
      iparam[1] = nprosy;
      iparam[2] = nprosz;
      iparam[3] = nstep;
      iparam[4] = nequil;
      iparam[5] = isave;
      iparam[6] = iscale;
      iparam[7] = nstart;
      iparam[8] = isvunf;
      iparam[9] = isvpfs;
      iparam[10] = nc;
      iparam[11] = natoms;
      iparam[12] = iprint;
      iparam[13] = mx;
      iparam[14] = my;
      iparam[15] = mz;

      fparam[0] = sigma;
      fparam[1] = rcut;
      fparam[2] = dt;
      fparam[3] = eqtemp;
      fparam[4] = dens;
      fparam[5] = freex;
      fparam[6] = boxlx;
      fparam[7] = boxly;
      fparam[8] = boxlz;
      fparam[9] = sfx;
      fparam[10] = sfy;
      fparam[11] = sfz;
      fparam[12] = cellix;
      fparam[13] = celliy;
      fparam[14] = celliz;
//      fparam[15] = sr6;
      fparam[16] = vrcut;
      fparam[17] = dvrcut;
      fparam[18] = dvrc12;

   }
   return (abort);
}
