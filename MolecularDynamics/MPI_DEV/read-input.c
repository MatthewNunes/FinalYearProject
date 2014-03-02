/*
FILE: read_input.c

DOC:  read_input reads in the input and control parameters
*/
#include "moldyn.h"
#include "math.h"
#include <stdio.h>

int read_input (pass, ip)
float pass[];
int *ip;
{
   int nprosx, nprosy, nprosz;
   int nstep, nequil;
   int isave, isvunf, isvpfs;
   int iscale, iprint;
   float eqtemp, dens, rcut, dt, freex;
   int nstart, nc, natoms;
   int m;
   float sigma, denlj, mass;

   printf ("\n **** PROGRAM MDLJ ****");
   printf ("\n MOLECULAR DYNAMICS SIMULATION ");
   printf ("\n WITH LINKED LIST");
      
   printf ("\n ENTER NUMBER OF PROCESSORS IN X DIRECTION");
   int chk = scanf  ("%d",&nprosx);
   printf ("\n ENTER NUMBER OF PROCESSORS IN Y DIRECTION");
   chk = scanf  ("%d",&nprosy);
   printf ("\n ENTER NUMBER OF PROCESSORS IN Z DIRECTION");
   chk = scanf  ("%d",&nprosz);

   printf ("\n ENTER NUMBER OF TIME STEPS");
   chk = scanf  ("%d", &nstep);
   printf ("\n ENTER NUMBER OF EQUILIBRATION TIME STEPS");
   chk = scanf  ("%d", &nequil);

   printf ("\n FREQUENCY OF FORMATTED DATA SAVES VIA HOST");
   chk = scanf  ("%d", &isave);
   printf ("\n FREQUENCY OF UNFORMATTED DATA SAVES VIA HOST");
   chk = scanf  ("%d", &isvunf);
   printf ("\n FREQUENCY OF UNFORMATTED DATA SAVES VIA PFS");
   chk = scanf  ("%d", &isvpfs);

   printf ("\n ENTER NUMBER OF STEPS BETWEEN SUMMARY OUTPUT");
   chk = scanf  ("%d", &iprint);
   printf ("\n ENTER NUMBER OF STEPS BETWEEN GLOBAL RESCALES");
   chk = scanf  ("%d", &iscale);

   printf ("\n\n ENTER THE FOLLOWING IN LENNARD-JONES UNITS");
   printf ("\n ENTER THE TEMPERATURE");
   chk = scanf  ("%f", &eqtemp);
   printf ("\n ENTER THE DENSITY");
   chk = scanf  ("%f", &dens);
   printf ("\n ENTER THE POTENTIAL CUTOFF DISTANCE");
   chk = scanf  ("%f", &rcut);
   printf ("\n ENTER THE TIMESTEP");
   chk = scanf  ("%f", &dt);

   printf ("\n NUMBER OF STEPS                    %10d",   nstep);
   printf ("\n NUMBER OF EQUIL STEPS              %10d",   nequil);
   printf ("\n FORMATTED SAVE FREQUENCY VIA HOST  %10d",   isave);
   printf ("\n UNFORMATTED SAVE FREQUENCY VIA HOST%10d",   isvunf);
   printf ("\n UNFORMATTED SAVE FREQUENCY VIA PFS %10d",   isvpfs);
   printf ("\n SUMMARY OUTPUT FREQUENCY           %10d",   iprint);
   printf ("\n GLOBAL RESCALE FREQUENCY           %10d",   iscale);
   printf ("\n TEMPERATURE                        %10.4f", eqtemp);
   printf ("\n DENSITY                            %10.4f", dens);
   printf ("\n POTENTIAL CUTOFF                   %10.4f", rcut);
   printf ("\n TIMESTEP                           %10.4f", dt);

   printf ("\n ENTER NC");
   chk = scanf  ("%d", &nc);

   natoms = 4*nc*nc*nc;
   *ip = iprint;
  
   printf ("\n NUMBER OF ATOMS BEING USED %8d", natoms);

   for (;;){
      printf ("\n Please choose option for initializing configuration:");
      printf ("\n    1...Cold start, scalable but unsafe (FCC)");
      printf ("\n    2...Cold start, safe but nonscalable (FCCBIG)");
      printf ("\n    3...Cold start, safe and sequential (CSTART)");
      printf ("\n    4...Cold start, safe and parallel (PARINT)");
      printf ("\n    5...Warm start from formatted file via host (WSTART)");
      printf ("\n    6...Warm start from unformatted file via host (WSTUNF)");
      printf ("\n    7...Warm start from pfs (PFSINT)");
      chk = scanf  ("%d", &nstart);
      if (nstart<1 || nstart>7) 
         printf ("\n *** Invalid option - try again ***");
      else if (nstart==1 || nstart==2 || nstart==4 || nstart==7)
         printf ("\n *** Sorry option not implemented - try again ***");
      else
         break;
   }

   sigma = pow (dens/(float)natoms, 1.0/3.0);
   rcut  = rcut*sigma;
   m     = (int)(1.0/rcut);
   mass  = 1.0;
   denlj = dens;
   dens  = dens/(sigma*sigma*sigma);
   dt    = dt*sigma;
   freex = (float)(3*(natoms-1));

   if (min(m/nprosx, min(m/nprosy,m/nprosz)) < 1){
      printf ("\n SYSTEM TOO SMALL FOR ARRAY");
      return (1);
   }

   pass[0] = sigma;
   pass[1] = rcut;
   pass[2] = dt;
   pass[3] = (float)nprosx;
   pass[4] = (float)nprosy;
   pass[5] = (float)nprosz;
   pass[6] = (float)nstep;
   pass[7] = (float)nequil;
   pass[8] = (float)isave;
   pass[9]= (float)iscale;
   pass[10]= eqtemp;
   pass[11]= (float)nstart;
   pass[12]= (float)isvunf;
   pass[13]= (float)isvpfs;
   pass[14]= (float)nc;
   pass[15]= freex;
   pass[16]= dens;

   return (0);
}
