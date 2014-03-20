/*
FILE: read_input.c

DOC:  read_input reads in the input and control parameters
*/
#include <math.h>
#include "moldyn.h"
#include <stdio.h>

int read_input (float *sigma, float *rcut, float *dt, float *eqtemp, float *dens, int *nstep, int *nequil, int *iscale, int *iprint, int *nc)
{
   int ierr, natoms, m;

   printf ("\n **** PROGRAM MDLJ ****");
   printf ("\n MOLECULAR DYNAMICS SIMULATION ");
   printf ("\n WITH LINKED LIST");
      
   printf ("\n ENTER NUMBER OF TIME STEPS");
   ierr=scanf  ("%d", nstep);
   printf ("\n ENTER NUMBER OF EQUILIBRATION TIME STEPS");
   ierr=scanf  ("%d", nequil);

   printf ("\n ENTER NUMBER OF STEPS BETWEEN SUMMARY OUTPUT");
   ierr=scanf  ("%d", iprint);
   printf ("\n ENTER NUMBER OF STEPS BETWEEN GLOBAL RESCALES");
   ierr=scanf  ("%d", iscale);

   printf ("\n\n ENTER THE FOLLOWING IN LENNARD-JONES UNITS");
   printf ("\n ENTER THE TEMPERATURE");
   ierr=scanf  ("%f", eqtemp);
   printf ("\n ENTER THE DENSITY");
   ierr=scanf  ("%f", dens);
   printf ("\n ENTER THE POTENTIAL CUTOFF DISTANCE");
   ierr=scanf  ("%f", rcut);
   printf ("\n ENTER THE TIMESTEP");
   ierr=scanf  ("%f", dt);

   printf ("\n NUMBER OF STEPS                    %10d",   *nstep);
   printf ("\n NUMBER OF EQUIL STEPS              %10d",   *nequil);
   printf ("\n SUMMARY OUTPUT FREQUENCY           %10d",   *iprint);
   printf ("\n GLOBAL RESCALE FREQUENCY           %10d",   *iscale);
   printf ("\n TEMPERATURE                        %10.4f", *eqtemp);
   printf ("\n DENSITY                            %10.4f", *dens);
   printf ("\n POTENTIAL CUTOFF                   %10.4f", *rcut);
   printf ("\n TIMESTEP                           %10.4f", *dt);

   printf ("\n ENTER NC");
   ierr=scanf  ("%d", nc);

   natoms = 4*(*nc)*(*nc)*(*nc);
  
   printf ("\n NUMBER OF ATOMS BEING USED %8d", natoms);

   *sigma = pow ((*dens)/(float)natoms, 1.0/3.0);
   *rcut  = (*rcut)*(*sigma);
   m     = (int)(1.0/(*rcut));
   *dens  = (*dens)/((*sigma)*(*sigma)*(*sigma));
   *dt    = (*dt)*(*sigma);

   if (m < 1){
      printf ("\n SYSTEM TOO SMALL FOR ARRAY");
      return (1);
   }

   return (0);
}
