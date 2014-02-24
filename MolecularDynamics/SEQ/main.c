/*
FILE  main.c

DOC   This program simulates a three-dimensional Lennard-Jones fluid.

LANG  C plus MPI message passing.

HIS   1) Parallel version originally written by University of
HIS      Southampton.
HIS   2) Adopted to run on the Intel iPSC/2 computer at Daresbury
HIS      Laboratory, and placed in the public domain through the
HIS      CCP5 programme.
HIS   3) Adopted to run on the Intel Paragon and enhanced by David W.
HIS      Walker at Oak Ridge National Laboratory, Tennessess, USA
HIS   4) Converted to use MPI message passing library by David W. Walker
HIS      at University of Wales Cardiff in July 1997.
HIS   5) Converted from Fortran to C in November 1997.

*/

#include <stdio.h>
#include <stdlib.h>
#include "moldyn.h"

int main ( int argc, char *argv[])
{
   float sigma, rcut, dt, eqtemp, dens, boxlx, boxly, boxlz, sfx, sfy, sfz, sr6, vrcut, dvrcut, dvrc12, freex; 
   int nstep, nequil, iscale, nc, mx, my, mz, iprint, map[MAPSIZ];
   float *rx, *ry, *rz, *vx, *vy, *vz, *fx, *fy, *fz;
   float ace, acv, ack, acp, acesq, acvsq, acksq, acpsq, vg, kg, wg;
   int   head[NCELL], list[NMAX];
   int   natoms=0;
   int ierror;
   int jstart, step, itemp;
   float potential, virial, kinetic;
   float tmpx;
   int i, icell;

   ierror = input_parameters (&sigma, &rcut, &dt, &eqtemp, &dens, &boxlx, &boxly, &boxlz, &sfx, &sfy, &sfz, &sr6, &vrcut, &dvrcut, &dvrc12, &freex, &nstep, &nequil, &iscale, &nc, &natoms, &mx, &my, &mz, &iprint, map);
   //printf ("\nReturned from input_parameters, natoms = %d\n", natoms);

   rx = (float *)malloc(2*natoms*sizeof(float));
   ry = (float *)malloc(2*natoms*sizeof(float));
   rz = (float *)malloc(2*natoms*sizeof(float));
   vx = (float *)malloc(natoms*sizeof(float));
   vy = (float *)malloc(natoms*sizeof(float));
   vz = (float *)malloc(natoms*sizeof(float));
   fx = (float *)malloc(natoms*sizeof(float));
   fy = (float *)malloc(natoms*sizeof(float));
   fz = (float *)malloc(natoms*sizeof(float));

   initialise_particles (rx, ry, rz, vx, vy, vz, nc);
   //printf ("\nReturned from initialise_particles\n");

   loop_initialise(&ace, &acv, &ack, &acp, &acesq, &acvsq, &acksq, &acpsq, sigma, rcut, dt);
   //printf ("\nReturned from loop_initialise\n");

   output_particles(rx,ry,rz,vx,vy,vz,fx,fy,fz,0);
      movout (rx, ry, rz, vx, vy, vz, sfx, sfy, sfz, head, list, mx, my, mz, natoms);
   //printf ("\nReturned from movout\n");
   //   check_cells(rx, ry, rz, head, list, mx, my, mz, natoms,0,0);
      force (&potential, &virial, rx, ry, rz, fx, fy, fz, sigma, rcut, vrcut, dvrc12, dvrcut, head, list, map, mx, my, mz, natoms,0);
   //printf ("\nReturned from force: potential = %f, virial = %f, kinetic = %f\n",potential, virial, kinetic);
   output_particles(rx,ry,rz,vx,vy,vz,fx,fy,fz,0);


   for(step=1;step<=nstep;step++){
   //printf ("\nStarted step %d\n",step);
      movea (rx, ry, rz, vx, vy, vz, fx, fy, fz, dt, natoms);
//      check_cells(rx, ry, rz, head, list, mx, my, mz, natoms,step,step);
   //printf ("\nReturned from movea\n");
      movout (rx, ry, rz, vx, vy, vz, sfx, sfy, sfz, head, list, mx, my, mz, natoms);
  // printf ("\nReturned from movout\n");
  //    check_cells(rx, ry, rz, head, list, mx, my, mz, natoms,step,step);
      force (&potential, &virial, rx, ry, rz, fx, fy, fz, sigma, rcut, vrcut, dvrc12, dvrcut, head, list, map, mx, my, mz, natoms, step);
   //printf ("\nReturned from force: potential = %f, virial = %f, kinetic = %f\n",potential, virial, kinetic);
      moveb (&kinetic, vx, vy, vz, fx, fy, fz, dt, natoms);
 //     check_cells(rx, ry, rz, head, list, mx, my, mz, natoms,step,step);
//   printf ("\nReturned from moveb: potential = %f, virial = %f, kinetic = %f\n",potential, virial, kinetic);
      sum_energies (potential, kinetic, virial, &vg, &wg, &kg);
      hloop (kinetic, step, vg, wg, kg, freex, dens, sigma, eqtemp, &tmpx, &ace, &acv, &ack, &acp, &acesq, &acvsq, &acksq, &acpsq, vx, vy, vz, iscale, iprint, nequil, natoms);
   }

   tidyup (ace, ack, acv, acp, acesq, acksq, acvsq, acpsq, nstep, nequil);
   
   return 0;
}
