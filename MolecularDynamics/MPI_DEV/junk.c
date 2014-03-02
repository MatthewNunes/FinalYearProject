#include <stdio.h>
#include <math.h>

main()
{
   float rijsq = 0.001952872;
   float rij;

   rij = sqrt ((double)rijsq);

   printf("\nrijsq=%13.6f, rij=%13.6f",rijsq,rij);
}
