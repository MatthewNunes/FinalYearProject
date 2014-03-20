#include "moldyn.h"

int maxParticles(int *head, int *list, int mx, int my, int mz)
{
   int icell, i, p;
   int pmax = 0;

   for(icell=0;icell<((mx+2)*(my+2)*(mz+2));icell++){
       i = head[icell];
       p = 0;
       while (i>=0) {
           p+=1;
	   i = list[i];
       }
       if(p>pmax) pmax = p;
   }
   return (pmax);
}
