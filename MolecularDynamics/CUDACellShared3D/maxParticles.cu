#include "moldyn.h"

int maxParticles(int *head, int *list, int gtx, int gty, int gtz)
{
   int icell, i, p;
   int pmax = 0;

   for(icell=0;icell<gtx*gty*gtz;icell++){
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
