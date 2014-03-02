int findit (p, list, head,iparam)
int p;
int list[];
int head[];
int *iparam;
{
   int found;
   int i, icell;

   int mx = iparam[13];
   int my = iparam[14];
   int mz = iparam[15];

   found = 0;
   for (icell=1;icell<=mx*my*mz; icell++) {
      i = head[icell-1];
      while (i != 0) {
         if (i==p) found++;
         i = list[i-1];
      }
   }

   return found;
}
     
  
   
