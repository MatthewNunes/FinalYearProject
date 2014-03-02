#include <stdio.h>

void edgprint (ilo, ihi, jlo, jhi, klo, khi, rx, ry, rz, list, head,iparam)
int ilo;
int ihi;
int jlo;
int jhi;
int klo;
int khi;
float rx[];
float ry[];
float rz[];
int head[];
int list[];
int *iparam;
{
   int i, j, k, indx, ipart, kadd, jadd;

   int mx = iparam[13];
   int my = iparam[14];
   int mz = iparam[15];

   for(k=klo;k<=khi;k++){
      kadd = (k-1)*my;
      for(j=jlo;j<=jhi;j++){
         jadd = (j-1+kadd)*mx;
         for(i=ilo;i<=ihi;i++){
            indx  = i + jadd;
            ipart = head[indx-1];
            printf("\nCell at (%2d,%2d,%2d) :",i,j,k);
            while (ipart != 0){
               printf("\nParticle %d: %f, %f, %f",ipart,
                      rx[ipart-1],ry[ipart-1],rz[ipart-1]);
               ipart = list[ipart-1];
            }
         }
      }
   }
}
