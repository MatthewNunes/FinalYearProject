#include <math.h>
#include <stdio.h>

void tidyup (float ace, float ack, float acv, float acp, float acesq, float acksq, float acvsq, float acpsq, int nstep, int nequil)
{
   float norm, ave, avk, avv, avp, avt;
   float fle=0.0;
   float flv=0.0;
   float flk=0.0;
   float flp=0.0;
   float flt=0.0;

   norm = (float)(nstep-nequil);
   ave  = ace/norm;
   avk  = ack/norm;
   avv  = acv/norm;
   avp  = acp/norm;

//      printf("\nacesq = %f, norm = %f, ace = %f, ave = %f",acesq,norm,ace,ave);

   acesq = (acesq/norm) - ave*ave;
   acksq = (acksq/norm) - avk*avk;
   acvsq = (acvsq/norm) - avv*avv;
   acpsq = (acpsq/norm) - avp*avp;

   if (acesq > 0.0) fle = sqrt((double)acesq);
   if (acksq > 0.0) flk = sqrt((double)acksq);
   if (acvsq > 0.0) flv = sqrt((double)acvsq);
   if (acpsq > 0.0) flp = sqrt((double)acpsq);

   avt = avk/1.5;
   flt = flk/1.5;

   printf ("\n AVERAGES  %10.5f  %10.5f  %10.5f  %10.5f  %10.5f",ave,avk,avv,avp,avt);
   printf ("\n FLUCTS    %10.5f  %10.5f  %10.5f  %10.5f  %10.5f\n",fle,flk,flv,flp,flt);
}
