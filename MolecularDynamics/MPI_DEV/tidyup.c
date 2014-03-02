#include <math.h>
#include <stdio.h>

void tidyup (int *iparam, float *fparam)
{
   float norm, ave, avk, avv, avp, avt;
   float fle=0.0;
   float flv=0.0;
   float flk=0.0;
   float flp=0.0;
   float flt=0.0;

   int nstep = iparam[3];
   int nequil = iparam[4];

   float ace   = fparam[23];
   float acv   = fparam[24];
   float ack   = fparam[25];
   float acp   = fparam[26];
   float acesq = fparam[27];
   float acvsq = fparam[28];
   float acksq = fparam[29];
   float acpsq = fparam[30];

   norm = (float)(nstep-nequil);
   ave  = ace/norm;
   avk  = ack/norm;
   avv  = acv/norm;
   avp  = acp/norm;

   acesq = (acesq/norm) - ave*ave;
   acksq = (acksq/norm) - avk*avk;
   acvsq = (acvsq/norm) - avv*avv;
   acpsq = (acpsq/norm) - avp*avp;
   fparam[27] = acesq;
   fparam[28] = acvsq;
   fparam[29] = acksq;
   fparam[30] = acpsq;

   if (acesq > 0.0) fle = sqrt((double)acesq);
   if (acksq > 0.0) flk = sqrt((double)acksq);
   if (acvsq > 0.0) flv = sqrt((double)acvsq);
   if (acpsq > 0.0) flp = sqrt((double)acpsq);

   avt = avk/1.5;
   flt = flk/1.5;

   printf ("\n AVERAGES  %10.5f  %10.5f  %10.5f  %10.5f  %10.5f",ave,avk,avv,avp,avt);
   printf ("\n FLUCTS    %10.5f  %10.5f  %10.5f  %10.5f  %10.5f",fle,flk,flv,flp,flt);
   printf("\n");
}


