#include "moldyn.h"

void tohost (bigbuf, npass, natm, rx, ry, rz, vx, vy, vz, xor, yor, zor)
float  bigbuf[];
int    npass;
int   *natm;
float rx[];
float ry[];
float rz[];
float vx[];
float vy[];
float vz[];
float  xor;
float  yor;
float  zor;
{
   int i;

   for(i=1;i<=6*npass;i+=6){
      rx[*natm] = bigbuf[i-1] - xor;
      ry[*natm] = bigbuf[i]   - yor;
      rz[*natm] = bigbuf[i+1] - zor;
      vx[*natm] = bigbuf[i+2];
      vy[*natm] = bigbuf[i+3];
      vz[*natm] = bigbuf[i+4];
      (*natm)++;
   }
}
