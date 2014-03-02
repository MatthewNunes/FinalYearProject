#define IMULT 16807
#define IMOD  2147483647

void pseudorand(is, c)
int *is;
float *c;
{
   double scale=0.0000000004656612875;
   int temp, is1, is2, iss2;

   temp = *is;
   if (temp<=0) temp = 1;
   is2 = temp%32768;
   is1 = (temp-is2)/32768;
   iss2 = temp*IMULT;
   is2  = iss2%32768;
   is1  = (is1*IMULT + (iss2-is2)/32768)%65536;
   temp = (is1*32768 + is2)%IMOD;
   *c   = 2.0*(scale*(float)temp-0.5)*0.000001;
   *is  = temp;
}
