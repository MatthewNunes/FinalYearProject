__global__
void force (float *virialArray, float *potentialArray, float *pval, float *vval, float *rx, float *ry, float *rz, float *fx, float *fy, float *fz, float sigma, float rcut, float vrcut, float dvrc12, float dvrcut, int *head, int *list, int mx, int my, int mz, int natoms, int step, float sfx, float sfy, float sfz)
{
   float sigsq, rcutsq;
   float rxi, ryi, rzi, fxi, fyi, fzi;
   float rxij, ryij, rzij, rijsq;
   float rij, sr2, sr6, vij, wij, fij, fxij, fyij, fzij;
   float potential, virial;
   int i, icell, j, jcell, nabor;
   int xi, yi, zi, ix, jx, kx, xcell, ycell, zcell;

   sigsq  = sigma*sigma;
   rcutsq = rcut*rcut;

   potential = 0.0;
   virial    = 0.0;
   int element = blockIdx.x * blockDim.x + threadIdx.x;
   if (element < natoms)
   {
      rxi = rx[element];
      ryi = ry[element];
      rzi = rz[element];
      fxi = 0.0;
      fyi = 0.0;
      fzi = 0.0;
      xi = (int)((rxi+0.5)/sfx) + 1;
      yi = (int)((ryi+0.5)/sfy) + 1;
      zi = (int)((rzi+0.5)/sfz) + 1;
      icell = xi + (mx+2)*(yi+zi*(my+2));
   //	 printf("Particle %5d, (xi,yi,zi) = %d,%d,%d, icell = %5d\n",i,xi,yi,zi,icell);
            for (ix=-1;ix<=1;ix++)
                for (jx=-1;jx<=1;jx++)
                    for (kx=-1;kx<=1;kx++)
                    {
                        xcell = ix+xi;
                        ycell = jx+yi;
                        zcell = kx+zi;
                        jcell = xcell + (mx+2)*(ycell+(my+2)*zcell);
   //	printf("%d (%d,%d,%d); ",jcell,xcell,ycell,zcell);
                        j = head[jcell];
                        while (j>=0) 
                        {
                           if (j!=element) 
                           {
                                rxij = rxi - rx[j];
                                ryij = ryi - ry[j];
                                rzij = rzi - rz[j];
                                rijsq = rxij*rxij + ryij*ryij + rzij*rzij;
                                if (rijsq < rcutsq) 
                                {
                                    //start force_ij
                                    float rij, sr2, sr6, fij;
                                    rij = (float) sqrt ((double)rijsq);
                                    sr2 = sigsq/rijsq;
                                    sr6 = sr2*sr2*sr2;
                                    vij = sr6*(sr6-1.0) - vrcut - dvrc12*(rij-rcut);
                                    wij = sr6*(sr6-0.5) + dvrcut*rij;
                                    fij = wij/rijsq;
                                    fxij = fij*rxij;
                                    fyij = fij*ryij;
                                    fzij = fij*rzij;
                                    //end force_ij
                                    //force_ij(rijsq, rxij, ryij, rzij, sigsq, vrcut, dvrc12, rcut, dvrcut, &vij, &wij, &fxij, &fyij, &fzij);
                                    wij *= 0.5;
                                    vij *= 0.5;
                                    potential += vij;
                                    virial    += wij;
                                    fxi       += fxij;
                                    fyi       += fyij;
                                    fzi       += fzij;
                                }
                           }
                              j = list[j];
                        }
                     }
            
            *(fx+element) = 48.0*fxi;
            *(fy+element) = 48.0*fyi;
            *(fz+element) = 48.0*fzi;
            virialArray[element] = virial;
            potentialArray[element] = potential;

   }
   
      if (element == 0)
      {
         for (i = 1; i < natoms; i++)
         {
            potential += potentialArray[i];
            virial += virialArray[i];
         }
         potential *= 4.0;
         virial    *= 48.0/3.0;
         *pval = potential;
         *vval = virial;
      }
}
