#include <math.h>

void force_ij(float rijsq, float rxij, float ryij, float rzij, float sigsq, float vrcut, float dvrc12, float rcut, float dvrcut, float *vij, float *wij, float *fxij, float *fyij, float *fzij)
{
    float rij, sr2, sr6, fij;

    rij = (float) sqrt ((double)rijsq);
    sr2 = sigsq/rijsq;
    sr6 = sr2*sr2*sr2;
    *vij = sr6*(sr6-1.0) - vrcut - dvrc12*(rij-rcut);
    *wij = sr6*(sr6-0.5) + dvrcut*rij;
    fij = *wij/rijsq;
    *fxij = fij*rxij;
    *fyij = fij*ryij;
    *fzij = fij*rzij;
}
