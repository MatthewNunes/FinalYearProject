#define RANKHOST 0
#define min(A,B) ((A) < (B) ? (A) : (B))
#define max(A,B) ((A) > (B) ? (A) : (B))
#define abs(A) ((A)>0 ? (A) : -(A))
#define MMX 50
#define MMY 50
#define MMZ 50
#define NCELL (MMX*MMY*MMZ)
#define NMAX 100000
#define NMAX2 90000
#define SCRSIZ 6*NMAX2
#define MYBUFSIZ 524288
#define MAXLST 2048
#define INSIZE 1000
#define INREAL 6*INSIZE
#define MAXPAS 256000/4
#define FOR1 0
#define FOR2 NMAX2
#define FOR3 (2*NMAX2)
#define OPSIZ (SCRSIZ/6)
#define OP1 0
#define OP2 OPSIZ
#define OP3 (2*OPSIZ)
#define OP4 (3*OPSIZ)
#define OP5 (4*OPSIZ)
#define OP6 (5*OPSIZ)
#define BLOCK_WIDTH 512
#define BLOCK_WIDTH2 512
#define BLOCK_WIDTH3 512
__global__ void scalet ( float *vx, float *vy, float *vz, float kinetic, float eqtemp, float tmpx, int iscale, int natoms, int step);
extern int input_parameters (float *sigma, float *rcut, float *dt, float *eqtemp, float *dens, float *boxlx, float *boxly, float *boxlz, float *sfx, float *sfy, float *sfz, float *sr6, float *vrcut, float *dvrcut, float *dvrc12, float *freex, int *nstep, int *nequil, int *iscale, int *nc, int *natoms, int *mx, int *my, int *mz, int *iprint);
extern int read_input (float *sigma, float *rcut, float *dt, float *eqtemp, float *dens, int *nstep, int *nequil, int *iscale, int *iprint, int *nc);
extern void initialise_particles (float rx[], float ry[], float rz[], float vx[], float vy[], float vz[], int nc);
extern void pseudorand(int *is, float *c);
extern void loop_initialise(float *ace, float *acv, float *ack, float *acp, float *acesq, float *acvsq, float *acksq, float *acpsq, float sigma, float rcut, float dt);
__global__ void initialMovout(float *rx, float *ry, float *rz, int natoms);
extern void copyToConstant(float *sig, float *rcu, float *vrc, float *dvr, float *dvrc, int * m, int *mm, int *mmm, int *nat, int *ste, float *sf, float *sff, float *sfff);
extern void copyRcut(float *rcu);
extern void copyVrcut(float *vrc);
extern void copyDvrc12(float *dvr);
extern void copyDvrcut(float *dvrc);
extern void copyMx(int *m);
extern void copyMy(int *mm);
extern void copyMz(int *mmm);
extern void copyNatoms(int *nat);
extern void copyStep(int *ste);
extern void copySfx(float *sf);
extern void copySfy(float *sff);
extern void copySfz(float *sfff);
extern long unsigned int get_tick();
//__global__ void force (float *virialArray, float *potentialArray, float *pval, float *vval, float *rx, float *ry, float *rz, float *fx, float *fy, float *fz, int *head, int *list);
__global__ void force(float *virialArray, float *potentialArray, float *rx, float *ry, float *rz, float *fx, float *fy, float *fz, float sigma, float rcut, float vrcut, float dvrc12, float dvrcut, int *head, int *list, int mx, int my, int mz, int natoms, float sfx, float sfy, float sfz);
extern void movout (float *rx, float *ry, float *rz, float *vx, float *vy, float *vz, float sfx, float sfy, float sfz, int *head, int *list, int mx, int my, int mz, int natoms);
__global__ void movea (float *rx, float *ry, float *rz, float *vx, float *vy, float *vz, float *fx, float *fy, float *fz, float dt, int natoms);
__global__ void moveb (float *kineticArray, float *vx, float *vy, float *vz, float *fx, float *fy, float *fz, float dt, int natoms);
extern void sum_energies (float v, float k, float w, float *vg, float *wg, float *kg);
extern void hloop (float kinetic, int step, float vg, float wg, float kg, float freex, float dens, float sigma, float eqtemp, float *tmpx, float *ace, float *acv, float *ack, float *acp, float *acesq, float *acvsq, float *acksq, float *acpsq, float *vx, float *vy, float *vz, int iscale, int iprint, int nequil, int natoms, int block_width, float *d_vx, float *d_vy, float *d_vz);
extern void tidyup (float ace, float ack, float acv, float acp, float acesq, float acksq, float acvsq, float acpsq, int nstep, int nequil);
extern void check_cells(float *rx, float *ry, float *rz, int *head, int *list, int mx, int my, int mz, int natoms, int step, int pstep);
extern void output_particles(float *rx, float *ry, float *rz, float *vx, float *vy, float *vz, float *fx, float *fy, float *fz,  int natm);
extern void force_ij(float rijsq, float rxij, float ryij, float rzij, float sigsq, float vrcut, float dvrc12, float rcut, float dvrcut, float *vij, float *wij, float *fxij, float *fyij, float *fzij);
