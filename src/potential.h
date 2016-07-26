#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#define DIMS 4
#define TRUE 0
#define FALSE 1
#define MAXSTR 512

#define NULLCHECK(ptr,name)  if(ptr == NULL) printf("Error allocating %s",name)
#define FREE(ptr) if (ptr != NULL) { free(ptr); ptr = NULL;}

#define real double

typedef struct Parameters {

    int n,nstars,nt,threads_per_block,ntargets;

    real dt,sigma,tol;
    char outputdir[MAXSTR], targetfile[MAXSTR];


} Parameters;

Parameters params;


void read_param_file(char *fname);
double log_likelihood(double *source_points, double *target_points, double *q, double *res, double h, int n_source, int n_target, int dims, double tol) ;
#ifdef __cplusplus 
extern "C"
#endif
void output(int time, real *x, real *y, real *vx, real *vy,int n);
#ifdef __cplusplus 
extern "C"
#endif
void gpu_evolve(real *pot_pars, real *points, real *final);
