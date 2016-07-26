#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#define DIMS 4
#define TRUE 1 
#define FALSE 0
#define MAXSTR 512

#define NULLCHECK(ptr,name)  if(ptr == NULL) printf("Error allocating %s",name)
#define FREE(ptr) if (ptr != NULL) { free(ptr); ptr = NULL;}

#define real double

typedef struct Parameters {

    int n,nstars,nt,threads_per_block,ntargets;
    int npars;
    int kdemethod;
    real dt,sigma,tol,size, simplex_step;
    char outputdir[MAXSTR], targetfile[MAXSTR], kdemethod_str[MAXSTR];

    int generate;


} Parameters;

Parameters params;


void read_param_file(char *fname);
void output_init(double *points, double *pot_pars);
int minimize(double *init_pars);
double log_likelihood(double *source_points, double *target_points, double *q, double *res, double h, int n_source, int n_target, int dims, double tol) ;
#ifdef __cplusplus 
extern "C" {
#endif
void output(int time, real *x, real *y, real *vx, real *vy,int n);
void gpu_init(void);
void gpu_free(void);
real gpu_evolve(const real *pot_pars, real *points, const real *ic, real *kde_tot, int silent);
void generate_system(const real *pot_pars, real *points, int silent);
void add_kde(double *source_points, double *kde_tot);
#ifdef __cplusplus 
}
#endif

