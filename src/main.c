#include "potential.h"


int main(int argc, char *argv[]) {

    read_param_file(argv[1]);


    printf("Reading parameters...\n");
    printf(
            "\tNSTARS         = %d\n\
        NT             = %d \n\
        DT             = %lg\n\
        ThreadsPerBock = %d\n\
        SIGMA          = %lg\n\
        TOL            = %.e\n\
        NTARGETS       = %d\n\
        TARGETFILE     = %s\n\
        OUTPUTDIR      = %s\n", 
            params.nstars, params.nt, params.dt, params.threads_per_block, params.sigma, params.tol, params.ntargets , params.targetfile,params.outputdir);


    params.n = (int)(real)sqrt(params.nstars);
    params.nstars = params.n*params.n;


    real *targets = (real *)malloc(sizeof(real)*params.ntargets*DIMS);
    NULLCHECK(targets,"targets");
    real *source = (real *)malloc(sizeof(real)*params.nstars*DIMS);
    NULLCHECK(source,"source");
    real *weights = (real *)malloc(sizeof(real)*params.nstars);
    NULLCHECK(weights,"weights");
    real *kde_res = (real *)malloc(sizeof(real)*params.ntargets);
    NULLCHECK(kde_res,"kde_res");

    FILE *f = fopen(params.targetfile,"r");
    fread(targets,sizeof(real),params.ntargets*DIMS,f);
    fclose(f);


    real pot_pars[2] = {atof(argv[2]), atof(argv[3])};

    gpu_evolve(pot_pars, targets,source);


    int i;

    for(i=0;i<params.nstars;i++) {
        weights[i] = pow(2*M_PI,-DIMS*.5) / (real)params.nstars; 
    }

    for(i=0;i<params.ntargets;i++) kde_res[i] = 0;


    double tot;
    
    tot = log_likelihood(source, targets, weights, kde_res, params.sigma, params.nstars, params.ntargets, DIMS, params.tol);

    for(i=0;i<params.ntargets;i++) { 
        if (kde_res[i] == 0.0) {
            printf("We have a zero kde value\n");
            break;
        }
    }
    printf("Final log-likelihood = %lg\n",tot);

    FREE(targets);
    FREE(source);
    FREE(weights);
    FREE(kde_res);
    return 0;

}


