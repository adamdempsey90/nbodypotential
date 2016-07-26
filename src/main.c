#include "potential.h"
#include <string.h>


real *targets, *source, *weights, *kde_res, *kde_work;

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
    params.ntargets = params.nstars;


    targets = (real *)malloc(sizeof(real)*params.ntargets*DIMS);
    NULLCHECK(targets,"targets");
    source = (real *)malloc(sizeof(real)*params.nstars*DIMS);
    NULLCHECK(source,"source");
    weights = (real *)malloc(sizeof(real)*params.nstars);
    NULLCHECK(weights,"weights");
    kde_res = (real *)malloc(sizeof(real)*params.ntargets);
    NULLCHECK(kde_res,"kde_res");
    kde_work = (real *)malloc(sizeof(real)*params.ntargets);
    NULLCHECK(kde_res,"kde_res");



    real pot_pars[2] = {atof(argv[2]), atof(argv[3])};

    int i;

    gpu_init();

    if (params.generate) {
        generate_system(pot_pars, source, FALSE);
        output_init(source,pot_pars);
    }
    else {
        FILE *f = fopen(params.targetfile,"r");
        fread(targets,sizeof(real),params.ntargets*DIMS,f);
        fclose(f);
        memcpy(source,targets,sizeof(real)*params.ntargets*DIMS);

        for(i=0;i<params.nstars;i++) {
            weights[i] = pow(2*M_PI,-DIMS*.5) / (real)params.nstars; 
        }

        for(i=0;i<params.ntargets;i++) { 
            kde_res[i] = 0;
            kde_work[i] = 0;
        }


        int status = minimize(pot_pars);
    }

    gpu_free();
    FREE(targets);
    FREE(source);
    FREE(weights);
    FREE(kde_res);
    FREE(kde_work);
    return 0;

}


