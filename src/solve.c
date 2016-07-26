#include "potential.h"
#include "gsl/gsl_multimin.h"
#define MAXITERATIONS 1000

extern real *targets, *source, *weights, *kde_res, *kde_work;

double func(const gsl_vector *x, void *p) {
    int i;
    int npars = params.npars;
    double pot_pars[npars];

    for(i=0;i<npars;i++) {
        pot_pars[i] = gsl_vector_get(x,i);
    }
    return gpu_evolve(pot_pars, source, targets, kde_res, TRUE);

}

int minimize(double *init_pars) {
    int i;
    int npars = params.npars;
    gsl_vector *x, *ss;
    x = gsl_vector_alloc(npars);
    for(i=0;i<npars;i++) gsl_vector_set(x, i, init_pars[i]);


    const gsl_multimin_fminimizer_type *T = gsl_multimin_fminimizer_nmsimplex2;
    gsl_multimin_fminimizer *s = NULL;

    gsl_multimin_function my_f;

    my_f.n = npars;
    my_f.f = &func;
    my_f.params = NULL;

    ss = gsl_vector_alloc(npars);
    gsl_vector_set_all(ss, params.simplex_step);

    s = gsl_multimin_fminimizer_alloc(T,npars);
    gsl_multimin_fminimizer_set(s, &my_f, x, ss);


    size_t iter = 0;
    int status;
    double size;


    do {
        iter++;
        status = gsl_multimin_fminimizer_iterate(s);
        
        if (status) {
            break;
        }
        size = gsl_multimin_fminimizer_size(s);
        status = gsl_multimin_test_size(size, params.size);

        if (status == GSL_SUCCESS) {
            printf("Minimum found at:\n");
        }
        printf("%5d\t",iter);
        for(i=0;i<npars;i++) {
            printf("%10.3e\t",gsl_vector_get(s->x, i));
        }
        printf("-log(L) = %.8e\tsize = %.4e\n",s->fval,size);

    }
    while (status == GSL_CONTINUE && iter < MAXITERATIONS);


    for(i=0;i<npars;i++) init_pars[i] = gsl_vector_get(s->x, i);

    gsl_vector_free(x);
    gsl_vector_free(ss);
    gsl_multimin_fminimizer_free(s);

    return status;
}
