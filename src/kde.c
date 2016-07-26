#include <string.h>
#include "potential.h"
#include "figtree_consts.h"



void kde(double *source_points, double *target_points, double *q, double h, int n_source, int n_target, int dims, double tol, double *res) {
    /* Evaluate the phase space density in dims dimensions 
     * at the location of the target points given the distribution of 
     * source points.
     * This uses figtree to compute the sum over gaussians centered on the source points x_i
     * with weights q_i and bandwidth h for a single target point x_j:
     * G(x_j) = \sum_i q_i exp( -(x_j - x_i)^2 / h^2 ) 
     * We return the sum of the logs of the gauss transforms to compute the log likelihood:
     *  \ln L = \sum_j \log(G(x_j)).
     *
     * source_points: n_source x dims array of source points, (i.e the x_i points)
     * target_points: n_target x dims array of targer points, (i.e the x_j points)
     * q: weight for the i'th source point. 
     * h: bandwidth of the gaussians.
     * tol: Tolerance for the computation.
     * res: output array of length n_target.
     */
//    double *weights = (double *)malloc(sizeof(double)*n_source);

    


//    for(i=0;i<n_source;i++) {
//        
//        q[i] = pow(2*M_PI,-dims*.5) / (double)n_source; 
//
//
//    }


    //printf("Dims    : %d\nN Source: %d\nN Target: %d\nW       : %d\nh       : %lg\ntol     : %.2e\n",
//            dims,n_source,n_target,W,h,tol);

    figtree(dims, n_source, n_target, 1, source_points, h, q , target_points, tol, res, params.kdemethod,FIGTREE_PARAM_NON_UNIFORM); 

    return;

}

double log_likelihood(double *source_points, double *target_points, double *q, double *res, double h, int n_source, int n_target, int dims, double tol) {
    int i;
    kde(source_points, target_points, q, h, n_source, n_target, dims, tol,res);
    double tot = 0;

    //printf("Result\n");
    for(i=0;i<n_target;i++) {
        if ( (fabs(res[i]) <= 1e-16) || (res[i] == 0.0)) {
            res[i] = 1e-16;
        }
        tot -= log(res[i]);
    }

    return tot;
}


