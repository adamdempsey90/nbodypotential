#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "figtree_consts.h"


void kde(int dims, int n_source, int n_target, int W, double *source_points, double h, double *q , double *target_points, double tol, double *res) {
    int i,j,k;
    double h2 = h*h;
    double temp,dist;
    for(j=0;j<n_target;j++) {

        temp =  0;
        for(i=0;i<n_source;i++) {
            dist = 0;
            for(k=0;k<dims;k++) {
                dist += pow(source_points[i*dims + k] - target_points[j*dims + k],2);
            }
            temp += q[i]*exp(- dist/h2); 
        }
        res[j] = temp; ///(double)n_source;

    }

    return;

}


int main(void) {
  int d = 7;

  // The number of targets (vectors at which gauss transform is evaluated).
  int M = 10;

  // The number of sources which will be used for the gauss transform. 
  int N = 20;

  // The bandwidth.  NOTE: this is not the same as standard deviation since 
  // the Gauss Transform sums terms exp( -||x_i - y_j||^2 / h^2 ) as opposed
  // to  exp( -||x_i - y_j||^2 / (2*sigma^2) ).  Thus, if sigma is known, 
  // bandwidth can be set to h = sqrt(2)*sigma.
  double h = .8;
  
  // Desired maximum absolute error after normalizing output by sum of weights.
  // If the weights, q_i (see below), add up to 1, then this is will be the 
  // maximum absolute error.
  // The smaller epsilon is, the more accurate the results will be, at the
  // expense of increased computational complexity.
  double epsilon = 1e-8;

  // The source array.  It is a contiguous array, where
  // ( x[i*d], x[i*d+1], ..., x[i*d+d-1] ) is the ith d-dimensional sample.
  // For example, below N = 20 and d = 7, so there are 20 rows, each
  // a 7-dimensional sample.
  double x[140] = {0.7165, 0.5113, 0.7764, 0.4893, 0.1859, 0.7006, 0.9827,
                0.8066, 0.7036, 0.4850, 0.1146, 0.6649, 0.3654, 0.1400,
                0.5668, 0.8230, 0.6739, 0.9994, 0.9616, 0.0589, 0.3603,
                0.5485, 0.2618, 0.5973, 0.0493, 0.5711, 0.7009, 0.9623,
                0.7505, 0.7400, 0.4319, 0.6343, 0.8030, 0.0839, 0.9455,
                0.9159, 0.6020, 0.2536, 0.8735, 0.5134, 0.7327, 0.4222,
                0.1959, 0.1059, 0.3923, 0.1003, 0.6930, 0.2069, 0.6094,
                0.1059, 0.0396, 0.2093, 0.9693, 0.1059, 0.3029, 0.3069,
                0.9692, 0.6029, 0.2222, 0.2059, 0.3059, 0.6092, 0.2133,
                0.9614, 0.0721, 0.5534, 0.2920, 0.8580, 0.3358, 0.6802,
                0.2473, 0.3527, 0.1879, 0.4906, 0.4093, 0.4635, 0.6109,
                0.1865, 0.0395, 0.5921, 0.1853, 0.9963, 0.1953, 0.7659,
                0.0534, 0.3567, 0.4983, 0.4344, 0.5625, 0.6166, 0.1133,
                0.8983, 0.7546, 0.7911, 0.8150, 0.6700, 0.2009, 0.2731,
                0.6262, 0.5369, 0.0595, 0.0890, 0.2713, 0.4091, 0.4740,
                0.1332, 0.6926, 0.0009, 0.1532, 0.9632, 0.3521, 0.9692,
                0.9623, 0.3532, 0.7432, 0.0693, 0.2336, 0.6022, 0.2936,
                0.3921, 0.6023, 0.6323, 0.9353, 0.3963, 0.2835, 0.9868,
                0.2362, 0.6682, 0.2026, 0.0263, 0.1632, 0.9164, 0.1153,
                0.9090, 0.5962, 0.3290, 0.4782, 0.5972, 0.1614, 0.8295 };

  // The target array.  It is a contiguous array, where
  // ( y[j*d], y[j*d+1], ..., y[j*d+d-1] ) is the jth d-dimensional sample.
  // For example, below M = 10 and d = 7, so there are 10 rows, each
  // a 7-dimensional sample.
  double y[70] = {0.9561, 0.5955, 0.0287, 0.8121, 0.6101, 0.7015, 0.0922,
                0.4249, 0.3756, 0.1662, 0.8332, 0.8386, 0.4516, 0.9566,
                0.1472, 0.8699, 0.7694, 0.4442, 0.6206, 0.9517, 0.6400,
                0.0712, 0.3143, 0.6084, 0.1750, 0.6210, 0.2460, 0.5874,
                0.5061, 0.4648, 0.5414, 0.9423, 0.3418, 0.4018, 0.3077,
                0.4116, 0.2859, 0.3941, 0.5030, 0.7220, 0.3062, 0.1122,
                0.4433, 0.4668, 0.0147, 0.6641, 0.7241, 0.2816, 0.2618,
                0.7085, 0.7839, 0.9862, 0.4733, 0.9028, 0.4511, 0.8045,
                0.8289, 0.1663, 0.3939, 0.5208, 0.7181, 0.5692, 0.4608,
                0.4453, 0.0877, 0.4435, 0.3663, 0.3025, 0.8518, 0.7595 };

  //double q[20] = {0.2280, 0.4496, 0.1722, 0.9688, 0.3557, 0.0490, 0.7553, 0.8948, 0.2861, 0.2512, 0.9327, 0.3353, 0.2530, 0.2532, 0.3352, 0.7235, 0.2506, 0.0235, 0.1062, 0.1061}; //, 0.7234, 0.1532};
  double q[20] = {1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0};

  int W = 1;  // in this case W = 1.

  double *g_auto         = (double *)malloc(sizeof(double)*W*M);
  double *g_sf           = (double *)malloc(sizeof(double)*W*M);
  double *g_sf_tree      = (double *)malloc(sizeof(double)*W*M);
  double *g_ifgt_u       = (double *)malloc(sizeof(double)*W*M);
  double *g_ifgt_tree_u  = (double *)malloc(sizeof(double)*W*M);
  double *g_ifgt_nu      = (double *)malloc(sizeof(double)*W*M);
  double *g_ifgt_tree_nu = (double *)malloc(sizeof(double)*W*M);
  double *g_me = (double *)malloc(sizeof(double)*W*M);
  
  // initialize all output arrays to zero
    int i; 
  for(i=0;i<W*M;i++) {
      g_auto[i] = 0;       
      g_sf[i] = 0;          
      g_sf_tree[i] = 0;     
      g_ifgt_u[i] = 0;      
      g_ifgt_tree_u[i] = 0; 
      g_ifgt_nu[i] = 0;     
      g_ifgt_tree_nu[i] = 0;
  }

  figtree( d, N, M, W, x, h, q, y, epsilon, g_auto );

  // evaluate gauss transform using direct (slow) method
  figtree( d, N, M, W, x, h, q, y, epsilon, g_sf, FIGTREE_EVAL_DIRECT );

  // evaluate gauss transform using direct method with approximate nearest neighbors
  figtree( d, N, M, W, x, h, q, y, epsilon, g_sf_tree, FIGTREE_EVAL_DIRECT_TREE );

  // evaluate gauss transform using FIGTREE (truncated series), estimating parameters with and without
  //   the assumption that sources are uniformly distributed
  figtree( d, N, M, W, x, h, q, y, epsilon, g_ifgt_u, FIGTREE_EVAL_IFGT, FIGTREE_PARAM_UNIFORM, 1 );
  figtree( d, N, M, W, x, h, q, y, epsilon, g_ifgt_nu, FIGTREE_EVAL_IFGT, FIGTREE_PARAM_NON_UNIFORM, 1 );

  // evaluate gauss transform using FIGTREE (truncated series), estimating parameters with and without
  //   the assumption that sources are uniformly distributed
  figtree( d, N, M, W, x, h, q, y, epsilon, g_ifgt_tree_u, FIGTREE_EVAL_IFGT_TREE, FIGTREE_PARAM_UNIFORM, 1 );
  figtree( d, N, M, W, x, h, q, y, epsilon, g_ifgt_tree_nu, FIGTREE_EVAL_IFGT_TREE, FIGTREE_PARAM_NON_UNIFORM, 1 );


  kde(d, N, M, W, x, h , q, y, epsilon, g_me);

  // compute absolute error of the Gauss Transform at each target and for all sets of weights.
  for( i = 0; i < W*M; i++) {
    g_auto[i]         = fabs(g_auto[i]        -g_sf[i]);
    g_sf_tree[i]      = fabs(g_sf_tree[i]     -g_sf[i]);
    g_ifgt_u[i]       = fabs(g_ifgt_u[i]      -g_sf[i]);
    g_ifgt_nu[i]      = fabs(g_ifgt_nu[i]     -g_sf[i]);
    g_ifgt_tree_u[i]  = fabs(g_ifgt_tree_u[i] -g_sf[i]);
    g_ifgt_tree_nu[i] = fabs(g_ifgt_tree_nu[i]-g_sf[i]);
    g_me[i]           = fabs(g_me[i]        -g_sf[i]);
  }

  // print out results for all six ways to evaluate
  printf("Results:\n");
  printf("Direct result auto error    sf-tree error ifgt-u error  ifgt-nu error  ifgt-tree-u e ifgt-tree-nu error my error\n");
  for( i = 0; i < W*M; i++ )
    printf("%13.4f %13.4e %13.4e %13.4e %13.4e %13.4e %13.4e %13.4e\n", 
           g_sf[i], g_auto[i], g_sf_tree[i], g_ifgt_u[i], g_ifgt_nu[i], g_ifgt_tree_u[i], g_ifgt_tree_nu[i], g_me[i] );
  printf("\n");

  // deallocate memory
  free(g_auto);
  free(g_sf);
  free(g_sf_tree);
  free(g_ifgt_u);
  free(g_ifgt_nu);
  free(g_ifgt_tree_u);
  free(g_ifgt_tree_nu);
  free(g_me);
  return 0;
}
