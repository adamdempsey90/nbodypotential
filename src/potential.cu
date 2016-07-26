#include "potential.h"
#include <time.h>

#define GPUERRCHK(cmd) { gpuAssert((cmd), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

typedef struct Particle {
    real x;
    real y;
    real z;
    real vx;
    real vy;
    real vz;
    real dt;
    real energy;
} Particle;

__global__ void evolve(real *x, real *y, real *vx, real *vy, real tend,int n,real *params);
__device__ void dy_potential(real x, real y, real *params, real *res) ;
__device__ void dx_potential(real x, real y, real *params, real *res) ;
__device__ void potential(real x, real y, real *params, real *res) ;
 void potential_cpu(real x, real y, real *params, real *res) ;
__device__ void leapfrog_step(Particle *p, real *params) ;
void set_particle_dt(Particle *p); 
void set_particle_ic(real *x, real *y,real *vx, real *vy, int n);
__device__ void set_energy(Particle *p,real *params);
 void set_energy_cpu(Particle *p,real *params);
void read_evaluation_points(char *fname, real *points, int n);
void copy_backward(real *p);
void copy_forward(real *p);
void gpu_init(void);
void gpu_free(void);


real *h_x, *h_y, *h_vx, *h_vy;
real *xi, *yi, *vxi, *vyi;
real *d_x, *d_y, *d_vx, *d_vy;
real *params_dev;



void gpu_init(void) {
    cudaError_t cudaStatus;
    size_t size; 

    int ntot = params.nstars;
    int nt = params.nt; 
    real dt = params.dt;
    int threadsPerBlock= params.threads_per_block;
    int n = params.n;
    size = params.nstars*sizeof(real);

    h_x = (real *)malloc(size);
    NULLCHECK(h_x,"h_x");
    h_y = (real *)malloc(size);
    NULLCHECK(h_y,"h_y");
    h_vx = (real *)malloc(size);
    NULLCHECK(h_vx,"h_vx");
    h_vy = (real *)malloc(size);
    NULLCHECK(h_vy,"h_vy");


    GPUERRCHK( cudaMalloc(&d_x,size) );
    GPUERRCHK( cudaMalloc(&d_y,size) );
    GPUERRCHK( cudaMalloc(&d_vx,size) );
    GPUERRCHK( cudaMalloc(&d_vy,size) );


    GPUERRCHK( cudaMalloc(&params_dev,sizeof(real)*params.npars) );
    return;

}

real gpu_evolve(const real *pot_pars, real *points, const real *ic, real *kde_tot, int silent) {
    int i,k;
    cudaError_t cudaStatus;
    size_t size; 

    int ntot = params.nstars;
    int nt = params.nt; 
    real dt = params.dt;
    int threadsPerBlock= params.threads_per_block;
    int n = params.n;

    size = params.nstars*sizeof(real);

    int blocksPerGrid = (ntot + threadsPerBlock-1)/threadsPerBlock;


    if (!silent) {
        printf("Using NTOT=%.4e\tnt=%d\tq^2=%f\tR^2=%f\n",(real)ntot,nt,pot_pars[0],pot_pars[1]);
        printf("Using %d blocksPerGrid, %d threadsPerBlock\n",threadsPerBlock,blocksPerGrid);
    }



    //Do kernel activity here




    GPUERRCHK( cudaMemcpy(params_dev,pot_pars,sizeof(real)*params.npars,cudaMemcpyHostToDevice) );

    if (!silent) {
        printf("Potential Parameters: %lg\t%lg\n",pot_pars[0],pot_pars[1]);
    }


    int j;
/*
    printf("Setting ics\n");
    set_particle_ic(xi,yi,vxi,vyi,n,params);
    output(0,xi,yi,vxi,vyi,n);
    printf("allocating\n");
*/  
/*
    for(i=0;i<ntot;i++) {
        h_x[i] =  points[0 + i*DIMS];
        h_y[i] =  points[1 + i*DIMS];
        h_vx[i] = points[2 + i*DIMS];
        h_vy[i] = points[3 + i*DIMS];
    }
*/
//    output(0,h_x,h_y,h_vx,h_vy,n);

    memcpy(points,ic,sizeof(real)*ntot*DIMS);

    for(i=0;i<ntot;i++) kde_tot[i] = 0.0;  // Initial points



    copy_forward(points);

    GPUERRCHK( cudaMemcpy(d_x,h_x, size,cudaMemcpyHostToDevice) );
    GPUERRCHK( cudaMemcpy(d_y,h_y, size,cudaMemcpyHostToDevice) );
    GPUERRCHK( cudaMemcpy(d_vx,h_vx, size,cudaMemcpyHostToDevice) );
    GPUERRCHK( cudaMemcpy(d_vy,h_vy, size,cudaMemcpyHostToDevice) );

    int kind;
//    srand(seed);
    for(j=1;j<nt;j++) {
//        printf("Starting time %d\n",j);
/*
        cudaEventCreate(&start);
        cudaEventRecord(start,0);
*/
        evolve<<<blocksPerGrid,threadsPerBlock>>>(d_x,d_y,d_vx,d_vy,j*dt,n,params_dev);
/*
        if (j > jstart) {
            for(k=0;k<nevals;k++) {
                
                cudaThreadSynchronize();
                kde<<<blocksPerGrid,threadsPerBlock>>>(d_x,d_y,d_vx,d_vy,
                                        points[0 + k*4], points[1 +4*k], points[2+4*k], points[3+4*k], sigma_dev,n,fac,d_kde);
            }
        }
*/
/*
        cudaEventCreate(&stop);
        cudaEventRecord(stop,0);
        cudaEventSynchronize(stop);

        cudaEventElapsedTime(&elapsedTime, start,stop);
        printf("Elapsed time : %.4e ms\n" ,elapsedTime);
    */
    
        GPUERRCHK( cudaPeekAtLastError() );
        GPUERRCHK( cudaMemcpy(h_x,d_x, size,cudaMemcpyDeviceToHost) );
        GPUERRCHK( cudaMemcpy(h_y,d_y, size,cudaMemcpyDeviceToHost) );
        GPUERRCHK( cudaMemcpy(h_vx,d_vx, size,cudaMemcpyDeviceToHost) );
        GPUERRCHK( cudaMemcpy(h_vy,d_vy, size,cudaMemcpyDeviceToHost) );

        copy_backward(points);
        add_kde(points, kde_tot);

        //ll += log_likelihood(points, targets, weights, kde_res, params.sigma, params.nstars, params.ntargets, DIMS, params.tol)/dt;


        //output(j,h_x,h_y,h_vx,h_vy,n);
        if (!silent) {
            printf("##########--%3.2f%%--##########\r",100*(real)(j+1)/nt);
            fflush(stdout);
        }
    }
    real ll = 0;
#ifdef _OPENMP
#pragma omp parallel for reduction(ll:-)
#endif
    for(i=0;i<ntot;i++) {
        ll -= log( kde_tot[i] / (real) nt);
        
    }


    return ll;
}

void gpu_free(void) {
    FREE(h_x);
    FREE(h_y);
    FREE(h_vx);
    FREE(h_vy);
    GPUERRCHK( cudaFree(d_x) );
    GPUERRCHK( cudaFree(d_y) );
    GPUERRCHK( cudaFree(d_vx) );
    GPUERRCHK( cudaFree(d_vy) );
    GPUERRCHK( cudaFree(params_dev) );

    return;
}
void generate_system(const real *pot_pars, real *points, int silent) {
    int i,k;
    cudaError_t cudaStatus;
    size_t size; 

    int ntot = params.nstars;
    int nt = params.nt; 
    real dt = params.dt;
    int threadsPerBlock= params.threads_per_block;
    int n = params.n;

    size = params.nstars*sizeof(real);

    int blocksPerGrid = (ntot + threadsPerBlock-1)/threadsPerBlock;


    if (!silent) {
        printf("Using NTOT=%.4e\tnt=%d\tq^2=%f\tR^2=%f\n",(real)ntot,nt,pot_pars[0],pot_pars[1]);
        printf("Using %d blocksPerGrid, %d threadsPerBlock\n",threadsPerBlock,blocksPerGrid);
    }



    GPUERRCHK( cudaMemcpy(params_dev,pot_pars,sizeof(real)*params.npars,cudaMemcpyHostToDevice) );

    if (!silent) {
        printf("Potential Parameters: %lg\t%lg\n",pot_pars[0],pot_pars[1]);
    }


    int j;

    set_particle_ic(h_x,h_y,h_vx,h_vy,n);
  
    GPUERRCHK( cudaMemcpy(d_x,h_x, size,cudaMemcpyHostToDevice) );
    GPUERRCHK( cudaMemcpy(d_y,h_y, size,cudaMemcpyHostToDevice) );
    GPUERRCHK( cudaMemcpy(d_vx,h_vx, size,cudaMemcpyHostToDevice) );
    GPUERRCHK( cudaMemcpy(d_vy,h_vy, size,cudaMemcpyHostToDevice) );

    evolve<<<blocksPerGrid,threadsPerBlock>>>(d_x,d_y,d_vx,d_vy,nt*dt,n,params_dev);

    GPUERRCHK( cudaPeekAtLastError() );
    GPUERRCHK( cudaMemcpy(h_x,d_x, size,cudaMemcpyDeviceToHost) );
    GPUERRCHK( cudaMemcpy(h_y,d_y, size,cudaMemcpyDeviceToHost) );
    GPUERRCHK( cudaMemcpy(h_vx,d_vx, size,cudaMemcpyDeviceToHost) );
    GPUERRCHK( cudaMemcpy(h_vy,d_vy, size,cudaMemcpyDeviceToHost) );

    copy_backward(points);
    
    return;
}


__global__ void evolve(real *x, real *y, real *vx, real *vy, real tend,int n,real *params) {
    int i,j,indx;
    real t=0;

    real dxp, dyp, pot;
    real R2 = params[1];
    real q2 = params[0];
    real dt=.1;

    i = threadIdx.x + blockIdx.x * blockDim.x;
    //j = threadIdx.y + blockIdx.y * blockDim.y;
   // indx = j + i*n;
    
    //if ( i<n && j<n) {
     
    if (i < n*n){
//            p[indx].vx = 10;
//      p[indx].dt  = 1;
        while (t <= tend) {
            x[i] += vx[i]*dt*.5;
            y[i] += vy[i]*dt*.5;
      //      dx_potential(x[i],y[i],params,&dxp);
     //       dy_potential(x[i],y[i],params,&dyp);
            vx[i] += -dt *  2*x[i]/(R2 + x[i]*x[i] + y[i]*y[i]/q2); 
            vy[i] += -dt * 2*y[i]/(q2*(R2 + x[i]*x[i]) + y[i]*y[i]); 
            //vx[i] += -dt*dxp;
            //vy[i] += -dt*dyp;
            x[i] += vx[i]*dt*.5;
            y[i] += vy[i]*dt*.5;
   //         potential(x[i],y[i],params,&pot);
            pot = (real)log(R2 + x[i]*x[i] + y[i]*y[i]/q2); 
            t += dt;
        }
    }
    
    return;
}

__device__ void set_energy(Particle *p,real *params) {
    real res;
    potential(p->x,p->y,params,&res);
    p->energy = .5*(p->vx*p->vx + p->vy*p->vy + p->vz*p->vz);
    p->energy += res;
    return;
}
void set_energy_cpu(Particle *p,real *params) {
    real res;
    potential_cpu(p->x,p->y,params,&res);
    p->energy = .5*(p->vx*p->vx + p->vy*p->vy + p->vz*p->vz);
    p->energy += res;
    return;
}
__device__ void leapfrog_step(Particle *p,real *params) {

    real dt = p->dt;
    real x = p->x;
    real y = p->y;
    real vx = p->vx;
    real vy = p->vy;
    real dxp, dyp;

    x += vx*dt*.5;
    y += vy*dt*.5;
    dx_potential(x,y,params,&dxp);
    dy_potential(x,y,params,&dyp);

    vx += -dt*dxp;
    vy += -dt*dyp; 
    x += vx*dt*.5;
    y += vy*dt*.5;
    p->x = x;
    p->y = y;
    p->vx = vx;
    p->vy = vy;

    return;
}

__device__ void potential(real x, real y, real *params,real *res) {
    real q2 = params[0];
    real R2 = params[1];

    *res = (real)log(x*x + y*y/q2 + R2);
    return;
}
 void potential_cpu(real x, real y, real *params,real *res) {
    real q2 = params[0];
    real R2 = params[1];

    *res = (real)log(x*x + y*y/q2 + R2);
    return;
}
__device__ void dx_potential(real x, real y, real *params,real *res) {
    real q2 = params[0];
    real R2 = params[1];
    *res = 2*x/(R2 + x*x + y*y/q2); 
    return;
}
__device__ void dy_potential(real x, real y, real *params,real *res) {
    real q2 = params[0];
    real R2 = params[1];

    *res =  2*y/( q2*(R2 + x*x) + y*y); 
    return;
}
void set_particle_ic(real *x, real *y,real *vx, real *vy, int n) {

    int i,j;

    for(i=0;i<n;i++) {
        for(j=0;j<n;j++) {
            x[j+i*n] = -6.5 + i*10.0/n;
            y[j+i*n] = -6.5 + j*10.0/n;
            vx[j+i*n] = 0;
            vy[j+i*n] = 0;
        }
    }
    return;

}

void set_particle_dt(Particle *p) {
    p->dt = .1;
    return;
}

void copy_forward(real *p) {
    int i;
    for(i=0;i<params.ntargets;i++) {
        h_x[i]  = p[0 + i*DIMS];
        h_y[i]  = p[1 + i*DIMS];
        h_vx[i] = p[2 + i*DIMS];
        h_vy[i] = p[3 + i*DIMS];
    }
    return;
}
void copy_backward(real *p) {
    int i;
    for(i=0;i<params.ntargets;i++) {
        p[0 + i*DIMS]=h_x[i] ; 
        p[1 + i*DIMS]=h_y[i]  ;
        p[2 + i*DIMS]=h_vx[i] ;
        p[3 + i*DIMS]=h_vy[i] ;
    }
    return;
}
