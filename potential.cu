#include <stdio.h>
#include <stdlib.h>
#include <math.h>


#define CHECK_ERROR(cmd) if(cudaStatus=cmd != cudaSuccess) printf("Erorr %s\n",cudaGetErrorString(cudaStatus)) 
#define real double

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
void output(int time, real *x, real *y, real *vx, real *vy,int n);
__device__ void dy_potential(real x, real y, real *params, real *res) ;
__device__ void dx_potential(real x, real y, real *params, real *res) ;
__device__ void potential(real x, real y, real *params, real *res) ;
 void potential_cpu(real x, real y, real *params, real *res) ;
__device__ void leapfrog_step(Particle *p, real *params) ;
void set_particle_dt(Particle *p); 
void set_particle_ic(real *x, real *y,real *vx,int n,real *params);
__device__ void set_energy(Particle *p,real *params);
 void set_energy_cpu(Particle *p,real *params);

int main(int argc, char *argv[]) {
    cudaError_t cudaStatus;
    size_t size; 
    real params[2]={1.0,1.0};
    int ntot, nt;
    ntot = atoi(argv[1]);
    nt = atoi(argv[2]);
    params[0] = atof(argv[3]);
    params[1] = atof(argv[4]);
    
    int n = (int)(real)sqrt(ntot);
    ntot = n*n;
    size = ntot*sizeof(real);

//    dim3 threadsPerBlock(8,8);
//    dim3 numBlocks(n/threadsPerBlock.x, n/threadsPerBlock.y);
    int threadsPerBlock=256;
    int blocksPerGrid = (ntot + threadsPerBlock-1)/threadsPerBlock;

    printf("Using NTOT=%d\tnt=%d\tq^2=%f\tR^2=%f\n",ntot,nt,params[0],params[1]);
    printf("Using %d blocksPerGrid, %d threadsPerBlock\n",threadsPerBlock,blocksPerGrid);

    real *h_x, *h_y, *h_vx, *h_vy;
    real *d_x, *d_y, *d_vx, *d_vy;
    
    h_x = (real *)malloc(sizeof(real)*ntot);
    h_y = (real *)malloc(sizeof(real)*ntot);
    h_vx = (real *)malloc(sizeof(real)*ntot);
    h_vy = (real *)malloc(sizeof(real)*ntot);

    cudaMalloc(&d_x,sizeof(real)*ntot);
    cudaMalloc(&d_y,sizeof(real)*ntot);
    cudaMalloc(&d_vx,sizeof(real)*ntot);
    cudaMalloc(&d_vy,sizeof(real)*ntot);

    real *params_dev;
    cudaMalloc(&params_dev,sizeof(real)*2);
    cudaMemcpy(params_dev,params,sizeof(real)*2,cudaMemcpyHostToDevice);

    int j;
    real dt = 1.;

    set_particle_ic(h_x,h_y,h_vx,n,params);
    output(0,h_x,h_y,h_vx,h_vy,n);
    
    cudaMemcpy(d_x,h_x, size,cudaMemcpyHostToDevice);
    cudaMemcpy(d_y,h_y, size,cudaMemcpyHostToDevice);
    cudaMemcpy(d_vx,h_vx, size,cudaMemcpyHostToDevice);
    cudaMemcpy(d_vy,h_vy, size,cudaMemcpyHostToDevice);
    for(j=1;j<nt;j++) {
        printf("Starting time %d\n",j);
        evolve<<<blocksPerGrid,threadsPerBlock>>>(d_x,d_y,d_vx,d_vy,j*dt,n,params_dev);
    cudaMemcpy(h_x,d_x, size,cudaMemcpyDeviceToHost);
    cudaMemcpy(h_y,d_y, size,cudaMemcpyDeviceToHost);
    cudaMemcpy(h_vx,d_vx, size,cudaMemcpyDeviceToHost);
    cudaMemcpy(h_vy,d_vy, size,cudaMemcpyDeviceToHost);
    output(j,h_x,h_y,h_vx,h_vy,n);
    }
    free(h_x);
    free(h_y);
    free(h_vx);
    free(h_vy);
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_vx);
    cudaFree(d_vy);
    return 1;
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
            dx_potential(x[i],y[i],params,&dxp);
            dy_potential(x[i],y[i],params,&dyp);
     //       dxp = 2*x[i]/(R2 + x[i]*x[i] + y[i]*y[i]/q2); 
     //       dyp = 2*y[i]/(q2*(R2 + x[i]*x[i]) + y[i]*y[i]); 
            vx[i] += -dt*dxp;
            vy[i] += -dt*dyp;
            x[i] += vx[i]*dt*.5;
            y[i] += vy[i]*dt*.5;
            potential(x[i],y[i],params,&pot);
   //         pot = (real)log(R2 + x[i]*x[i] + y[i]*y[i]/q2); 
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
void set_particle_ic(real *x, real *y,real *vx,int n,real *params) {

    int i,j;

    for(i=0;i<n;i++) {
        for(j=0;j<n;j++) {
            x[j+i*n] = -6.5 + i*10.0/n;
            y[j+i*n] = -6.5 + j*10.0/n;
        }
    }
    return;

}

void set_particle_dt(Particle *p) {
    p->dt = .1;
    return;
}

void output(int time, real *x, real *y, real *vx, real *vy,int n) {
    FILE *f;
    int i;
    char fname[100];
    sprintf(fname,"outputs/particles_%d.dat",time);
    f = fopen(fname,"w");

    for(i=0;i<n*n;i++) {
        fwrite(&x[i],sizeof(real),1,f);
        fwrite(&y[i],sizeof(real),1,f);
        fwrite(&vx[i],sizeof(real),1,f);
        fwrite(&vy[i],sizeof(real),1,f);
//    fprintf(f,"%d\t%lg\t%lg\t%lg\t%lg\t%lg\t%lg\n",
//                i,p[i].x,p[i].y,p[i].z,p[i].vx,p[i].vy,p[i].vz);
    }
    fclose(f);
    return;

}
