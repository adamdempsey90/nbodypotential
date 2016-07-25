#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define real double
#define CHECK_ERROR(cmd) if(cudaStatus=cmd != cudaSuccess) printf("Erorr %s\n",cudaGetErrorString(cudaStatus)) 

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

__global__ void evolve(Particle *p, real tend, int n, real *params); 
void output(int time, Particle *p, int n) ;
__device__ void dy_potential(real x, real y, real *params, real *res) ;
__device__ void dx_potential(real x, real y, real *params, real *res) ;
__device__ void potential(real x, real y, real *params, real *res) ;
 void potential_cpu(real x, real y, real *params, real *res) ;
__device__ void leapfrog_step(Particle *p, real *params) ;
void set_particle_dt(Particle *p); 
void set_particle_ic(Particle *p,int n,real *params);
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
    
    int n = (int)sqrt(ntot);
    ntot = n*n;
    size = ntot*sizeof(Particle);

//    dim3 threadsPerBlock(8,8);
//    dim3 numBlocks(n/threadsPerBlock.x, n/threadsPerBlock.y);
    int threadsPerBlock=256;
    int blocksPerGrid = (ntot + threadsPerBlock-1)/threadsPerBlock;

    printf("Using NTOT=%d\tnt=%d\tq^2=%f\tR^2=%f\n",ntot,nt,params[0],params[1]);
    printf("Using %d blocksPerGrid, %d threadsPerBlock\n",threadsPerBlock,blocksPerGrid);

    Particle *particles;
    particles = (Particle *)malloc(size);

    Particle *particles_dev;
    cudaMalloc((void **)&particles_dev, size);
    

    int j;
    real dt = 1.;

    set_particle_ic(particles,n,params);
    output(0,particles,n);
    
    cudaMemcpy(particles_dev,particles,size,cudaMemcpyHostToDevice);
    for(j=1;j<nt;j++) {
        evolve<<<blocksPerGrid,threadsPerBlock>>>(particles_dev,j*dt,n,params);
        cudaStatus=cudaGetLastError();
        if (cudaStatus != cudaSuccess) printf("%s\n",cudaGetErrorString(cudaStatus));
        cudaMemcpy(&particles[0],particles_dev,size,cudaMemcpyDeviceToHost);
        output(j,particles,n);
    }
    free(particles);
    cudaFree(particles_dev);
    return 1;
}


__global__ void evolve(Particle *p, real tend,int n,real *params) {
    int i,j,indx;
    real t=0;

    real dxp, dyp, pot;
    real R2 = params[1];
    real q2 = params[0];

    indx = threadIdx.x + blockIdx.x * blockDim.x;
    //j = threadIdx.y + blockIdx.y * blockDim.y;
   // indx = j + i*n;
    
    //if ( i<n && j<n) {
     
    if (indx < n*n){
//            p[indx].vx = 10;
//      p[indx].dt  = 1;
    //    while (t <= tend) {
            p[indx].x += p[indx].vx*p[indx].dt*.5;
            p[indx].y += p[indx].vy*p[indx].dt*.5;
            dx_potential(p[indx].x,p[indx].y,params,&dxp);
            dy_potential(p[indx].x,p[indx].y,params,&dyp);
            dxp = 2*p[indx].x/(R2 + p[indx].x*p[indx].x + p[indx].y*p[indx].y/q2); 
            dyp = 2*p[indx].y/(q2*(R2 + p[indx].x*p[indx].x) + p[indx].y*p[indx].y); 
            p[indx].vx += -p[indx].dt*dxp;
            p[indx].vy += -p[indx].dt*dyp;
            p[indx].x += p[indx].vx*p[indx].dt*.5;
            p[indx].y += p[indx].vy*p[indx].dt*.5;
            potential(p[indx].x,p[indx].y,params,&pot);
            pot = log(R2 + p[indx].x*p[indx].x + p[indx].y*p[indx].y/q2); 
            p[indx].energy = .5*(p[indx].vx*p[indx].vx +p[indx].vy*p[indx].vy) + pot;
            t += p[indx].dt;
      //  }
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

    *res = log(x*x + y*y/q2 + R2);
    return;
}
 void potential_cpu(real x, real y, real *params,real *res) {
    real q2 = params[0];
    real R2 = params[1];

    *res = log(x*x + y*y/q2 + R2);
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
void set_particle_ic(Particle *p,int n,real *params) {

    int i,j;

    for(i=0;i<n;i++) {
        for(j=0;j<n;j++) {
            p[j+i*n].x = -6.5 + i*10.0/n;
            p[j+i*n].y = -6.5 + j*10.0/n;
            set_energy_cpu(&p[j+i*n],params);
            set_particle_dt(&p[j+i*n]);
        }
    }
    return;

}

void set_particle_dt(Particle *p) {
    p->dt = .1;
    return;
}

void output(int time, Particle *p,int n) {
    FILE *f;
    int i;
    char fname[100];
    sprintf(fname,"outputs/particles_%d.dat",time);
    f = fopen(fname,"w");

    for(i=0;i<n*n;i++) {
        fwrite(&p[i].x,sizeof(real),1,f);
        fwrite(&p[i].y,sizeof(real),1,f);
        fwrite(&p[i].z,sizeof(real),1,f);
        fwrite(&p[i].vx,sizeof(real),1,f);
        fwrite(&p[i].vy,sizeof(real),1,f);
        fwrite(&p[i].vz,sizeof(real),1,f);
        fwrite(&p[i].energy,sizeof(real),1,f);
//    fprintf(f,"%d\t%lg\t%lg\t%lg\t%lg\t%lg\t%lg\n",
//                i,p[i].x,p[i].y,p[i].z,p[i].vx,p[i].vy,p[i].vz);
    }
    fclose(f);
    return;

}
