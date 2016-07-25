#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>


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
__global__ void kde(real *xi, real *yi, real *vxi, real *vyi, 
                real px, real py, real pvx, real pvy,
                real *sigma, int n, real fac, real *res);
void output(int time, real *x, real *y, real *vx, real *vy,int n);
__device__ void dy_potential(real x, real y, real *params, real *res) ;
__device__ void dx_potential(real x, real y, real *params, real *res) ;
__device__ void potential(real x, real y, real *params, real *res) ;
 void potential_cpu(real x, real y, real *params, real *res) ;
__device__ void leapfrog_step(Particle *p, real *params) ;
void set_particle_dt(Particle *p); 
void set_particle_ic(real *x, real *y,real *vx, real *vy, int n,real *params);
__device__ void set_energy(Particle *p,real *params);
 void set_energy_cpu(Particle *p,real *params);
void read_evaluation_points(char *fname, real *points, int n);

int main(int argc, char *argv[]) {
    int i,k;
    cudaError_t cudaStatus;
    size_t size; 
    real params[2]={1.0,1.0};
    int ntot, nt,nevals;
    ntot = atoi(argv[1]);
    nt = atoi(argv[2]);
    nevals = atoi(argv[3]);
    int seed = atoi(argv[4]);
    int threadsPerBlock= atoi(argv[5]);
    params[0] = atof(argv[6]);
    params[1] = atof(argv[7]);
    
    int n = (int)(real)sqrt(ntot);
    ntot = n*n;
    size = ntot*sizeof(real);

//    dim3 threadsPerBlock(8,8);
//    dim3 numBlocks(n/threadsPerBlock.x, n/threadsPerBlock.y);
    int blocksPerGrid = (ntot + threadsPerBlock-1)/threadsPerBlock;

    printf("Using NTOT=%.4e\tnt=%d\tq^2=%f\tR^2=%f\n",(real)ntot,nt,params[0],params[1]);
    printf("Using %d blocksPerGrid, %d threadsPerBlock\n",threadsPerBlock,blocksPerGrid);

    cudaEvent_t start, stop;
    float elapsedTime;


    //Do kernel activity here

    printf("malloc\n");
    real *points = (real *)malloc(sizeof(real)*nevals*4);
    FILE *f = fopen("ic/points.dat","r");
    fread(points,sizeof(real),nevals*4,f);
    fclose(f);


    real *h_x, *h_y, *h_vx, *h_vy, *h_kde;
    real *xi, *yi, *vxi, *vyi;
    real *d_x, *d_y, *d_vx, *d_vy, *d_kde;

    
    h_x = (real *)malloc(size);
    h_y = (real *)malloc(size);
    h_vx = (real *)malloc(size);
    h_vy = (real *)malloc(size);
    xi = (real *)malloc(size);
    yi = (real *)malloc(size);
    vxi = (real *)malloc(size);
    vyi = (real *)malloc(size);
    h_kde = (real *)malloc(size);

    for(i=0;i<ntot;i++) h_kde[i] = 0;


    cudaMalloc(&d_x,size);
    cudaMalloc(&d_y,size);
    cudaMalloc(&d_vx,size);
    cudaMalloc(&d_vy,size);

    cudaMalloc(&d_kde,size);

    real *params_dev;
    cudaMalloc(&params_dev,sizeof(real)*2);
    cudaMemcpy(params_dev,params,sizeof(real)*2,cudaMemcpyHostToDevice);

    real sigma[4] = {2.0, 3.0, 1.0, 4.0};
    real *sigma_dev;
    cudaMalloc(&sigma_dev,sizeof(real)*4);
    cudaMemcpy(sigma_dev,&sigma[0],sizeof(real)*4,cudaMemcpyHostToDevice);


    int j;
    real dt = 1.;
    real fac = 1.0/(real)(nevals*nt);

    printf("Setting ics\n");
    set_particle_ic(xi,yi,vxi,vyi,n,params);
    output(0,xi,yi,vxi,vyi,n);
    printf("allocating\n");
    
    memcpy(h_x,xi,sizeof(real)*ntot);
    memcpy(h_y,yi,sizeof(real)*ntot);
    memcpy(h_vx,vxi,sizeof(real)*ntot);
    memcpy(h_vy,vyi,sizeof(real)*ntot);
    cudaMemcpy(d_x,h_x, size,cudaMemcpyHostToDevice);
    cudaMemcpy(d_y,h_y, size,cudaMemcpyHostToDevice);
    cudaMemcpy(d_vx,h_vx, size,cudaMemcpyHostToDevice);
    cudaMemcpy(d_vy,h_vy, size,cudaMemcpyHostToDevice);
    cudaMemcpy(d_kde,h_kde, size,cudaMemcpyHostToDevice);

    int kind;
    srand(seed);
    for(j=1;j<nt;j++) {
        printf("Starting time %d\n",j);
/*
        cudaEventCreate(&start);
        cudaEventRecord(start,0);
*/
        evolve<<<blocksPerGrid,threadsPerBlock>>>(d_x,d_y,d_vx,d_vy,j*dt,n,params_dev);
        for(k=0;k<nevals;k++) {
        /*
            kind = k; //rand() % ntot;
            point[0] = xi[kind];
            point[1] = yi[kind];
            point[2] = vxi[kind];
            point[3] = vyi[kind];
        */
            
            cudaThreadSynchronize();
            kde<<<blocksPerGrid,threadsPerBlock>>>(d_x,d_y,d_vx,d_vy,
                                    points[0 + k*4], points[1 +4*k], points[2+4*k], points[3+4*k], sigma_dev,n,fac,d_kde);
        }

/*
        cudaEventCreate(&stop);
        cudaEventRecord(stop,0);
        cudaEventSynchronize(stop);

        cudaEventElapsedTime(&elapsedTime, start,stop);
        printf("Elapsed time : %.4e ms\n" ,elapsedTime);
    */
    

        cudaMemcpy(h_x,d_x, size,cudaMemcpyDeviceToHost);
        cudaMemcpy(h_y,d_y, size,cudaMemcpyDeviceToHost);
        cudaMemcpy(h_vx,d_vx, size,cudaMemcpyDeviceToHost);
        cudaMemcpy(h_vy,d_vy, size,cudaMemcpyDeviceToHost);
        output(j,h_x,h_y,h_vx,h_vy,n);
    }
    cudaMemcpy(h_kde,d_kde, size,cudaMemcpyDeviceToHost);

    real tot = 0;
    for(i=0;i<nevals;i++) {
        tot += log(h_kde[i]);
    }
    printf("Log liklihood %.16f\n", tot);

    free(h_x);
    free(h_y);
    free(h_vx);
    free(h_vy);
    free(xi);
    free(yi);
    free(vxi);
    free(vyi);
    free(h_kde);
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_vx);
    cudaFree(d_vy);
    cudaFree(d_kde);
    cudaFree(sigma_dev);
    cudaFree(params_dev);

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
      //      dx_potential(x[i],y[i],params,&dxp);
     //       dy_potential(x[i],y[i],params,&dyp);
            dxp = 2*x[i]/(R2 + x[i]*x[i] + y[i]*y[i]/q2); 
            dyp = 2*y[i]/(q2*(R2 + x[i]*x[i]) + y[i]*y[i]); 
            vx[i] += -dt*dxp;
            vy[i] += -dt*dyp;
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
void set_particle_ic(real *x, real *y,real *vx, real *vy, int n,real *params) {

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

__global__ void kde(real *xi, real *yi, real *vxi, real *vyi, 
                real px, real py, real pvx, real pvy,
                real *sigma, int n, real fac, real *res) {
    int i;

    real dist;
    real norm = fac*pow(2*M_PI,-3.0); 
    i = threadIdx.x + blockIdx.x * blockDim.x;
     
    if (i < n*n){
        dist =  pow((px - xi[i])/sigma[0],2);
        dist += pow((py - yi[i])/sigma[1],2);
        dist += pow((pvx - vxi[i])/sigma[2],2);
        dist += pow((pvy - vyi[i])/sigma[3],2);
        res[i] += norm * exp(-.5*dist);
    }
    
    return;
}
void read_evaluation_points(char *fname, real *points, int n) {

    printf("opening %s\n",fname);
    printf("Reading %d points\n",n);
    FILE *f; 

    size_t tot = (size_t)(n*4);
    size_t status;
    
    printf("reading\n");
    f = fopen("ic/points.dat","r");
    if (f==NULL) printf("Error opening file\n");

    status = fread(&points[0], sizeof(real),tot,f);
    printf("Read %d points\n",status);


    fclose(f);
    return;


}
