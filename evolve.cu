#include "nbody.h"


__global__ void evolve_kernel(Particle *p, double tend) {
    int i,j,k;
    double t;

    i = threadIdx.x + blockIdx.x * blockDim.x;
    j = threadIdx.y + blockIdx.y * blockDim.y;

    while (t <= tend) {
        leapfrog_step(p);
        set_energy(p);
        t += p->dt;
    }
    return;
}

void set_energy(Particle *p) {
    p->energy = .5*(p->vx*p->vx + p->vy*p->vy + p->vz*p->vz);
    p->energy += potential(p->x,p->y);
    return;
}
