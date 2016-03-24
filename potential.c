#include <stdio.h>
#include <stdlib.h>
#include <math.h>



typedef struct Particle {
    double x;
    double y;
    double z;
    double vx;
    double vy;
    double vz;
    double dt;
    double energy;
} Particle;

void evolve(Particle *p, double tend); 
void output(int n, Particle *p) ;
double dy_potential(double x, double y) ;
double dx_potential(double x, double y) ;
double potential(double x, double y) ;
void leapfrog_step(Particle *p) ;
void set_particle_dt(Particle *p); 
void set_particle_ic(Particle *p);
void set_energy(Particle *p);
int ntot, nt;
double params[2] = {0.8,1.5};

int main(int argc, char *argv[]) {
    
    ntot = atoi(argv[1]);
    nt = atoi(argv[2]);
    params[0] = atof(argv[3]);
    params[1] = atof(argv[4]);

    printf("Using NTOT=%d\tnt=%d\tq^2=%f\tR^2=%f\n",ntot,nt,params[0],params[1]);

    Particle *particles = (Particle *)malloc(sizeof(Particle)*ntot);
    

    int i,j;
    double dt = 1.;

    set_particle_ic(particles);
    output(0,particles);
    for(j=1;j<nt;j++) {
        for(i=0;i<ntot;i++) {
            evolve(&particles[i],j*dt);
        }
        output(j,particles);
    }
    return 1;
}

void evolve(Particle *p, double tend) {
    double t=0;

    set_particle_dt(p);
    
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
void set_particle_ic(Particle *p) {

    int i,j;
    int n = (int)sqrt(ntot);

    for(j=0;j<n;j++) {
        for(i=0;i<n;i++) {
            p[j+i*n].x = -6.5 + i*10.0/sqrt(ntot);
            p[j+i*n].y = -6.5 + j*10.0/sqrt(ntot);
            set_energy(&p[j+i*n]);
        }
    }
    return;

}

void set_particle_dt(Particle *p) {
    p->dt = .1;
    return;
}

void output(int n, Particle *p) {
    FILE *f;
    int i;
    char fname[100];
    sprintf(fname,"outputs/particles_%d.dat",n);
    f = fopen(fname,"w");

    for(i=0;i<ntot;i++) {
        fwrite(&p[i].x,sizeof(double),1,f);
        fwrite(&p[i].y,sizeof(double),1,f);
        fwrite(&p[i].z,sizeof(double),1,f);
        fwrite(&p[i].vx,sizeof(double),1,f);
        fwrite(&p[i].vy,sizeof(double),1,f);
        fwrite(&p[i].vz,sizeof(double),1,f);
        fwrite(&p[i].energy,sizeof(double),1,f);
//    fprintf(f,"%d\t%lg\t%lg\t%lg\t%lg\t%lg\t%lg\n",
//                i,p[i].x,p[i].y,p[i].z,p[i].vx,p[i].vy,p[i].vz);
    }
    fclose(f);
    return;

}
void leapfrog_step(Particle *p) {

    double dt = p->dt;
    double x = p->x;
    double y = p->y;
    double vx = p->vx;
    double vy = p->vy;

    x += vx*dt*.5;
    y += vy*dt*.5;
    vx += -dt*dx_potential(x,y);
    vy += -dt*dy_potential(x,y); 
    x += vx*dt*.5;
    y += vy*dt*.5;
    p->x = x;
    p->y = y;
    p->vx = vx;
    p->vy = vy;

    return;
}

double potential(double x, double y) {
    double q2 = params[0];
    double R2 = params[1];

    return log(x*x + y*y/q2 + R2);
}
double dx_potential(double x, double y) {
    double q2 = params[0];
    double R2 = params[1];
    return 2*x/(R2 + x*x + y*y/q2); 
}
double dy_potential(double x, double y) {
    double q2 = params[0];
    double R2 = params[1];

    return 2*y/( q2*(R2 + x*x) + y*y); 
}
