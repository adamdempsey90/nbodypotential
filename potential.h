


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
