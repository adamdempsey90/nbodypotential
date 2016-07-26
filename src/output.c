#include "potential.h"

void output(int time, real *x, real *y, real *vx, real *vy,int n) {
    FILE *f;
    int i;
    char fname[100];
    sprintf(fname,"%sparticles_%d.dat",params.outputdir,time);
    f = fopen(fname,"w");

    for(i=0;i<n*n;i++) {
        fwrite(&x[i],sizeof(real),1,f);
        fwrite(&y[i],sizeof(real),1,f);
        fwrite(&vx[i],sizeof(real),1,f);
        fwrite(&vy[i],sizeof(real),1,f);
    }
    fclose(f);
    return;

}

void output_init(double *points, double *pot_pars) {
    FILE *f;
    char fname[100];
    sprintf(fname,"%sinit_points_%d_%.2f_%.2f.dat",params.outputdir,params.nstars,pot_pars[0],pot_pars[1]);
    f = fopen(fname,"w");

    fwrite(points, sizeof(real), params.nstars,f);
    fclose(f);
    return;

}
