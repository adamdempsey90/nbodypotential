#include "potential.h"

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
    }
    fclose(f);
    return;

}
