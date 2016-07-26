#include "potential.h"
#include <string.h>
#include <ctype.h>



void set_var(char *name,int int_val, real real_val, int bool_val, char *strval) {
    if (strcmp(name,"NSTARS") == 0) {
        params.nstars = int_val;
    }
    else if (strcmp(name,"NT") == 0) {	
        params.nt = int_val;

    }
    else if (strcmp(name,"DT") == 0) {	
        params.dt = real_val;

    }
    else if (strcmp(name,"THREADSPERBLOCK") == 0) {	
        params.threads_per_block = int_val;

    }
    else if (strcmp(name,"SIGMA") == 0) {	
        params.sigma = real_val;

    }
    else if (strcmp(name,"TOL") == 0) {	
        params.tol = real_val;

    }
    else if (strcmp(name,"SIZE") == 0) {	
        params.size = real_val;

    }
    else if (strcmp(name,"SIMPLEX_STEP") == 0) {	
        params.simplex_step = real_val;

    }
    else if (strcmp(name,"NTARGETS") == 0) {	
        params.ntargets = int_val;

    }
    else if (strcmp(name,"NPARS") == 0) {	
        params.npars = int_val;

    }
    else if (strcmp(name,"GENERATE") == 0) {	
        params.generate = bool_val;

    }
    else if (strcmp(name,"TARGETFILE") == 0) {	
        sprintf(params.targetfile, "%s",strval);

    }
    else if (strcmp(name,"OUTPUTDIR") == 0) {	
        sprintf(params.outputdir, "%s",strval);

    }
    else if (strcmp(name,"KDEMETHOD") == 0) {	
        sprintf(params.kdemethod_str, "%s",strval);

        if (strcmp(strval,  "direct") == 0) {
            params.kdemethod = 0;
        }
        else if (strcmp(strval,  "ifgt") == 0) {
            params.kdemethod = 1;
        }
        else if (strcmp(strval,  "direct_tree") == 0) {
            params.kdemethod = 2;
        }
        else if (strcmp(strval,  "ifgt_tree") == 0) {
            params.kdemethod = 3;
        }
        else if (strcmp(strval,  "auto") == 0) {
            params.kdemethod = 4;
        }
        else if (strcmp(strval,  "size") == 0) {
            params.kdemethod = 5;
        }
        else {
            printf("Invalid KDEMETHOD, defaulting to auto.\n");
            params.kdemethod = 4;
        }

    }
    

    return;
}
void read_param_file(char *fname) {
    FILE *f;

    char tok[20] = "\t :=>";

    char line[100],name[100],strval[100];
    char *data;
    real temp;
    int status;
    int int_val;
    int bool_val;
    char testbool;
    unsigned int i;

    f= fopen(fname,"r");

    while (fgets(line,100,f)) {
       // printf("%s\n",line);
        status = sscanf(line,"%s",name);
        //printf("%s\n",line);
      //  printf("%s\n",name);
        if (name[0] != '#' && status == 1) {
        
             data = line + (int)strlen(name);
             //sscanf(data + strspn(data,tok),"%lf",&temp);
             sscanf(data + strspn(data,tok),"%s",strval);
             temp = atof(strval);
             int_val = atoi(strval);
          //   printf("%s\t%lf\t%d\t%s\n",name,temp,int_val,strval);
            //int_val = (int)temp;
            testbool = toupper(strval[0]);
            if (testbool == 'Y') bool_val = TRUE;
            else bool_val = FALSE;
            
            for (i = 0; i<strlen(name); i++) name[i] = (char)toupper(name[i]);
            
            set_var(name,int_val,temp,bool_val,strval);

        }
    }

    return;
}
