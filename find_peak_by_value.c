#include <stdio.h>
#include <time.h> 

#define LEN 10000

void FindBandByValue(double *x, double th, int *rising_edge, int *falling_edge, int *maxloc){
    int idmax = 0;
    double xmax = x[idmax];
    int ptr_rising = 0;
    int ptr_falling = 0;
    int ptr_maxloc = 0;
    int last_rising = 0;
    for (int i=1;i<LEN;i++) {
        double d = x[i];
        if (d>xmax) {
            idmax = i;
            xmax = x[idmax];
        } else if (xmax-d>th) {
            if (x[idmax-1]<xmax) {
                int rising = idmax;
                while (xmax-x[rising]<th && rising>last_rising) {
                    rising--;
                }
                if (rising>last_rising){
                    last_rising = rising;
                    // printf("rising/max/falling: %d, %d, %d",rising,idmax,i);
                    rising_edge[ptr_rising++] = rising;
                    falling_edge[ptr_falling++] = i;
                    maxloc[ptr_maxloc++] = idmax;
                }
            }
            idmax = i;
            xmax = x[idmax];
        }        
    }
}

int main() {
    const char *filename = "test.dat";
    FILE *file = fopen(filename,"rb");
    double sp[LEN];
    int rising_edge[LEN];
    int falling_edge[LEN];
    int maxloc[LEN];
    for (int i=0;i<LEN;i++){
        rising_edge[i]=0;
        falling_edge[i]=0;
        maxloc[i]=0;
    }
    fread(sp,sizeof(double),LEN,file);
    fclose(file);
    double th = 0.7;
    double times[7];
    for (int i=0;i<7;i++){
        clock_t start, end;
        double cpu_time_used;   
        start = clock();
        for (int j=0;j<10000;j++){    
            FindBandByValue(sp,th,rising_edge,falling_edge,maxloc);
        }
        end = clock();
        cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
        times[i] = cpu_time_used/10000;
        printf("%f, ",times[i]);        
    }
    printf("\n");
    for (int i=0;i<LEN;i++){
        if (rising_edge[i]==0){break;}
        printf("rising/max/falling: %d, %d, %d\n",rising_edge[i],maxloc[i],falling_edge[i]);
        
    }
    return 0;
}