#include <stdio.h>
#include "mkl.h"

#define ALIGMENT 64
#define LEN 10000

void cumsum(MKL_INT n, double *x){
    for (int i=1;i<n;i++){
        x[i] += x[i-1];
    }
}

void get_edge(double *x, double th, double thp, int *np, int *ploc,int *nv, int *vloc){
    int ptrP=-1,ptrV=-1;
    int pl=0,vl=0;
    double maxPeak=0.0,minPeak=0.0;
    int current = 0;
    double Nrth = th-thp,Nfth=-th+thp,fth=-th;
    double d;
    for (int i=0;i<LEN;i++){
        d = x[i];
        if (d<Nrth && d>Nfth){
            switch (current) {
            case 1:
                if (ptrP-ptrV==1){
                    ploc[ptrP] = pl;
                } else {
                    ploc[++ptrP] = pl;
                }
                maxPeak = 0;
                current = 0;
                break;
            
            case 2:
                if (ptrP-ptrV==1) {
                    vloc[++ptrV] = vl;
                }
                minPeak = 0;
                current = 0;
                break;
            case 0:
                break;
            }     
        } else if (d>th) {
            current = 1;
            if (d>maxPeak) {
                maxPeak = d;
                pl = i;
            }
        } else if (d<fth) {
            current = 2;
            if (d<minPeak) {
                minPeak = d;
                vl = i;
            }
        }        
    }
    if (ptrP-ptrV==1){
        ptrP--;
    }
    *np = ptrP+1;
    *nv = ptrV+1;
}

void find_peak_slope(double *sp, MKL_INT width, double slope_th){    
    // csp = cumsum(sp)
    double *csp = mkl_malloc(LEN*sizeof(double),ALIGMENT);
    cblas_dcopy(LEN,sp,1,csp,1);
    cumsum(LEN,csp);
    // sm = csp[width:]-csp[:-width] <= equivalent to mySmooth(sp,width,n=0)
    double *sm = mkl_calloc(LEN,sizeof(double),ALIGMENT);
    vdSub(LEN-width,&csp[width],csp,&sm[width/2]);
    // dsp = sm[width:]-sm[:-width]
    double *dsp = mkl_calloc(LEN,sizeof(double),ALIGMENT);
    vdSub(LEN-2*width-1,&sm[width+width/2+1],&sm[width/2],&dsp[width+1]);
    dsp[width] = sm[width+width/2] - csp[width-1];
    // get_edge(dsp,slope_th,slope_th/3) 
    int np,nv;
    int capacity = 400; // max stack size
    int *ploc = mkl_calloc(capacity,sizeof(int),ALIGMENT);
    int *vloc = mkl_calloc(capacity,sizeof(int),ALIGMENT);  

    get_edge(dsp, slope_th, slope_th/3, &np, ploc, &nv, vloc);
        
    mkl_free(sm);
    mkl_free(csp);
    mkl_free(dsp);
    mkl_free(ploc);
    mkl_free(vloc);
}

int main(){
    // parameter
    const char *filename = "test.dat";
    MKL_INT width = 400;
    double slope_th = 32.0;
    // input log10(abs(fft result))
    FILE *file = fopen(filename,"rb");
    double *sp = mkl_malloc(LEN*sizeof(double),ALIGMENT);
    fread(sp,sizeof(double),LEN,file);
    fclose(file);

    double times[7];
    for (int i=0;i<7;i++){
        double tic=dsecnd();
        for (int j=0;j<100;j++){    
            find_peak_slope(sp, width, slope_th);
        }
        times[i] = (dsecnd()-tic)/100;
        printf("%f, ",times[i]);        
    }
    
    // printf("using %f ms\n",(dsecnd()-tic)*1000);

    // printf("np %d nv %d\n",np,nv);
    // for (int i=0;i<np;i++) {
    //     printf("%d  %d\n",ploc[i],vloc[i]);
    // }



    // find_peak_slope(sp,width,slope_th)
    mkl_free(sp);
    return 0;
}