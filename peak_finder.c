#include <stdio.h>
#include "mkl.h"

#define ALIGMENT 64
#define LEN 524288
#define MINBANDPTS 3

int findBandInRange(double *x, double peakThreshold, int start, int end,int *peaks, int init) {
    int mxi = start + cblas_idamax(end-start,&x[start],1);
    int i = mxi-1, j = mxi+1;
    double th = x[mxi] - peakThreshold;
    while (i>=start && x[i]>th) { i--; }
    while (j<end && x[j]>th) { j++; }
    if (i<start && j==end){ return 0; } 
    static int peakLen = 0;
    if (init) { peakLen = 0; }
    if (i>=start+MINBANDPTS) { 
        findBandInRange(x, peakThreshold, start, i, peaks, 0); 
    }
    if (i>=start && j<end) {
        peaks[peakLen++] = i;
        peaks[peakLen++] = j;        
    }
    if (j<end-MINBANDPTS) { 
        findBandInRange(x, peakThreshold, j, end, peaks, 0); 
    }
    return peakLen;
}

int findBandInRange2(int n, double *x, double peakThreshold, int *ploc, int *vloc) {
    int idmax = 0, idmin = 0, len = 0, rising = -1;
    double d;
    for (int i=0;i<n;i++){
        d = x[i];
        if (d>x[idmax]) {
            idmax = i;
            while (d-x[idmin]>peakThreshold) {idmin++;}
            rising = idmin;
        }
        if (d<x[idmin]) {
            idmin = i;
            while (x[idmax]-d>peakThreshold) {idmax++;}
            if (rising>=0) {
                int t = idmax-1;
                while(t>rising && x[idmax]-x[t]<peakThreshold){t--;}
                ploc[len] = t;
                vloc[len++] = i;
                rising = -1;
            }
        }
    }
    return len;
}

int main() {
    // parameter
    const char *filename = "test.dat";
    FILE *file = fopen(filename,"rb");
    double *sp = mkl_malloc(LEN*sizeof(double),ALIGMENT);
    fread(sp,sizeof(double),LEN,file);
    fclose(file);

    int *peaks,*ploc,*vloc,peakLen;
    double times[7];
    double th = 0.1;

    // O(nlogn) implenment
    for (int i=0;i<7;i++){
        double tic=dsecnd();
        for (int j=0;j<1000;j++){
            peaks = mkl_malloc(600*sizeof(int),ALIGMENT); // max stack size 300        
            peakLen = findBandInRange(sp, th, 0, LEN, peaks, 1);     
            ploc = mkl_malloc(peakLen/2*sizeof(int),ALIGMENT);
            vloc = mkl_malloc(peakLen/2*sizeof(int),ALIGMENT);       
            for (int i=0;i<peakLen/2;i++){
                ploc[i] = peaks[2*i];
                vloc[i] = peaks[1+2*i];
            }
            mkl_free(ploc);
            mkl_free(vloc);
            mkl_free(peaks);
        }
        times[i] = (dsecnd()-tic)/1000;
        printf("%f, ",times[i]);        
    }
    printf("\navg: using %f ms\n",(cblas_dasum(7,times,1))/7*1000);
    printf("Total peak: %d\n",peakLen/2);
    for (int i=0;i<peakLen/2;i++) {
        printf("%d  %d\n",ploc[i],vloc[i]);
    }

    // O(n) implenment
    for (int i=0;i<7;i++){
        double tic=dsecnd();
        for (int j=0;j<1000;j++){             
            ploc = mkl_malloc(300*sizeof(int),ALIGMENT);
            vloc = mkl_malloc(300*sizeof(int),ALIGMENT);  
            peakLen = findBandInRange2(LEN, sp, th, ploc, vloc);
            mkl_free(ploc);
            mkl_free(vloc);
        }
        times[i] = (dsecnd()-tic)/1000;
        printf("%f, ",times[i]);        
    }
    printf("\navg: using %f ms\n",(cblas_dasum(7,times,1))/7*1000);
    printf("Total peak: %d\n",peakLen);
    for (int i=0;i<peakLen;i++) {
        printf("%d  %d\n",ploc[i],vloc[i]);
    }

    mkl_free(sp);
    return 0;
}