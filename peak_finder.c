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
    if (init) {peakLen = 0;}
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

int main() {
    // parameter
    const char *filename = "test.dat";
    MKL_INT width = 400;
    double slope_th = 32.0;
    // input log10(abs(fft result))
    FILE *file = fopen(filename,"rb");
    double *sp = mkl_malloc(LEN*sizeof(double),ALIGMENT);
    fread(sp,sizeof(double),LEN,file);
    fclose(file);

    int *peaks,*ploc,*vloc,peakLen;

    double times[7];
    for (int i=0;i<7;i++){
        double tic=dsecnd();
        for (int j=0;j<1000;j++){
            peaks = mkl_malloc(300*sizeof(int),ALIGMENT); // max stack size 300        
            peakLen = findBandInRange(sp, 0.5, 0, LEN, peaks, 1);     
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
    printf("using %f ms\n",(cblas_dasum(7,times,1))/7*1000);
    printf("Total peak: %d\n",peakLen/2);
    for (int i=0;i<peakLen/2;i++) {
        printf("%d  %d\n",ploc[i],vloc[i]);
    }
    mkl_free(sp);
    return 0;
}