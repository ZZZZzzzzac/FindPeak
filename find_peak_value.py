import numpy as np
import numba

@numba.njit()
def findBandInRange(x,peakThreshold,start,end,minBandPts=3):    
    mxi = start + x[start:end].argmax()
    th = x[mxi]-peakThreshold
    i = mxi - 1   
    while i>=start and x[i] >= th:
        i -= 1
    j = mxi + 1
    while j<end and x[j]>=th:
        j += 1
    pl,vl = [0 for _ in range(0)],[0 for _ in range(0)]
    ipl,ivl = [0 for _ in range(0)],[0 for _ in range(0)]
    jpl,jvl = [0 for _ in range(0)],[0 for _ in range(0)]
    if i<start and j==end:
        return pl,vl  
    if i>=start+minBandPts:
        ipl,ivl = findBandInRange(x,peakThreshold,start,i,minBandPts)
    if i>=start and j<end:
        pl,vl = [i],[j]
    if j<end-minBandPts:
        jpl,jvl = findBandInRange(x,peakThreshold,j,end,minBandPts)   
    return ipl+pl+jpl, ivl+vl+jvl

@numba.njit()
def findBandInRange2(x,peakThreshold):
    stack = [[0,x.size]]
    ploc,vloc = [],[]
    while stack: 
        start,end = stack.pop()
        idx = start + x[start:end].argmax()
        i,j = idx-1,idx+1
        th = x[idx] - peakThreshold
        while i>=start and x[i]>=th:
            i-=1
        while j<end and x[j]>=th:
            j+=1
        if i>=start and j<end:
            ploc.append(i)
            vloc.append(j)
        if i>=start+3:
            stack.append([start,i])
        if j<end-3:
            stack.append([j,end])
    ploc.sort()
    vloc.sort()
    return ploc,vloc

@numba.njit()
def find_peak_value(x,th):
    ploc,vloc = findBandInRange(x,th,0,x.size)
    return np.array(ploc),np.array(vloc)

@numba.njit()
def get_local_maximum(x):
    return np.array([i for i in range(1,x.size-1) if x[i]>x[i+1] and x[i]>x[i-1]])

