import numpy as np
import numba

# algorithm in this file have same logic:
# find a peak at location <i> on signal <x>, then find nearest left/right point that match x[i]-x[left]>th, x[i]-x[right]>th
# The left/right points are boundary of this peak, then find another

@numba.njit
def findBandInRange(x,peakThreshold,start,end,minBandPts=3):    
    """recursive method, find max peak on <x>, then recursively find left-hand side and right-hand side"""
    mxi = start + x[start:end].argmax()
    th = x[mxi]-peakThreshold
    i = mxi - 1   
    while i>=start and x[i] >= th:
        i -= 1
    j = mxi + 1
    while j<end and x[j]>=th:
        j += 1
    pl,vl = [0 for _ in range(0)],[0 for _ in range(0)] # numba only support this weird shit
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

@numba.njit
def findBandInRange2(x,peakThreshold):
    """iteration method, same algorithm as `findBandInRange`"""
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


@numba.njit
def FindBandByValue(x,th):
    """two pointer algorithm,"""
    idmax = 0
    xmax = x[idmax]
    rising_edge, falling_edge = [0], []
    maxloc = []
    for i,d in enumerate(x):      
        if d>xmax:
            idmax = i
            xmax = x[idmax]
        elif xmax-d>th:
            if x[idmax-1]<xmax:                
                rising = idmax
                while xmax-x[rising]<th:
                    rising -= 1   
                    if rising < rising_edge[-1]:
                        break
                else:             
                    rising_edge.append(rising)
                    falling_edge.append(i)
                    maxloc.append(idmax)
            idmax = i
            xmax = x[idmax]
    return rising_edge[1:],falling_edge,maxloc

# main entry
@numba.njit()
def find_peak_value(x,th):
    ploc,vloc,maxloc,minloc = FindBandByValue(x,th)
    return np.array(ploc),np.array(vloc),np.array(maxloc),np.array(minloc)