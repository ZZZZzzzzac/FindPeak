import numpy as np
import numba

@numba.njit()
def get_local_maximum(x):
    return (np.diff(np.sign(np.diff(x))) == -2).nonzero()[0]+1 
    
@numba.njit()
def foo(x,loc,thp,start,end):         
    idx = start+x[loc[start:end+1]].argmax()
    if start==0: 
        left_min = x[:loc[idx]].min()
    else:
        left_min = x[loc[start-1]:loc[idx]].min() 
    if end==loc.size-1:
        right_min = x[loc[idx]:].min()
    else:
        right_min = x[loc[idx]:loc[end+1]].min()
    prominence = x[loc[idx]]-max(left_min,right_min)
    peaks = [0 for i in range(0)]
    if idx>start:        
        peaks += foo(x,loc,thp,start,idx-1)
    if prominence>thp:
        peaks += [loc[idx]]
    if idx<end:
        peaks += foo(x,loc,thp,idx+1,end)
    return peaks

def findPeakInProminence(x,thp):
    loc = get_local_maximum(x)
    return foo(x,loc,thp,0,loc.size-1)