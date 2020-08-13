import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
from scipy.signal import find_peaks
import numba

from rd import Reader
from zacLib import nextpow2,mySmooth

def rd_reader(file,nframe=1,pos=0):
    """Read <nframe> frames from rd <file> started at <pos>-th frame."""
    read = Reader(file)
    fs = read.header.dump()['sampling']['sample_rate']
    framesize = read.header.frame_size
    return read.read(1,pos,nframe*framesize).reshape(nframe,framesize).sum(axis=0),fs

def fft(signal,nfft):    
    t1 = np.fft.fft(signal,nfft)
    t2 = np.abs(t1)
    t3 = np.log10(t2)[:nfft//2]
    return t3

@numba.njit()
def dev(x,n=1):
    """this function calculate derivative/slope x' of x.

    current algorithm use `(sum of right n point) - (sum of left n point)`\n    
    i.e. : x'[i] = x[i+1:i+n+1].sum() -  x[i-n:n].sum()\n 
    this algorithm is equivalent to smooth with n point flat kernel then get 
    `x'[i] = smoothed_x[i+n]-smoothed_x[i-n] `   

    when n is too small, it may miss some low slope signal\n 
    when n is too large, narrow band signal may be flatten and missed

    !!! Due to its implementation, the first and last n point in result are 
        all 0, so you must ensure the peak you want are in x[n:-n].
        To ensure that, either padding zeros before processing or take signal
        that no peak you want is in first and last n point.
    """
    # TODO: find a more accurate and noise insensitive way to calculate slope
    y = np.zeros(x.size,dtype=x.dtype)
    cx = np.cumsum(x)
    dcx = cx[n:]-cx[:-n] # cx[n:]-cx[:-n] is equivalent to mySmooth(x,n,0)
    y[n+1:-n] = dcx[n+1:] - dcx[:-n-1]
    y[n] = dcx[n] - cx[n-1]
    return y,np.concatenate((np.zeros(n//2),dcx,np.zeros(n-n//2)))/n

@numba.njit((numba.float64[::1],numba.float64,numba.float64))
def get_edge(x,th,thp):
    """Get rising/falling edge in slope <x> that above threshold <th>.

    Iterate along x once, so it is O(n). \n 
    Current algorithm is a state machine:\n 
    state "Nethier"(0), -th+thp < x[i] < th-thp \n 
           note: this `thp` is to prevent fake peak due to noise in x \n 
    state "Rising"(1), x[i] > th \n 
    state "Falling"(2), x[i] < -th \n 
    state transfer:  \n 
    "N"->"N", "N"->"R" and "N"->"F": pass \n 
    "R"->"R": update maxPeak; "F"->"F": update minPeak \n 
    "R"->"N", rising edge ends, append maxPeak location then reset maxPeak \n 
    "F"->"N", falling edge ends, append minPeak location then reset minPeak

    Following `if` is for finding 'stair' signal (narrow signal mounted upon a wide signal)
        when append maxPeak(pl), `if len(ploc)-len(vloc)==1` means it finds two rising edge 
        in a row, will choose last one. \n 
        This will keep the rising edge that most close to next falling edge

        when append minPeak(vl), `if len(ploc)-len(vloc)==1` means last rising edge has no 
        corresponding falling edge, it will append this falling edge. \n 
        This will stop appending two falling edge in a row and keep the one that most 
        close to corresponding rising edge.

    Args: 
        x <list or numpy array>, derivative/slope of spectrum
        th <double>: slope threshold, only x[i]>th(or x[i]<-th) will be treated as 
                     rising(falling) edge
        thp <double>: non-rising(falling) edge threshold, only x[i]<th-thp(or x[i]>-th+thp)
                      will be treated as end of rising/falling edge. Use th-thp(-th+thp) 
                      instead of th(-th) to ensure that x[i] is below threshold because of
                      end of rising/falling edge(the trend in x), not due to noise in x
    Returns:
        rising_edge,falling_edge <np.array(int)>: location of corresponding rising/falling edge, 
                                           should have same length

    !!! this algorithm runs very slow in python due to iteration, but should be fast in C !!!
    """
    ploc,vloc=[],[]
    maxPeak,minPeak = 0,0
    pl,vl = 0,0
    current = 0
    th_R2N = th-thp # threshold from state "R" to "N"
    th_F2N = -th+thp
    th_N2F = -th
    th_N2R = th
    for i,d in enumerate(x):
        if d<th_R2N and d>th_F2N: # use th-thp/-th+thp to prevent fake peak due to noise in x
            if current == 0:
                continue
            elif current == 1:
                if len(ploc)-len(vloc)==1: # two rising edge
                    ploc[-1] = pl # choose last one
                else:
                    ploc.append(pl)
                maxPeak = 0      
                current = 0          
            else:
                if len(ploc)-len(vloc)==1: # not two falling edge
                    vloc.append(vl) # keep first one
                minPeak = 0
                current = 0
        elif d>th_N2R:
            current = 1
            if d>maxPeak:
                maxPeak = d
                pl = i
        elif d<th_N2F:
            current = 2
            if d<minPeak:
                minPeak = d
                vl = i 
    if len(ploc)-len(vloc)==1:ploc.pop() # last rising edge has no falling edge, discard it 
    return np.array(ploc),np.array(vloc)

@numba.njit()
def sample2frequency(sm,rising_edge,falling_edge):
    """Calculate corresponding central frequency, bandwidth based on detected rising/falling edge

    Args:
        sm <list or numpy 1darray>: smoothed spectrum
        rising_edge,falling_edge <np.array(int)>: corresponding rising/falling edge location(in spectrum points)

    Returns <all np.array(int) with same length as rising_edge/falling_edge, unit in spectrum points>:
        avg_frq: frequency in middle of rising/falling edge
        max_frq: frequency with max spectrum magnitude between rising/falling edge
        dev_band: corresponding band of `falling edge location - rising edge location`
        half_left/half_right: location at rising/falling edge with height of the 'half' height of peak
        half_band: corresponding band of `half_right - half_left`
    """
    npeak = len(rising_edge)
    avg_frq = (rising_edge+falling_edge)//2
    max_frq = np.zeros(npeak,dtype=np.int32)
    dev_band = falling_edge-rising_edge
    half_left = np.zeros(npeak,dtype=np.int32)
    half_right = np.zeros(npeak,dtype=np.int32)
    def get_nearest_minimum(x,i,direction='left'):
        if direction == 'left':
            while i>1:
                if x[i-1]>x[i] and x[i]<x[i+1]:
                    break
                i-=1
            return i
        elif direction == 'right':
            while i<x.size-1:
                if x[i-1]>x[i] and x[i]<x[i+1]:
                    break
                i+=1
            return i
        else:
            return None
    for i,(up,down) in enumerate(zip(rising_edge,falling_edge)):   
        max_frq[i] = up + sm[up:down].argmax()
        maxi = sm[max_frq[i]]
        # TODO: half_left/right not very accurate, nearest minimum is not best method to find height reference
        left_min = get_nearest_minimum(sm,up,'left') 
        right_min = get_nearest_minimum(sm,down,'right')
        mini = max(sm[left_min],sm[right_min])
        height = (maxi+mini)/2
        left_idx = up
        if sm[up]<height:
            while sm[left_idx]<height:
                left_idx+=1
        else:
            while sm[left_idx]>height:
                left_idx-=1
        right_idx = down
        if sm[right_idx]<height:
            while sm[right_idx]<height:
                right_idx-=1
        else:
            while sm[right_idx]>height:
                right_idx+=1
        half_left[i] = left_idx
        half_right[i] = right_idx
    half_band = half_right-half_left
    return avg_frq,max_frq,dev_band,half_left,half_right,half_band

def plot_all(sp,sm,dsp,slope_th,rising_edge,falling_edge,avg_frq,
             max_frq,dev_band,half_left,half_right,f=1,origin=None):    
    """Plot spectrum, its slope and corresponding edge/band/frequency"""
    f_axis = np.arange(sm.size)*f
    plt.figure()
    ax1 = plt.subplot(211)
    if origin is not None:
        plt.plot(origin,label='origin')
    plt.plot(sp,'.',markersize=0.5,label='Raw') # if sp.size>1e7, plot sp as scatter will be very slow
    plt.plot(f_axis,sm,label='Smoothed')
    plt.plot(rising_edge*f,sm[rising_edge],'r^',label='rising')
    plt.plot(falling_edge*f,sm[falling_edge],'gv',label='falling')
    plt.plot(half_left*f,sm[half_left],'b>',label='half_left')
    plt.plot(half_right*f,sm[half_right],'b<',label='half_right')
    plt.plot(avg_frq*f,sm[avg_frq],'d',label='avg_peak')
    plt.plot(max_frq*f,sm[max_frq],'s',label='max_peak')
    plt.legend(loc='right')
    plt.subplot(212,sharex=ax1)
    plt.plot(f_axis,dsp)
    plt.plot(f_axis,np.ones(sm.size)*slope_th,'--')
    plt.plot(f_axis,np.ones(sm.size)*-slope_th,'--')
    plt.plot(rising_edge*f,dsp[rising_edge],'r^',markersize=4)
    plt.plot(falling_edge*f,dsp[falling_edge],'gv',markersize=4)
    plt.show()


def find_peak_slope(sp,width,slope_th):
    """Given signal <x>, smooth width <width> and slope threshold <slope_th>, return edge/band/frequency"""
    # padding zeros due to dev() cannot calculate derivative at both end
    # sp = np.concatenate([np.ones(width)*sp[0],sp,np.ones(width)*sp[-1]])
    # sm = mySmooth(sp,width,n=1) # n control shape of smooth kernel (lager n, kernel is more like gaussian(bell) shape)
    dsp,sm = dev(sp,width) # calculate derivative/slope
    rising_edge,falling_edge = get_edge(dsp,slope_th,slope_th/3) 
    avg_frq,max_frq,dev_band,half_left,half_right,half_band = sample2frequency(sm,rising_edge,falling_edge)
    return sm,dsp,rising_edge,falling_edge,avg_frq,max_frq,dev_band,half_left,half_right,half_band

def find_peak_slope_from_rd(file,width,slope_th,nframe=10,pos=10):
    """Main Entry, read rd data from <file>, do find_peak_slope(data,width,slope_th) then plot result"""
    signal,fs = rd_reader(file,nframe=10,pos=10) # Import data
    nfft = nextpow2(signal.shape[0],n=0) 
    sp = fft(signal,nfft) # get spectrum (fft)

    # Main function find_peak_slope
    (sm,dsp,rising_edge,falling_edge,avg_frq,max_frq,
     dev_band,half_left,half_right,half_band) = find_peak_slope(sp,width,slope_th)

    # result representation
    # f = 1/nfft*2*fs/1e6 # using frequency as x-axis
    f = 1 # using sample points as x-axis
    plot_all(sp,sm,dsp,slope_th,rising_edge,falling_edge,avg_frq,max_frq,dev_band,half_left,half_right,f)
    df = pd.DataFrame({ 'avg_frq':avg_frq*f,'max_frq':max_frq*f,'slope band':dev_band*f,'half_band':half_band*f})
    return df


