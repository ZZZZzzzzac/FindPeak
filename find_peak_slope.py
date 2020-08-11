import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
from rd import Reader
from zacLib import nextpow2,mySmooth
import matplotlib

def rd_reader(file,nframe=1,pos=0):
    read = Reader(file)
    fs = read.header.dump()['sampling']['sample_rate']
    framesize = read.header.frame_size
    return read.read(1,pos,nframe*framesize).reshape(nframe,framesize),fs

def fft(signal,nfft):    
    t1 = np.fft.fft(signal,nfft)
    t2 = t1.sum(axis=0)
    t3 = np.abs(t2)
    t4 = np.log10(t3)[:nfft//2]
    return t4

def dev(x,n=1):
    # derivative of x: y[i] = y[i+1:i+n+1].sum() -  y[i-n:n].sum()
    y = np.zeros(x.size,dtype=x.dtype)
    cx = np.cumsum(x)
    dcx = cx[n:]-cx[:-n] ## cx[n:]-cx[:-n] is equivalent to mySmooth(x,n,0)
    y[n+1:-n] = dcx[n+1:] - dcx[:-n-1]
    y[n] = dcx[n] - cx[n-1]
    return y

def filter_extremum(x,ploc,vloc):
    # this function choose rising edge with maximum magnitude slope between two falling edge
    # and choose falling edge with minimum magnitude slope between two rising edge
    if len(ploc)==0 or len(vloc)==0:
        raise ValueError("input ploc and vloc must not be empty")
    ploc=ploc[::-1].tolist() # this transform is for following pop() operation
    vloc=vloc[::-1].tolist()
    pl = ploc.pop()
    vl = vloc.pop()
    p = x[pl]
    v = x[vl]
    new_peak = [pl]
    new_pits = [vl]
    while len(ploc)>0 and len(vloc)>0: 
        if pl>vl:            
            if x[vl]<v:
                new_pits[-1] = vl 
                v = x[vl]
            vl = vloc.pop()
            if vl>pl:
                new_pits.append(vl)
                v=x[vl]
        else:            
            if x[pl]>p:
                new_peak[-1] = pl
                p = x[pl]
            pl = ploc.pop()
            if pl>vl:
                new_peak.append(pl)
                p=x[pl]
    if len(ploc)==0 and len(vloc)>0:
        new_pits[-1] = vloc[x[vloc].argmin()]
    elif len(ploc)>0 and len(vloc)==0:
        new_peak[-1] = ploc[x[ploc].argmax()]
    while new_pits[0] < new_peak[0]:
        new_pits.pop(0)
    return np.array(new_peak),np.array(new_pits)    

def sample2frequency(sm,new_peak,new_pits):
    npeak = len(new_peak)
    avg_frq = (new_peak+new_pits)//2
    max_frq = np.zeros(npeak,dtype=int)
    dev_band = new_pits-new_peak
    half_left = np.zeros(npeak,dtype=int)
    half_right = np.zeros(npeak,dtype=int)
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
    for i,(up,down) in enumerate(zip(new_peak,new_pits)):   
        max_frq[i] = up + sm[up:down].argmax()
        maxi = sm[max_frq[i]]
        left_min = get_nearest_minimum(sm,up,'left')
        right_min = get_nearest_minimum(sm,down,'right')
        mini = max(sm[left_min],sm[right_min])
        height = (maxi+mini)/2
        left_idx = up
        while sm[left_idx]<height:
            left_idx+=1
        right_idx = down
        while sm[right_idx]<height:
            right_idx-=1
        half_left[i] = left_idx
        half_right[i] = right_idx
    return avg_frq,max_frq,dev_band,half_left,half_right,half_right-half_left

def plot_all(sp,sm,dsp,slope_th,
             new_peak,new_pits,avg_frq,max_frq,dev_band,half_left,half_right,f=1):    
    matplotlib.use("Qt5Agg")
    f_axis = np.arange(sm.size)*f
    plt.figure()
    ax1 = plt.subplot(211)
    # plt.plot(sp,'.',markersize=1,label='Raw') # if sp.size>1e6, plot sp as scatter will be very slow
    plt.plot(f_axis,sm,label='Smoothed')
    plt.plot(new_peak*f,sm[new_peak],'r^',label='rising')
    plt.plot(new_pits*f,sm[new_pits],'gv',label='falling')
    plt.plot(half_left*f,sm[half_left],'b>',label='half_left')
    plt.plot(half_right*f,sm[half_right],'b<',label='half_right')
    plt.plot(avg_frq*f,sm[avg_frq],'d',label='avg_peak')
    plt.plot(max_frq*f,sm[max_frq],'s',label='max_peak')
    plt.legend(loc='right')
    plt.subplot(212,sharex=ax1)
    plt.plot(f_axis,dsp)
    plt.plot(f_axis,np.ones(sm.size)*slope_th,'--')
    plt.plot(f_axis,np.ones(sm.size)*-slope_th,'--')
    plt.plot(new_peak*f,dsp[new_peak],'r^',markersize=4)
    plt.plot(new_pits*f,dsp[new_pits],'gv',markersize=4)
    plt.show()

def find_peak_slope(sp,width,slope_th):    
    # padding zeros due to dev() cannot calculate derivative at both end
    # sp = np.concatenate([np.ones(width)*sp[0],sp,np.ones(width)*sp[-1]])
    sm = mySmooth(sp,width,n=1) # n control shape of smooth kernel (lager n, kernel is more like gaussian(bell) shape)
    dsp = dev(sp,width) # calculate derivative/slope
    ploc = np.where(dsp>slope_th)[0] # slope > slope_th ==> rising edge 
    vloc = np.where(dsp<-slope_th)[0] # slope < -slope_th ==> falling edge
    new_peak,new_pits = filter_extremum(dsp,ploc,vloc) # find maximun/minimum slope point at rising/falling edge
    avg_frq,max_frq,dev_band,half_left,half_right,half_band = sample2frequency(sm,new_peak,new_pits)
    return sm,dsp,new_peak,new_pits,avg_frq,max_frq,dev_band,half_left,half_right,half_band

def find_peak_slope_from_rd(file,width,slope_th,nframe=10,pos=10):
    signal,fs = rd_reader('siga.rd',nframe=10,pos=10) # Import data
    framesize = signal.shape[1]
    nfft = nextpow2(framesize,n=0) 
    sp = fft(signal,nfft) # get spectrum (fft)
    # Main function find_peak_slope
    sm,dsp,new_peak,new_pits,avg_frq,max_frq,\
        dev_band,half_left,half_right,half_band = find_peak_slope(sp,width,slope_th)

    # result representation
    f = 1/nfft*2*fs/1e6 # using frequency as x-axis
    # f = 1 # using sample points as x-axis
    plot_all(sp,sm,dsp,slope_th,new_peak,new_pits,avg_frq,max_frq,dev_band,half_left,half_right,f)
    df = pd.DataFrame({ 'avg_frq':avg_frq*f,'max_frq':max_frq*f,
                        'slope band':dev_band*f,
                        'half_band':half_band*f})
    return df

