import numpy as np
from zacLib import awgn

def bell(n,h):
    return h*np.exp(-np.arange(-n//2,n//2)**2/2/(n/8)**2)
def square(n,h,alpha=0,ripple=0):
    if alpha>n//2: alpha = n//2
    s = np.zeros(n)
    s[0:alpha] = np.cos(np.linspace(np.pi,np.pi*2,alpha))+1
    s[n-alpha:n] = np.cos(np.linspace(0,np.pi,alpha))+1
    s[alpha:n-alpha] = 2+ripple*np.sin(np.linspace(0,12*np.pi,n-2*alpha))
    return 0.5*h*s
def stair(n,h1,h2,alpha=0,ripple=0):
    return bell(n,h1)+square(n,h2,alpha,ripple)

def squares(noise=0.1):
    x = np.zeros(14000)
    x[1100:2000] += square(900,1,alpha=0)
    x[2100:3000] += square(900,1,alpha=100)
    x[3100:4000] += square(900,1,alpha=300)
    x[4100:5000] += square(900,1,alpha=450)
    x[5100:6000] += square(900,0.5,alpha=0)
    x[6100:7000] += square(900,0.5,alpha=100)
    x[7100:8000] += square(900,0.5,alpha=300)
    x[8100:9000] += square(900,0.5,alpha=450)
    x[9100:9400] += square(300,1,alpha=0)
    x[9500:9800] += square(300,1,alpha=60)
    x[9900:10200] += square(300,1,alpha=150)
    x[10300:10400] += square(100,1,0)
    x[10500:10600] += square(100,1,50)
    x[10700:10800] += square(100,1,25)
    return x,awgn(x,None,noise)
def stairs(noise=0.1):
    x = np.zeros(10000)
    x[1100:2000] = stair(900,1,0.1,alpha=100,ripple=0)
    x[2100:3000] = stair(900,0.4,1,alpha=200,ripple=0)
    x[3100:4000] = stair(900,1,1,alpha=30,ripple=0)
    x[4100:5000] = stair(900,1,1,alpha=300,ripple=0)
    return x,awgn(x,None,noise)
def bells(noise=0.1):
    x = np.zeros(20000)
    x[1100:1300] = bell(200,1)
    x[1400:1600] = bell(200,1)
    x[1700:2100] = bell(400,1)
    x[2200:2800] = bell(600,1)
    x[2900:4000] = bell(1100,1)
    x[4100:6100] = bell(2000,1)
    x[6200:10000] = bell(3800,1)

    x[10100:10300] = bell(200,0.3)
    x[10400:10600] = bell(200,0.3)
    x[10700:11100] = bell(400,0.3)
    x[11200:11800] = bell(600,0.3)
    x[11900:13000] = bell(1100,0.3)
    x[13100:15100] = bell(2000,0.3)
    x[15200:19000] = bell(3800,0.3)
    return x,awgn(x,None,noise)

def random(noise=0.1):
    x = np.zeros(10000)
    n = 30
    for i in range(n):
        f = np.random.randint(0,10)
        if f<5:
            w = np.random.randint(10,400)
        elif f>=5 and f<8:
            w = np.random.randint(400,800)
        elif f>=8:
            w = np.random.randint(800,1000)
        p = np.random.randint(0,2)
        h = np.random.rand()*2 + 0.5
        if p==0:
            a = bell(w,h)
        elif p==1:
            a = square(w,h,int(np.random.randint(0,w//2)))
        elif p==2:
            a = stair(n,h,np.random.rand(0,3),int(np.random.randint(0,w//2)))
        pos = np.random.randint(1000,9000-w)
        x[pos:pos+w] += a
    return x,awgn(x,None,noise)