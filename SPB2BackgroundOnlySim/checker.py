import numpy as np
import math
from matplotlib import pyplot as plt
from scipy.integrate import quad as q
from  scipy.special import gamma as g
from scipy.special import comb as c
from scipy.stats import poisson
plt.style.use('presentation')




def pois(k,l):
    return l**k * np.exp(-1*l)/(math.factorial(k))

def ps(l,n):
    spix=l+n*np.sqrt(l)
    spix1=int(math.ceil(spix))
    #print spix, spix1
    summ=0
    for i in range(0,spix1):
        summ += pois(i,l)
    #print 1-summ
    return 1-summ

def nHot(npix,l,n):
    pS=ps(l,n)
    summ=0
    for i in range(npix):
        summ +=c(27,i)*(pS**i)*((1.0-pS)**(27.0-i))
    #print summ
    return 1.0-summ

def nActive(nAct,npix,l,n):
    temp=nHot(npix,l,n)
    total=23*23*126
    summ=0
    for i in range(1):
        summ+=c(total,i)*(temp**i)*((1.0-temp)**(total-i))
    #return temp*total
    return (1.0-summ)

def camera(nAct,npix,l,n):
    temp=nActive(nAct,npix,l,n)
    return 1-(1-temp)**3



tot=0
CSM=0
TG=0
print(camera(5,5,4,3.6)/(128.0*10.0**-6))
for i in range(10):
    m=np.loadtxt(open(str(i)+".txt").readlines()[:-1],usecols=[2,4])
    x=np.transpose(m)
    tot+=len(x[0])
    CSM+=sum(x[0])
    TG+=sum(x[1])
    #print(len(x[0]),sum(x[0]),sum(x[1]),i,camera(5,3,4,(4.0+(i/10))),camera(5,3,4,(4.0+(i/10)))*len(x[0]))
print(tot,CSM,TG)



#x1=np.random.poisson(lam=4.0,size=10000)
#x2=[np.random.poisson(lam=1.0)+np.random.poisson(lam=1.0)+np.random.poisson(lam=1.0)+np.random.poisson(lam=1.0) for i in range(10000)]
#bi=[-0.5+i for i in range(20)]
#plt.hist(x1,bins=bi)
#plt.hist(x2,bins=bi)
#plt.show()
