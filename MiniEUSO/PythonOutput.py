import numpy as np
import sys
import os
eventID=sys.argv[-1]
outputDir=sys.argv[-2]
M=[np.loadtxt(outputDir+'/example-'+str(i)+'.txt') for i in range(128)]
os.system('rm '+outputDir+'/example-*.txt')
def Trigger(m,nSigma,nHot,nActive,lenTrigger):
    AvgBG=1.0*4.0
    PDM=[np.transpose(np.transpose(m)[:48]),np.transpose(np.transpose(m)[48:96]),np.transpose(np.transpose(m)[96:])]
    TriggerGTU=[-1,-1,-1]
    trigger=[0,0,0]
    nh=[]
    for pdm in range(len(PDM)):
        X=PDM[pdm]
        thresh = AvgBG+ nSigma*np.sqrt(AvgBG)
        MP=[[[0 for i in range(24)]for j in range(24)]for k in range(128)]
        NH=[[[0 for i in range(24)]for j in range(24)]for k in range(128)]
        for t in range(128):
            for x in range(48):
                for y in range(48):
                    MP[t][int(np.floor(x/2))][int(np.floor(y/2))]+=X[t][x][y]
        for t in range(1,127):
            for x in range(1,23):
                for y in range(1,23):
                    for it in [-1,0,1]:
                        for ix in [-1,0,1]:
                            for iy in [-1,0,1]:
                                if MP[t+it][x+ix][y+iy]>int(thresh):
                                    NH[t][x][y]+=1

        for t in range(128-lenTrigger):
            total=0
            for tt in range(t,t+lenTrigger):
                for x in range(24):
                    for y in range(24):
                        if NH[tt][x][y]>=nHot:
                            total+=1
            if total>=nActive and TriggerGTU[pdm]==-1:
                trigger[pdm]=1
                TriggerGTU[pdm]=tt
        nh.append(NH)
    return trigger,TriggerGTU,nh




m=M
hT,tGTU,NH=Trigger(M,5.0,2,34,20)
print(tGTU)
dT=20
for pdm in range(3):
    PDM=[np.transpose(np.transpose(m)[:48]),np.transpose(np.transpose(m)[48:96]),np.transpose(np.transpose(m)[96:])]
    if hT[pdm]==1:
        M=PDM[pdm]
        trigGTU=tGTU[pdm]
        if trigGTU <=dT:
            trigGTU=dT+1
        MM=[]
        for i in range(trigGTU-dT,trigGTU+dT):
            MM.append(M[i])
        np.save(outputDir+"/event-"+eventID+"-pdm-"+str(pdm)+".npy",MM)
