import numpy as np

tot=0
CSM=0
TG=0

for i in range(10):
    m=np.loadtxt(str(i)+".txt",usecols=[2,4],max_rows=2300)
    x=np.transpose(m)
    tot+=len(x[0])
    CSM+=sum(x[0])
    TG+=sum(x[1])
print(tot,CSM,TG)
