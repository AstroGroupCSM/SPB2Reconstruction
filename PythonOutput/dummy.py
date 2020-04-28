import numpy as np
import sys
w=sys.argv[-1]
F=int(sys.argv[-2])
M=[np.loadtxt('txts/example-'+str(i)+'.txt') for i in range(128)]
if F>=200:
    header="Signal/"
else:
    header="Noise/"
np.save(header+"event-"+str(w)+".npy",M)
