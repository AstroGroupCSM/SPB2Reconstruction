import numpy as np
import sys
import os 
w=sys.argv[-1]
F=int(sys.argv[-2])
M=[np.loadtxt('example-'+str(i)+'.txt') for i in range(128)]
os.system('rm example-*.txt')
np.save("event-"+str(w)+".npy",M)
