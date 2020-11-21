import numpy as np
import sys
import os 
eventID=sys.argv[-1]
outputDir=sys.argv[-2]
M=[np.loadtxt(outputDir+'/example-'+str(i)+'.txt') for i in range(128)]
os.system('rm '+outputDir+'/example-*.txt')
np.save(outputDir+"/event-"+eventID+".npy",M)
