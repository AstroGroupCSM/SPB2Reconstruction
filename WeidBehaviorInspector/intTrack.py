import numpy as np
import matplotlib.pyplot as plt
import matplotlib.widgets as WID
import matplotlib.colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
import sys
plt.style.use('presentation')

FileName=sys.argv[-1]
if FileName=='EventPlotter.py':
    print("Tell me a file to use!")
    exit()
m=np.load(FileName)

pe_MAX=0
for i in range(len(m)):
    for j in range(len(m[i])):
        for k in range(len(m[i][j])):
            m[0][j][k]+=m[i][j][k]
            if pe_MAX <= m[0][j][k]:
                pe_MAX=m[0][j][k]

for j in range(len(m[0])):
    for k in range(len(m[0][j])):
        if m[0][j][k]==0:
            m[0][j][k]=np.nan
gtu=0
#sets up canvas
fig =plt.figure()
fig.set_figheight(9)
fig.set_figwidth(16)
ax = fig.add_axes([0.05,0.05,0.9,0.9])



delta_gtu = 1
max_GTU=len(m)
axcolor = 'lightgoldenrodyellow'
norm=plt.Normalize(1,pe_MAX)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)

print(m[0])

im=ax.imshow(m[0], norm=norm,origin='lower',cmap='YlOrBr')
ax.set_xlabel('X Pixel')
ax.set_ylabel('Y Pixel')
ax.set_title('Example Event -- Double Counted QE')
fig.colorbar(im,orientation='vertical',cax=cax,label='PE/pix')
plt.grid(True)
plt.savefig('previousQE.png')
plt.show()
