import numpy as np
import matplotlib.pyplot as plt
import matplotlib.widgets as WID
import matplotlib.colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
import sys
#plt.style.use('presentation')


FileName=sys.argv[-1]
if FileName=='EventPlotter.py':
    print("Tell me a file to use!")
    exit()
m=np.load(FileName)
pe_MAX=0
for i in range(len(m)):
    for j in range(len(m[i])):
        for k in range(len(m[i][j])):
            if pe_MAX <= m[i][j][k]:
                pe_MAX=m[i][j][k]
            if m[i][j][k]==0:
                m[i][j][k]=np.nan
gtu=0
#sets up canvas 
fig =plt.figure()
fig.set_figheight(9)
fig.set_figwidth(16)
ax = fig.add_axes([0.05,0.05,0.9,0.9])


    
delta_gtu = 1
max_GTU=len(m)
axcolor = 'lightgoldenrodyellow'
axslidder = plt.axes([0.15, 0.05, 0.55, 0.03], facecolor=axcolor)
axbuttona = plt.axes([0.75, 0.05, 0.05, 0.03], facecolor=axcolor)
axbuttonb = plt.axes([0.05,0.05,0.05,0.03],facecolor=axcolor)
axtxt = plt.axes([0.85,0.05,0.05,0.03],facecolor=axcolor)
slider = WID.Slider(axslidder, '', 1, max_GTU, valinit=gtu, valstep=delta_gtu,valfmt= "%1d")
buttona = WID.Button(axbuttona,'Next')
buttonb = WID.Button(axbuttonb,'Prev')
text_box= WID.TextBox(axtxt,'',initial=str(gtu))
norm=plt.Normalize(1,pe_MAX/2.0)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)

im=ax.imshow(m[0], norm=norm,origin='lower',cmap='YlOrBr')
ax.set_xlabel('X Pixel')
ax.set_ylabel('Y Pixel')
ax.set_title('GTU '+str(gtu))
fig.colorbar(im,orientation='vertical',cax=cax,label='PE/pix')

def s_update(val):
    global gtu
    gtu=int(slider.val)
    ax.clear()
    ax.imshow(m[gtu-1], norm=norm,origin='lower',cmap='YlOrBr')
    ax.set_xlabel('X Pixel')
    ax.set_ylabel('Y Pixel')
    ax.set_title('GTU '+str(gtu))
    plt.draw() 

def b_updatea(self):
    global gtu
    if gtu <max_GTU:
        gtu+=1
    slider.set_val(gtu)
    ax.clear()
    ax.imshow(m[gtu-1], norm=norm,origin='lower',cmap='YlOrBr')
    ax.set_xlabel('X Pixel')
    ax.set_ylabel('Y Pixel')
    ax.set_title('GTU '+str(gtu))
    plt.draw()
    
def b_updateb(self):
    global gtu
    if gtu>0:
        gtu-=1 
    slider.set_val(gtu)
    ax.clear()
    ax.imshow(m[gtu-1], norm=norm,origin='lower',cmap='YlOrBr')
    ax.set_xlabel('X Pixel')
    ax.set_ylabel('Y Pixel')
    ax.set_title('GTU '+str(gtu))
    plt.draw()

def text_update(text):
    global gtu
    gtu = int(text)
    slider.set_val(gtu)
    ax.clear()
    ax.imshow(m[gtu-1], norm=norm,origin='lower',cmap='YlOrBr')
    ax.set_xlabel('X Pixel')
    ax.set_ylabel('Y Pixel')
    ax.set_title('GTU '+str(gtu))
    plt.draw()

slider.on_changed(s_update)
buttona.on_clicked(b_updatea)
buttonb.on_clicked(b_updateb)
text_box.on_submit(text_update)
plt.show()
    
