import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from sklearn.neighbors import KernelDensity
import pickle

pkl_file = open('dates.pkl', 'rb')
dates = pickle.load(pkl_file)
pkl_file.close()

sites = ['PuertoPrincesa','Ranai','Singapore']
varnames = ['capes','colvap','shear']
colors = ['purple','green','gray']
#colors = ['b','g','k','r','m','c']
lines = {}
for var in varnames:
    lines[var] = []
    for site in sites:
        x = np.load(site+var+'.npz')
        nprof = len(dates[site])
        y = np.zeros(nprof)
        for i,t in enumerate(dates[site]):
            y[i] = x[str(t)]
            if y[i] >4000:
                y[i]=4000.
        lines[var].append(y)
    maxes = np.asarray([max(a) for a in lines[var]])
    mins = np.asarray([min(a) for a in lines[var]])
    bins = np.linspace(mins.min(),maxes.max(),100)
    
    for i in range(3):
        yv = plt.hist(lines[var][i],normed = True,bins = bins,color= colors[i],label = sites[i],histtype='step')
    plt.legend()
    plt.savefig('comparehist3'+var+'.png')
    plt.clf()
    
fig = plt.figure()
ax = plt.axes(projection='3d')
for i in range(3):
    ax.scatter3D(lines['colvap'][i],lines['capes'][i],lines['shear'][i],c=colors[i],color = colors[i],label = sites[i],depthshade=False)
ax.legend(loc='best')
ax.set_xlabel('RH',fontweight='bold')
ax.set_ylabel('   CAPE',fontweight='bold')
ax.set_zlabel('Shear',fontweight='bold')
fig.savefig('test3dscatter3.png')
plt.close()

for var in varnames:
    modelvals = np.load('model'+var+'.npz')
    model = np.array([modelvals[f] for f in modelvals.keys()])
    
    
    obs = []
    for i in range(3):
        obs.extend(lines[var][i])
    maxes = np.asarray([max(a) for a in lines[var]])
    mins = np.asarray([min(a) for a in lines[var]])
    bins = np.linspace(mins.min(),maxes.max(),100)
    
        
    h=plt.hist(obs, normed =True, bins=bins, histtype='step')
    modely=np.ones(75)*(max(h[0])/5.)
    plt.scatter(model, modely)
    plt.hist(obs, normed =True, bins=bins, histtype='step')
    #plt.hist(model, normed = True, bins=bins, histtype='bar')
    plt.savefig(var+'withmodel.png')
    plt.clf()
    

modelvals = np.load('modelcapes.npz')
model = np.array([modelvals[f] for f in modelvals.keys()])
obs = []
for i in range(3):
    obs.extend(lines['capes'][i])
obs=np.array(obs)

bins = np.linspace(obs.min(),obs.max(),100)

fig = plt.figure() 
ax1 = fig.add_subplot(3,1,1)    
h=ax1.hist(obs, normed =True, bins=bins, histtype='step',label='obs')
modely=np.ones(75)*(max(h[0])/5.)
ax1.scatter(model, modely,label='model')
ax1.hist(obs, normed =True, bins=bins, histtype='step')
ax1.set_title('CAPE')


modelvals = np.load('modelcolvap.npz')
model = np.array([modelvals[f] for f in modelvals.keys()])
obs = []
for i in range(3):
    obs.extend(lines['colvap'][i])
obs=np.array(obs)
bins = np.linspace(obs.min(),obs.max(),100)

ax2 = fig.add_subplot(3,1,2)    
h=ax2.hist(obs, normed =True, bins=bins, histtype='step',label='obs')
modely=np.ones(75)*(max(h[0])/5.)
ax2.scatter(model, modely,label='model')
ax2.hist(obs, normed =True, bins=bins, histtype='step')
ax2.set_title('Column Vapor')



modelvals = np.load('modelshear.npz')
model = np.array([modelvals[f] for f in modelvals.keys()])
obs = []
for i in range(3):
    obs.extend(lines['shear'][i])
obs=np.array(obs)
    
bins = np.linspace(obs.min(),obs.max(),100)

ax3 = fig.add_subplot(3,1,3)    
h=ax3.hist(obs, normed =True, bins=bins, histtype='step',label='obs')
modely=np.ones(75)*(max(h[0])/5.)
ax3.scatter(model, modely,label='model')
ax3.hist(obs, normed =True, bins=bins, histtype='step')
ax3.set_title('Shear')

plt.tight_layout()    
plt.savefig('panel3hist.png')
plt.close(fig)
    
    
    
    