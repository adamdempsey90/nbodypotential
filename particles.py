import matplotlib.pyplot as plt
import numpy as np


def load_time(t,n=100,nf=4):
    dat=np.fromfile('outputs/particles_{0:d}.dat'.format(t))
    dat = dat.reshape(len(dat)/nf,nf)
    return dat

def animate(irange,n=100,nf=4):
    dat=[]
    times=[]
    for i,j in enumerate(irange):
        dat.append(load_time(j,n=n,nf=nf))
        times.append(i)
    lineq,linep,fig,axes = plot_time(dat[0])

    for d,t in zip(dat[1:],times[1:]):
        plt.pause(.0000001)
        update_plot(d,lineq,linep)
        axes[0].set_title('%d'%t)
        #plt.draw()
        fig.canvas.draw()

def load_energy(irange,n=1e3):
    dat0 = load_time(0,n=n)[:,-1]
    print dat0.shape
    if irange[0] == 0:
        irange = irange[1:]
    s = (len(dat0),1)
    dat0 = dat0.reshape(s)
    dat = np.ones(dat0.shape)
    norm = 1./dat0
    norm[dat0==0] = 1
    for i in irange:
        dat = np.hstack( (dat,load_time(i,n=n)[:,-1].reshape(s) *norm) )
    return dat
def plot_time(dat,ax=None):
    if ax is None:
        fig,axes=plt.subplots(1,2,figsize=(15,10))

    lineq,=axes[0].plot(dat[:,0],dat[:,1],'.',markersize=5)
    linep,=axes[1].plot(dat[:,2],dat[:,3],'.',markersize=5)
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    axes[1].set_xlabel('vx')
    axes[1].set_xlabel('vy')
    return lineq,linep,fig,axes

def update_plot(dat,lineq,linep):
    lineq.set_data( dat[:,0],dat[:,1])
    linep.set_data( dat[:,2],dat[:,3])
    return
