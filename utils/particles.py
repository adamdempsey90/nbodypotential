import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree
from sklearn.neighbors import KDTree


def load_ic(nf=4):
    dat = np.fromfile('ic/points.dat')
    dat = dat.reshape(len(dat)/nf,nf)
    return dat

def load_time(t,n=100,nf=4):
    dat=np.fromfile('outputs/particles_{0:d}.dat'.format(t))
    dat = dat.reshape(len(dat)/nf,nf)
    return dat
def load_times(irange,nf=4):

    dat = np.fromfile('outputs/particles_{0:d}.dat'.format(irange[0]))
    n = len(dat)/nf
    dat = dat.reshape(n,nf)
    for i in irange[1:]:
        dat = np.vstack(( dat, np.fromfile('outputs/particles_{0:d}.dat'.format(i)).reshape(n,nf)))
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
def plot_time(dat,ax=None,pointstyle='.b'):
    if ax is None:
        fig,axes=plt.subplots(1,2,figsize=(15,10))

    lineq,=axes[0].plot(dat[:,0],dat[:,1],pointstyle,markersize=5)
    linep,=axes[1].plot(dat[:,2],dat[:,3],pointstyle,markersize=5)
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    axes[1].set_xlabel('vx')
    axes[1].set_xlabel('vy')
    return lineq,linep,fig,axes

def update_plot(dat,lineq,linep):
    lineq.set_data( dat[:,0],dat[:,1])
    linep.set_data( dat[:,2],dat[:,3])
    return



def build_tree(points):
    return cKDTree(points)

def find_neighbors(tree_a, tree_b, bandwidth=.1):
    return np.array(tree_a.query_ball_tree(tree_b,bandwidth))


def log_likelihood(tree_a, tree_b,bandwidth=.1):
    s = tree_a.data.shape
    norm = (2*np.pi)**(-s[1]/2.)
    norm /= s[0]
    return np.log(np.array([norm*np.exp(-.5*(((tree_a.data[i,:]-tree_b.data[x])/bandwidth)**2 ).sum(axis=1)).sum() for i,x in enumerate(np.array(tree_a.query_ball_tree(tree_b,bandwidth)))]).sum())

def sklog_likelihood(points_a, points_b,sigma=.1,norm=1.0):
    return (norm*KDTree(points_a).kernel_density(points_b,h=sigma,kernel='gaussian',return_log=True)).sum()

def compute_ll(times, fname_ref,sigma=.1,norm=1.0,nd=4):
    dat_r = np.fromfile(fname_ref)
    dat_r = dat_r.reshape(len(dat_r)/nd,nd)

    norm = 1./len(times)
    dat = load_times(times,nf=nd)

    return -sklog_likelihood(dat,dat_r, sigma=sigma,norm=norm)

