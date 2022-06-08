import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import pandas as pd

#--------------------------------------------------------------------------------------
## ~~~ FUNCTIONS FOR NEWMARK SCHEME ~~~
def newmark(N, M, w, tau = None, h = None):
    #Newmark Scheme for the wave equation, x in (0,1) and t in (0,1)
    #dt^2 u(x,t) - dx^2 u(x,t) = 0
    #u(0,t) = u(1,t) = 0
    #u(x,0) = w(x) 
    #parameters:
    #N        : number of internal steps on the interval [0,1]
    #h        : space step
    #M        : number of internal steps on the interval [0,1]
    #tau      : time step
    #t        : current time
    #u0     : N-vector, u0(i) is an approximation of u(x_i,t_n-1)
    #u1     : N-vecteur, u1(i) is an approximation of u(x_i,t_n)
    #u2     : N-vecteur, u2(i) is an approximation of u(x_i,t_n+1)
    #w      : function, initial condition.
    #output : Newmark solution at time 1.
    if tau==None :
        tau=1/M #since t is from 0 to 1, and for the stability. So that we have tau = h.
    
    if h==None : 
        h=1/(N+1)

    lambda_=(tau/h)**2
    u0=np.zeros(N)
    for i in range(N):
        u0[i]=w((i+1)*h) #from h to N*h
    
    u1=np.zeros(N)
    u1[0]=(1-lambda_)*u0[0]+lambda_/2*u0[1]
    
    for i in range(1,N-1):
        u1[i]=(1-lambda_)*u0[i]+lambda_/2*(u0[i-1]+u0[i+1])
    
    u1[N-1]=(1-lambda_)*u0[N-1]+lambda_/2*u0[N-2]
    
    #Newmark scheme
    t=tau
    u2=np.zeros(N)
    for n in range(1,M+1):
        t=t+tau
        u2[0]=2*(1-lambda_)*u1[0]+lambda_*u1[1]-u0[0]
        for i in range(1,N-1):
            u2[i]=2*(1-lambda_)*u1[i]+lambda_*(u1[i-1]+u1[i+1])-u0[i]
        u2[N-1]=2*(1-lambda_)*u1[N-1]+lambda_*u1[N-2]-u0[N-1]
        
        #actualize the sol
        for i in range(N):
            u0[i]=u1[i]
            u1[i]=u2[i]
    return u2, t

def sol_given_fct2(fct):
    #given the initial condition fct, i.e. u(x,0)=fct(x), it returns the exact solution of the wave equation.
    def uex(x,t):
        return 0.5*(fct(x-t)+fct(x+t))
    return uex
#--------------------------------------------------------------------------------------

## ~~~ FUNCTIONS TO CREATE SUM OF SINE FROM PARAMETERS ~~~~
def fct(Mus, K = 1):
#returns the function sum of mu_j / j^K * sin(j*pi*x)
    J=np.arange(1,len(Mus)+1)
    def somme(x):
        scal=Mus/J**K
        vec=scal*np.sin(np.pi*J*x)
        return sum(vec)
    return somme

def int_MC(N_mu):
    #returns the integral I of 2^{N_mu} int_[-1,1]^{N_mu} int_[0, 1] |u_exact(x, mu)|^2 dx dmu, for K = 1.
    J =np.arange(1,N_mu+1)
    return 1/6*np.sum(1/J**2)
#--------------------------------------------------------------------------------------

## ~~~ UTILITIES ~~~~
def plot_sol(N,u):
    #plot the solution u (vector of size N) of N points.
    h=1/(N+1)
    H=np.zeros(N)
    for i in range(N):
        H[i]=(i+1)*h 
    plt.plot(H, u)

def to_latex(N_hs, N_Ms, err):
    #to create a table in latex
    hs = 1/(N_hs + 1)
    hs_extend = hs[0]*np.ones(len(N_Ms))
    N_Ms_extend = N_Ms
    for l in range(len(hs)-1):
        hs_extend = np.concatenate((hs_extend, hs[l+1]*np.ones(len(N_Ms))))   
        N_Ms_extend = np.concatenate((N_Ms_extend, N_Ms))

    data = pd.DataFrame(dict(h = hs_extend, N_M = N_Ms_extend, Error = err.reshape(-1)))
    data[['Error']] = data[['Error']].applymap('{:.2E}'.format)
    data[['h']] = data[['h']].applymap('{:.3f}'.format)
    data[['N_M']] = data[['N_M']].applymap('{:.0f}'.format)
    return data.to_latex(index=False)

def export_legend(legend, filename):
    fig  = legend.figure
    fig.canvas.draw()
    bbox  = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(filename, dpi = "figure", bbox_inches=bbox)
#--------------------------------------------------------------------------------------

## ~~~ Neural Network ~~~~
class Net(nn.Module):
        def __init__(self, N_HL, N_n, N_h, N_mu):
            super(Net, self).__init__()
            # an affine operation: y = Wx + b
            self.hidden = nn.ModuleList()
            self.fc1 = nn.Linear(N_mu, N_n)
            for k in range(N_HL-1):
                self.hidden.append(nn.Linear(N_n, N_n))
            self.fc4 = nn.Linear(N_n, N_h)

        def forward(self, x):
            x = F.relu(self.fc1(x))
            for l in self.hidden:
                x = F.relu(l(x))
            x = self.fc4(x)
            return x

class Inputs(Dataset):
    #generate N_x samples, each with N_mu parameters, and compute the solution y given by the Newmark scheme with N time steps.
    def __init__(self, N_x, N_mu, N, K = 1, device = 'cpu'):
        x_np=np.zeros((N_x, N_mu))
        print("x shape : ", x_np.shape)
        M=N
        h=1/(N+1)
        H=np.zeros(N)
        for i in range(N):
            H[i]=(i+1)*h 
        y_np=np.zeros((N_x, N))
        #generate the datas and the solutions to learn from (training set)
        for i in range(N_x):
            Mus=np.random.uniform(-1,1,N_mu)
            x_np[i,:]=Mus  
            somme=fct(Mus, K)
            y_np[i,:], _ =newmark(N,M,somme, tau = h, h = h)        
        print("y shape : ", y_np.shape)

        self.x= torch.from_numpy(x_np).float().to(device)
        self.y = torch.from_numpy(y_np).float().to(device)
        self.n_samples=N_x
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples