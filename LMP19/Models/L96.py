import numpy as np
from ..RK import RK4

class L96_1:
    '''
    Lorenz 96 single scale system.
    This system is listed in Eq. (1) of Lorenz 1996 "Predictability, a problem partial solved."
    
    :param nK:
        The number of states X.
    '''
    
    def __init__(self,nK):
        self.nK = nK
        assert self.nK > 4, \
            "Must have more than four X states."
                   
    def f(self,x,theta=10):
        dX = np.zeros(self.nK)
        
        for kk in range(self.nK):
            dX[kk] = -x[kk-2]*x[kk-1]+x[kk-1]*x[(kk+1)%self.nK]-x[kk]+theta
        return(dX)
    
    def M(self,x,theta=10,dt=0.01):
        return(RK4(x,self.f,dt,theta))

    def get_data(self,x0=None,theta=10,nSteps = 50,dt=0.01):
        
        if x0 is None:
            x0 = np.random.uniform(-5*np.ones(self.nK),5*np.ones(self.nK))            
            for _ in range(1000):
                x0 = self.M(x0,theta=theta,dt=dt)

        x_path = np.zeros((self.nK,nSteps))
        x_path[:,0] = np.ravel(x0)

        for kk in range(1,nSteps):
            x_path[:,kk] = self.M(x_path[:,kk-1],theta=theta,dt=dt)
        return(x_path)

    def get_L(self,loc_val):
        L = np.zeros((self.nK,self.nK))
        for kk in range(self.nK):
            for jj in range(self.nK):
                p1 = np.array((np.cos(2*np.pi*kk/self.nK),np.sin(2*np.pi*kk/self.nK)))
                p2 = np.array((np.cos(2*np.pi*jj/self.nK),np.sin(2*np.pi*jj/self.nK)))

                L[kk,jj] = np.exp(-np.linalg.norm(p1-p2)/loc_val)
                L[jj,kk] = L[kk,jj]          
        return(L)

class L96_23:
    '''
    Lorenz 96 coupled fast/slow system.
    This system are listed in Eqs. (2) and (3) of Lorenz 1996 "Predictability, a problem partial solved."
    Also see: Schneider, T., Lan, S., Stuart, A., & Teixeira, J. (2017). Earth system modeling 2.0
    
    :param nK:
        The number of 'slow' states X.
    :param nJ:
        The number of 'fast' states Y coupled to each X state.
    '''
    
    def __init__(self,nK,nJ):
        self.nK = nK
        assert self.nK > 4, \
            "Must have more than four X states."
        self.nJ = nJ
        assert self.nJ > 4, \
            "Must have more than four Y states."
        self.nDim = nK*(1+nJ)

    def X_Y_to_XY(self,X,Y):
        
        nSteps = X.shape[1]
        
        XY = np.zeros((self.nK+self.nK*self.nJ,nSteps))
        XY[:self.nK,:] = X
        for kk in range(self.nK):
            XY[self.nJ*kk+self.nK:self.nJ*(kk+1)+self.nK,:] = Y[:,kk,:]
        return(XY)

    def XY_to_X_Y(self,XY):

        nSteps = XY.shape[1]
        
        X = np.zeros((self.nK,nSteps))
        Y = np.zeros((self.nJ,self.nK,nSteps))

        X = XY[:self.nK,:]
        for kk in range(self.nK):
            Y[:,kk,:] = XY[self.nJ*kk+self.nK:self.nJ*(kk+1)+self.nK,:]
        return(X,Y)
                   
    def f(self,x,theta=[10,10,10,1]):
        b,c,F,h = theta 
        
        X = x[:self.nK]
        Y = x[self.nK:]
        
        dX = np.zeros(self.nK)
        dY = np.zeros((self.nJ*self.nK))
        
        for kk in range(self.nK):
            dX[kk] = -X[(kk-1)%self.nK]*(X[(kk-2)%self.nK]-X[(kk+1)%self.nK])-X[kk]-h*c*np.sum(Y[kk*self.nJ:(kk+1)*self.nJ])/b+F
            for jj in range(kk*self.nJ,(kk+1)*self.nJ):
                dY[jj]= -c*b*Y[(jj+1)%(self.nJ*self.nK)]*(Y[(jj+2)%(self.nJ*self.nK)]-Y[(jj-1)%(self.nJ*self.nK)])-c*Y[jj]+c*h*X[kk]/b
        return(np.concatenate((dX,dY)))
    
    
    def M(self,x,theta=[10,10,10,1],dt=0.001):
        return(RK4(x,self.f,dt,theta))

    def get_data(self,x0=None,theta=[10,10,10,1],nSteps = 500,dt=0.001):
        
        if x0 is None:
            X = np.random.uniform(-5*np.ones(self.nK),5*np.ones(self.nK))
            Y = 5*np.random.randn(self.nJ*self.nK)
            x0 = np.concatenate((X,Y))
            
            for _ in range(10000):
                x0 = self.M(x0,theta=theta,dt=dt)

        x_path = np.zeros((self.nK+self.nK*self.nJ,nSteps))
        x_path[:,0] = np.ravel(x0)

        for kk in range(1,nSteps):
            x_path[:,kk] = self.M(x_path[:,kk-1],theta=theta,dt=dt)
        
        return(x_path)

    def get_L(self,loc_val):
        L = np.zeros((self.nDim,self.nDim))
        for kk in range(self.nK):
            for jj in range(self.nK):
                p1 = np.array((np.cos(2*np.pi*kk/self.nK),np.sin(2*np.pi*kk/self.nK)))
                p2 = np.array((np.cos(2*np.pi*jj/self.nK),np.sin(2*np.pi*jj/self.nK)))

                L[kk,jj] = np.exp(-np.linalg.norm(p1-p2)/loc_val)
                L[jj,kk] = L[kk,jj]

        for k1 in range(self.nK):
            for k2 in range(self.nK):
                L[k1,k2*self.nJ+self.nK:(k2+1)*self.nJ+self.nK]+=L[k1,k2]
        L[self.nK:,:self.nK] = L[:self.nK,self.nK:].T

        for kk in range(self.nK,self.nDim):
            for jj in range(self.nK,self.nDim):
                p1 = np.array((np.cos(2*np.pi*(kk-self.nK)/self.nDim),np.sin(2*np.pi*(kk-self.nK)/self.nDim)))
                p2 = np.array((np.cos(2*np.pi*(jj-self.nK)/self.nDim),np.sin(2*np.pi*(jj-self.nK)/self.nDim)))

                L[kk,jj] = np.exp(-np.linalg.norm(p1-p2)/loc_val)
                L[jj,kk] = L[kk,jj]            
        return(L)