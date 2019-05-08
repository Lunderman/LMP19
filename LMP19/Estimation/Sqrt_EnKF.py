import numpy as np

class Sqrt_EnKF:
    '''
    Square root ensemble Kalman filter
    See
    
    :param M:
        The model that advances the state one step in time, the only imput should be the current state x.
    :param nDim:
        The state dimension
    :param H:
        Observation matrix
    :param H:
        Observation covariance
    :param L:
        Localization matrix
    :param inflation:
        Inflation scalar
    '''
    
    def __init__(self,M,nDim,H=None,R=None,L=None,inflation=0):
        self.M = M
        self.nDim = nDim
        if H is None:
            self.H = np.identity(nDim)
        else:
            self.H = H
        self.obs_Dim = self.H.shape[0]
        if R is None:
            self.R = np.identity(self.obs_Dim)
        else:
            assert R.shape == ((self.obs_Dim,self.obs_Dim)), \
                'R must be a square matrix with dimension equal to H.shape[0].'
            self.R = R
        if L is None:
            self.L = np.ones((nDim,nDim))
        else:
            self.L = L
        self.inflation = inflation
        

    def run(self,x,y):
        self.nE = x.shape[1]
        x_f = np.zeros((self.nDim,self.nE))
        for ee in range(self.nE):
            x_f[:,ee] = self.M(x[:,ee])

        mu_f = np.mean(x_f,axis=1 ).reshape(self.nDim,1)
        x_f = mu_f + np.sqrt(1+self.inflation)*(x_f-mu_f)
        
        X_f = (1/np.sqrt(self.nE-1))*(x_f-mu_f)
        P_f = self.L*np.dot(X_f,X_f.T)
        
        K = np.linalg.solve((np.dot(np.dot(self.H,P_f),self.H.T)+self.R).T , (np.dot(P_f,self.H.T)).T  ) .T
        y_po = y.reshape((self.obs_Dim,1)) + np.random.multivariate_normal(np.zeros(self.obs_Dim),self.R,self.nE).T
        
        x_a = np.array([x_f[:,ee]+np.dot(K,y_po[:,ee]-np.dot(self.H,x_f[:,ee])) for ee in range(self.nE)]).T  
        return(x_a)