import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

class GBO:
    '''
    See Frazier and Wang (2015) Bayesian optimization for materials design
    
    :param func:
        The cost function to be maximized in the form y* = func(x^*)
    :param nDim:
        Dimension of func's input variable
    :param X:
        array like with shape (nObs,nDim)
        Current states where we have observed self.func
    :param y:
        array line with shape (nObs,1)
        Current observations of self.func, i.e., y = self.func(X)
    :param bounds:
        array line with shape (nDim,2)
        lower bound and upper bound for parameters
    '''
    
    def __init__(self,func,nDim,X,y,bounds):
        self.func = func
        self.nDim = nDim
        assert np.array(X).shape[1] == nDim ,\
            'X must have shape (nObs,nDim)'
        self.X = np.array(X)
        assert np.array(y).shape == (np.array(X).shape[0],1) ,\
            'y must have shape (nObs,1) where X.shape = (nObs,nDim)'
        self.y = np.array(y)
        assert np.array(bounds).shape == (nDim,2),\
            'bounds must have shape (nDim,2)'
        self.bounds = np.array(bounds)
        
    def EI(self,x):
        # Returns negative expected improvement for minimization algo
        x = x.reshape((1,self.nDim))
        f_star = np.max(self.y)
        mu_x , std_x = self.GP.predict(x,return_std=True)
        return(-(((mu_x - f_star)*norm.cdf((mu_x-f_star)/std_x)+
                  std_x*norm.pdf((mu_x-f_star)/std_x))[0,0]))
    
    
    def Expected_Improvement(self,min_method ='L-BFGS-B',nx0 = 20):
        '''
        :param min_method:
            scipy.optimize minimization method to minimize the expected imporvement function
            ********
            I should add the option to use the EI gradient
            ********
        :param nx0:
            The number of samples used to find a good initial condition for the minimizer
        '''
        if not hasattr(self,'GP'):
            raise AttributeError('Must define (unfitted) Gaussian process')

        self.GP.fit(self.X,self.y)
        
        x0 = self.X[np.argmax(self.y),:]
        y0 = self.EI(x0)
        for kk in range(nx0):
            x0_ = np.random.uniform(self.bounds[:,0],self.bounds[:,1])
            y0_ = self.EI(x0_)
            if y0_ < y0: x0,y0 = x0_,y0_
        try:
            res = minimize(self.EI,x0,method = min_method,bounds = self.bounds)
        except Warning:
            'WARNING:\n'
            print(res.message)
        self.X = np.concatenate((self.X,res.x.reshape((1,self.nDim))))
        self.y = np.concatenate((self.y,self.func(res.x).reshape((1,1))))
        
        
    