#from __future__ import division
import numpy as np
from multiprocessing import Pool
from itertools import product
from contextlib import contextmanager
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from numpy import genfromtxt
import time
import sys
import types
from pairwise_sampling import sample_pairwise
#Difference between Python3 and 2
if sys.version_info[0] < 3:
    import copy_reg as copyreg
else:
    import copyreg

def _pickle_method(m):
    """Handles the pickles for the multiprocessing"""
    class_self = m.im_class if m.im_self is None else m.im_self
    return getattr, (class_self, m.im_func.func_name)
copyreg.pickle(types.MethodType, _pickle_method)



class fit_pairwise(object):
    """
    Maximum entropy class.

    Parameters
    ----------
    data : ndarray
        Spiketrains/simplextrains (#time_points,#neurons/simplex)
    J0 : ndarray,None
        Intial guess for correlations.
    learning_rate : float
    M_samples : int
        Number of samples used in Monte carlo simulations
    n_jobs : int
        Number of jobs in parallel
    Methods
    -------
    prepare_sampling
    grad_descent
    fit_pairwise
    Dloss
    __getstate__
    """
    def __init__(self,train, J0, learning_rate,M_samples,n_jobs,test=None,
                 save_loss=False):
        # Do basic checks on the inputs.
        [n,m] = np.shape(train)

        assert m == np.shape(J0)[0] and m == np.shape(J0)[1], \
            'The shape of J should be ({2},{2}) and not ({0},{1}).' \
            .format(np.shape(J0)[0],np.shape(J0)[1],m)

        assert (float(M_samples)/n_jobs).is_integer(), \
                'Adjust M_samples ({0}), and n_jobs ({1}) so the quotient is' \
                 ' int. Currenty M_samples/n_jobs = {2}.' \
                .format(M_samples,n_jobs,float(M_samples)/n_jobs)


        [n,m] = np.shape(train)
        self.n = n
        self.m = m
        self.J = J0
        self.train = train
        self.test = test
        self.M_samples = M_samples
        self.learning_rate = learning_rate
        #NB Will change to the one below when I know when the code works.
        #n_pools = pool._processes
        if self.test is not None:
            print("Testdata included")
            self.emp_cov_test = np.dot(test.T,test).astype(float)/n
            if save_loss:
                self.test_error = []
        else:
            print("No testdata included")
            self.emp_cov_test = None
        if save_loss:
            print("Saving the errors")
            self.train_error = []
        self.save_loss = save_loss
        self.emp_cov_train = np.dot(train.T,train).astype(float)/n
        self.pool = Pool(n_jobs)
        self.n_pools = self.pool._processes
        self.samples =None

    def prepare_sampling(self,k):
        """
        Prepare the data before monte carlo sampling.
        Parameters
        ----------
        k : int
           Number of parallel proccees
        """
        #Create two different arrays. Gain in speed, but lose memory.
        np_arr = np.squeeze(self.samples_batch[:,:,k])
        sp_csc =sp.csc_matrix(np_arr)
        if self.first_batch:
            #Burn-in period to get to a stable phase before sampling.
            steps = self.burn_in
        else:
            #Sampling period.
            steps = self.gibbs_steps
        #Return MC samples.
        samples = sample_pairwise(sp_csc,self.J,steps)
        return samples

    def grad_descent(self,pars0, iter,gibbs_steps):
        """
        Prepare the data before monte carlo sampling.
        Parameters
        ----------
        pars0 : ndarry
           Intiial parameters
        iter : int
           Number of steps of gradient descent
        gibbs_steps : int
           Number of monte carlo iterations
        """
        pars = pars0
        #Get the intiial error (gradient
        g = self.Dloss(pars0, gibbs_steps)
        iteration = 0
        while iteration <= iter:
            pars = pars - self.learning_rate*g
            #pars = pars - np.exp(-iteration)*g
            #Get the error (gradient)
            g = self.Dloss(pars, gibbs_steps)
            iteration = iteration + 1
            print("Iteration: ",iteration," ||Gradient||: ",round(np.max(np.abs(g)),6))
        return pars


    def fit_pairwise(self, iter, gibbs_steps):
        """
        Prepare the data before monte carlo sampling.
        Parameters
        ----------
        iter : ndarry
           Number of iter
        iter : int
           Number of steps of gradient descent
        gibbs_steps : int
           Number of monte carlo iterations
        """
        np.random.seed(seed=1)
        #randi = np.array(genfromtxt('fisrt_sample.csv', delimiter=',')-1).astype(int)

        #J is J0 in beginning
        J0_lin = self.J.flatten()
        #get the number of neurons
        high = np.shape(self.train)[0]
        #randomly samples M_samples from the data
        randi = np.random.randint(0, high=high, size=(self.M_samples,))
        self.samples = self.train[randi,:]
        self.samples = np.array(self.samples[:int(np.floor(np.shape(self.samples)[0]/self.n_pools)*self.n_pools),:])
        self.samples_batch = np.zeros((self.M_samples/self.n_pools, self.m, self.n_pools))
        #Divide the data into n_pools set for parallized monte carlo
        for k in range(0,self.n_pools):
            idx_samples = range(k*(self.M_samples/self.n_pools),((k+1)*self.M_samples/self.n_pools))
            self.samples_batch[:,:,k] = self.samples[idx_samples,:]
        #Approximately needed burn-in-step
        self.burn_in = 10*gibbs_steps
        self.first_batch = True
        #Get the intial samples after burn-in
        samples_batch = np.array(self.pool.map(self.prepare_sampling,range(0,self.n_pools)))
        self.first_batch = False
        self.samples_batch = np.moveaxis(samples_batch,0,-1)
        #Gradient descent to learn paramters
        J_lin = self.grad_descent(J0_lin, iter, gibbs_steps)
        #Get model correlations matrix
        J = J_lin.reshape((self.m,self.m))
        #m_cov = self.model_cov.reshape((self.m,self.m))
        if self.save_loss:
            if self.test is not None:
                return J, self.emp_cov_train ,self.model_cov_tmp,self.samples, self.train_error, self.test_error
            else:
                return J, self.emp_cov_train ,self.model_cov_tmp,self.samples, self.train_error
        else:
            return J, self.emp_cov_train ,self.model_cov_tmp,self.samples




    def Dloss(self,J_lin, gibbs_steps):
        """
        Prepare the data before monte carlo sampling.
        Parameters
        ----------
        J_lin : ndarry
           paremters
        iter : int
           Number of steps of gradient descent
        gibbs_steps : int
           Number of monte carlo iterations
        """
        np.random.seed(seed=1)
        J_lin = np.array(J_lin)
        self.gibbs_steps = gibbs_steps
        self.J = J_lin.reshape((self.m,self.m))
        model_covs = np.zeros((self.m**2, self.n_pools))
        samples_batch = np.array(self.pool.map(self.prepare_sampling,
                                             range(0,self.n_pools)))
        self.samples_batch = np.moveaxis(samples_batch,0,-1)
        n,m,k = np.shape(self.samples_batch)
        # Combine all montecarlo
        for i in range(0,k):
            self.model_cov_tmp = np.dot(self.samples_batch[:,:,i].T,self.samples_batch[:,:,i])/n
            model_covs[:,i] = self.model_cov_tmp.flatten()
        model_cov = np.sum(model_covs, axis=1)/self.n_pools;
        #Calculate the difference in mpdel correlation and emperical correlation
        Dloss_train = (-model_cov + self.emp_cov_train.flatten())

        if self.save_loss:
            self.train_error.append(max(Dloss_train))

        if self.emp_cov_test is not None:
            Dloss_test = (-model_cov + self.emp_cov_test.flatten())
            if self.save_loss:
                self.test_error.append(Dloss_test)

        return Dloss_train

    def __getstate__(self):
        """Used to make multi-processing work."""
        self_dict = self.__dict__.copy()
        del self_dict['pool']
        return self_dict
