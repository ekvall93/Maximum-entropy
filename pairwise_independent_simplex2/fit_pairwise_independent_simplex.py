"""Fitting for the pairwise model. Written by Markus Ekvall 2018-07-09."""
import numpy as np
from multiprocessing import Pool
import scipy.sparse as sp
import sys
import types
from pairwise_independent_simplex_sampling import \
   pairwise_independent_simplex_sample
from get_simplex_train import get_simplex_train

if sys.version_info[0] < 3:
    import copy_reg as copyreg
else:
    import copyreg


def _pickle_method(m):
    """Handle the pickles for the multiprocessing."""
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

    def __init__(self, train, J0, psi, learning_rate,
                 M_samples, n_jobs, test=None, save_loss=False):
        """
        __init__.

        Parameters
        ----------
        train : ndarray
            Training data
        J0 : ndarray
            Intial guess
        learning_rate : float
           How much of the gradient that will be used
        M_samples : int
            Number of samples used in Monte carlo simulations
        n_jobs : int
            Number of jobs in parallel
        test : ndarray
            Test data
        save_loss : boolean
            Save the loss
        Methods
        -------
        prepare_sampling
        grad_descent
        fit_pairwise
        Dloss
        __getstate__

        """
        # Sanity-checks
        [n, m] = np.shape(train)

        assert m == np.shape(J0)[0] and m == np.shape(J0)[1], \
            "The shape of J should be ({2},{2}) and not ({0},{1})." \
            .format(np.shape(J0)[0], np.shape(J0)[1], m)

        assert (float(M_samples)/n_jobs).is_integer(), \
            "Adjust M_samples ({0}), and n_jobs ({1}) so the quotient is" \
            " int. Currenty M_samples/n_jobs = {2}." \
            .format(M_samples, n_jobs, float(M_samples)/n_jobs)
        [n, m] = np.shape(train)
        self.n = n
        self.m = m
        self.J = J0
        self.psi = psi
        self.train = train
        self.test = test
        self.M_samples = M_samples
        self.learning_rate = learning_rate
        if self.test is not None:
            print("Testdata included")
            self.emp_cov_test = np.dot(test.T, test).astype(float)/n
            test_simplices = get_simplex_train(test)
            self.test_rigidity = np.mean(test_simplices, axis=0)
            if save_loss:
                self.test_error = []
        else:
            print("No testdata included")
            self.emp_cov_test = None
        if save_loss:
            print("Saving the errors")
            self.train_error = []
        self.save_loss = save_loss
        self.emp_cov_train = np.dot(train.T, train).astype(float)/n
        self.simplices = get_simplex_train(train)
        self.rigidity = np.mean(self.simplices, axis=0)
        self.pool = Pool(n_jobs)
        self.n_pools = self.pool._processes
        self.samples = None

    def prepare_sampling(self, k):
        """
        Prepare the data before monte carlo sampling.

        Parameters
        ----------
        k : int
           Number of parallel proccees

        """
        np_arr = np.squeeze(self.samples_batch[:, :, k])
        sp_csc = sp.csc_matrix(np_arr)
        np_simp_arr = np.squeeze(self.simplex_train[:, :, k])
        simplex_train = sp.csc_matrix(np_simp_arr)
        if self.first_batch:
            # Burn-in period to get to a stable phase before sampling.
            steps = self.burn_in
        else:
            # Sampling period.
            steps = self.gibbs_steps
        samples, simplex_train = pairwise_independent_simplex_sample(
                                sp_csc, simplex_train, self.J, self.psi, steps)
        batch = np.append(samples, simplex_train, axis=1)
        return batch

    def grad_descent(self, pars0, iter, gibbs_steps):
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
        # Get the intiial error (gradient
        g = self.Dloss(pars0, gibbs_steps)
        iteration = 0
        limit = 0.01
        import time
        start_time = time.time()
        print("---- STARTING GRADIENT ASCENT ----")

        while iteration < iter:
            # pars = pars - self.learning_rate*g
            pars = pars - self.learning_rate*g
            # Get the error (gradient)
            g = self.Dloss(pars, gibbs_steps)
            iteration = iteration + 1
            if limit < float(iteration)/iter:
                print("||Progress||: "+str(float(limit*100))+" %",
                      "||Maximal error||: "+str(round(np.max(np.abs(g)), 6)),
                      "||Time left||: %s s" %
                      str(np.around(float(time.time()
                          - start_time)*(1/(float(iteration)/iter)-1), 2)))
                limit = float(iteration)/iter+0.1

        print("||Progress||: 100 %",
              "||Maximal error||: "+str(round(np.max(np.abs(g)), 6)),
              "||Time left||: %s s" %
              str(np.around(float(time.time()
                  - start_time)*(1/(float(iteration)/iter)-1), 2)))
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
        J0_lin = np.append(self.J.flatten(), self.psi.flatten())
        # get the number of neurons
        high = np.shape(self.train)[0]
        # randomly samples M_samples from the data
        randi = np.random.randint(0, high=high, size=(self.M_samples,))
        self.samples = self.train[randi, :]
        self.simplex_samples = self.simplices[randi, :]
        self.samples = np.array(self.samples[:int(
            np.floor(np.shape(self.samples)[0]/self.n_pools)*self.n_pools), :])
        self.simplex_samples = np.array(self.simplex_samples[:int(np.floor(
           np.shape(self.simplex_samples)[0]/self.n_pools)*self.n_pools), :])
        self.samples_batch = np.zeros((self.M_samples/self.n_pools,
                                       self.m, self.n_pools))
        self.simplex_train = np.zeros((self.M_samples/self.n_pools,
                                       np.shape(self.psi)[0], self.n_pools))
        # Divide the data into n_pools set for parallized monte carlo
        for k in range(0, self.n_pools):
            idx_samples = range(k*(self.M_samples/self.n_pools),
                                ((k+1)*self.M_samples/self.n_pools))
            self.samples_batch[:, :, k] = self.samples[idx_samples, :]
            self.simplex_train[:, :, k] = self.simplex_samples[idx_samples, :]
        # Approximately needed burn-in-step
        self.burn_in = 10*gibbs_steps
        self.first_batch = True
        # Get the intial samples after burn-in
        print("---- INITIATE BURN-IN ----")
        batch = np.array(self.pool.map(self.prepare_sampling,
                         range(0, self.n_pools)))
        batch = np.moveaxis(batch, 0, -1)
        self.samples_batch = batch[:, :self.m].reshape(
                             self.M_samples/self.n_pools, self.m,
                             self.n_pools)
        self.simplex_train = batch[:, self.m:].reshape(
                             self.M_samples/self.n_pools, self.m/3,
                             self.n_pools)

        self.first_batch = False
        # Gradient ascent to learn paramters
        J_lin = self.grad_descent(J0_lin, iter, gibbs_steps)

        # Get model correlations matrix
        J = J_lin[:self.m**2].reshape((self.m, self.m))
        psi = J_lin[self.m**2:]
        mod_cov = self.model_cov.reshape((self.m, self.m))
        if self.save_loss:
            if self.test is not None:
                return (J, self.emp_cov_train, mod_cov, self.samples,
                        psi, self.simp_mean, self.rigidity, self.train_error,
                        self.test_error)
            else:
                return (J, self.emp_cov_train, mod_cov, self.samples,
                        psi, self.simp_mean, self.rigidity, self.train_error)
        else:
            return (J, self.emp_cov_train, mod_cov, self.samples,
                    psi, self.simp_mean, self.rigidity)

    def Dloss(self, J_lin, gibbs_steps):
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
        J = np.array(J_lin[:self.m**2])
        self.gibbs_steps = gibbs_steps
        self.J = J.reshape((self.m, self.m))
        self.psi = np.array(J_lin[self.m**2:]).reshape(self.m/3,)
        model_covs = np.zeros((self.m**2, self.n_pools))
        # NB need to chage this later on
        simplex_means = np.zeros((self.m/3, self.n_pools))
        batch = np.array(self.pool.map(self.prepare_sampling,
                         range(0, self.n_pools)))
        batch = np.moveaxis(batch, 0, -1)
        self.samples_batch = batch[:, :self.m].reshape(
                             self.M_samples/self.n_pools, self.m, self.n_pools)
        self.simplex_train = batch[:, self.m:].reshape(
                             self.M_samples/self.n_pools, self.m/3,
                             self.n_pools)
        n, m, k = np.shape(self.samples_batch)
        # Combine all montecarlo
        for i in range(0, k):
            model_cov_tmp = np.dot(self.samples_batch[:, :, i].T,
                                   self.samples_batch[:, :, i])/n
            model_covs[:, i] = model_cov_tmp.flatten()
            simplex_means[:, i] = np.mean(self.simplex_train[:, :, i],
                                          axis=0).flatten()
        self.model_cov = np.sum(model_covs, axis=1)/self.n_pools
        self.simp_mean = np.sum(simplex_means, axis=1)/self.n_pools
        # Calculate the difference in mpdel correlation and emperical
        # correlation
        Dloss_n = (-self.model_cov + self.emp_cov_train.flatten())
        Dloss_s = (-self.simp_mean + self.rigidity.flatten())
        Dloss_train = np.append(Dloss_n, Dloss_s)
        if self.save_loss:
            g_t = round(np.max(np.abs(Dloss_train)), 6)
            self.train_error.append(g_t)
        if self.emp_cov_test is not None:
            Dloss_test = (-self.model_cov + self.emp_cov_test.flatten())
            Dloss_s_test = (-self.simp_mean + self.test_rigidity.flatten())
            Dloss_test = np.append(Dloss_test, Dloss_s_test)
            if self.save_loss:
                g_te = round(np.max(np.abs(Dloss_test)), 6)
                self.test_error.append(g_te)
        return Dloss_train

    def __getstate__(self):
        """Use to make multi-processing work."""
        self_dict = self.__dict__.copy()
        del self_dict['pool']
        return self_dict
