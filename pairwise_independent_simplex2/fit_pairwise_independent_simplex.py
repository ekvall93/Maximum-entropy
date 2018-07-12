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
    MC_samples : int
        Number of samples used in Monte carlo simulations
    n_jobs : int
        Number of jobs in parallel
    Methods
    -------
    sample_preperation
    gradient_descent
    fit_pairwise
    error
    __getstate__

    """

    def __init__(self, train, J0, psi, learning_rate,
                 MC_samples, n_jobs, test=None, save_loss=False):
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
        MC_samples : int
            Number of samples used in Monte carlo simulations
        n_jobs : int
            Number of jobs in parallel
        test : ndarray
            Test data
        save_loss : boolean
            Save the loss
        Methods
        -------
        sample_preperation
        gradient_descent
        fit_pairwise
        error
        __getstate__

        """
        # Sanity-checks
        [n, m] = np.shape(train)

        assert m == np.shape(J0)[0] and m == np.shape(J0)[1], \
            "The shape of J should be ({2},{2}) and not ({0},{1})." \
            .format(np.shape(J0)[0], np.shape(J0)[1], m)

        assert (float(MC_samples)/n_jobs).is_integer(), \
            "Adjust MC_samples ({0}), and n_jobs ({1}) so the quotient is" \
            " int. Currenty MC_samples/n_jobs = {2}." \
            .format(MC_samples, n_jobs, float(MC_samples)/n_jobs)
        [n, m] = np.shape(train)
        self.n = n
        self.m = m
        self.J = J0
        self.psi = psi
        self.train = train
        self.test = test
        self.MC_samples = MC_samples
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

    def sample_preperation(self, k):
        """
        Prepare the data before monte carlo sampling.

        Parameters
        ----------
        k : int
           Number of parallel proccees

        """
        np_arr = np.squeeze(self.batch_samples[:, :, k])
        sp_csc = sp.csc_matrix(np_arr)
        np_simp_arr = np.squeeze(self.simplex_train[:, :, k])
        simplex_train = sp.csc_matrix(np_simp_arr)
        if self.first_batch:
            # Burn-in period to get to a stable phase before sampling.
            steps = self.burn_in
        else:
            # Sampling period.
            steps = self.gibbs_iter
        samples, simplex_train = pairwise_independent_simplex_sample(
                                sp_csc, simplex_train, self.J, self.psi, steps)
        batch = np.append(samples, simplex_train, axis=1)
        return batch

    def gradient_descent(self, param0, iter, gibbs_iter):
        """
        Prepare the data before monte carlo sampling.

        Parameters
        ----------
        param0 : ndarry
           Intiial parameters
        iter : int
           Number of steps of gradient descent
        gibbs_iter : int
           Number of monte carlo iterations

        """
        param = param0
        # Get the intiial error (gradient
        err = self.error(param0, gibbs_iter)
        iteration = 0
        limit = 0.01
        import time
        start_time = time.time()
        print("---- STARTING GRADIENT ASCENT ----")

        while iteration < iter:
            param = param - self.learning_rate*err
            # Get the error (gradient)
            err = self.error(param, gibbs_iter)
            iteration = iteration + 1
            if limit < float(iteration)/iter:
                print("||Progress||: "+str(float(limit*100))+" %",
                      "||Maximal error||: "+str(round(np.max(np.abs(err)), 6)),
                      "||Time left||: %s s" %
                      str(np.around(float(time.time()
                          - start_time)*(1/(float(iteration)/iter)-1), 2)))
                limit = float(iteration)/iter+0.1

        print("||Progress||: 100 %",
              "||Maximal error||: "+str(round(np.max(np.abs(err)), 6)),
              "||Time left||: %s s" %
              str(np.around(float(time.time()
                  - start_time)*(1/(float(iteration)/iter)-1), 2)))
        return param

    def fit_pairwise(self, iter, gibbs_iter):
        """
        Prepare the data before monte carlo sampling.

        Parameters
        ----------
        iter : ndarry
           Number of iter
        iter : int
           Number of steps of gradient descent
        gibbs_iter : int
           Number of monte carlo iterations

        """
        np.random.seed(seed=1)
        J0_flat = np.append(self.J.flatten(), self.psi.flatten())
        # get the number of neurons
        high = np.shape(self.train)[0]
        # randomly samples MC_samples from the data
        randi = np.random.randint(0, high=high, size=(self.MC_samples,))
        self.samples = self.train[randi, :]
        self.simplex_samples = self.simplices[randi, :]
        self.samples = np.array(self.samples[:int(
            np.floor(np.shape(self.samples)[0]/self.n_pools)*self.n_pools), :])
        self.simplex_samples = np.array(self.simplex_samples[:int(np.floor(
           np.shape(self.simplex_samples)[0]/self.n_pools)*self.n_pools), :])
        self.batch_samples = np.zeros((self.MC_samples/self.n_pools,
                                       self.m, self.n_pools))
        self.simplex_train = np.zeros((self.MC_samples/self.n_pools,
                                       np.shape(self.psi)[0], self.n_pools))
        # Divide the data into n_pools set for parallized monte carlo
        for k in range(0, self.n_pools):
            idx_samples = range(k*(self.MC_samples/self.n_pools),
                                ((k+1)*self.MC_samples/self.n_pools))
            self.batch_samples[:, :, k] = self.samples[idx_samples, :]
            self.simplex_train[:, :, k] = self.simplex_samples[idx_samples, :]
        # Approximately needed burn-in-step
        self.burn_in = 10*gibbs_iter
        self.first_batch = True
        # Get the intial samples after burn-in
        print("---- INITIATE BURN-IN ----")
        batch = np.array(self.pool.map(self.sample_preperation,
                         range(0, self.n_pools)))
        batch = np.moveaxis(batch, 0, -1)
        self.batch_samples = batch[:, :self.m].reshape(
                             self.MC_samples/self.n_pools, self.m,
                             self.n_pools)
        self.simplex_train = batch[:, self.m:].reshape(
                             self.MC_samples/self.n_pools, self.m/3,
                             self.n_pools)

        self.first_batch = False
        # Gradient ascent to learn paramters
        J_flat = self.gradient_descent(J0_flat, iter, gibbs_iter)

        # Get model correlations matrix
        J = J_flat[:self.m**2].reshape((self.m, self.m))
        psi = J_flat[self.m**2:]
        mod_cov = self.cov_model.reshape((self.m, self.m))
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

    def error(self, J_flat, gibbs_iter):
        """
        Prepare the data before monte carlo sampling.

        Parameters
        ----------
        J_flat : ndarry
           paremters
        iter : int
           Number of steps of gradient descent
        gibbs_iter : int
           Number of monte carlo iterations

        """
        np.random.seed(seed=1)
        J = np.array(J_flat[:self.m**2])
        self.gibbs_iter = gibbs_iter
        self.J = J.reshape((self.m, self.m))
        self.psi = np.array(J_flat[self.m**2:]).reshape(self.m/3,)
        cov_models = np.zeros((self.m**2, self.n_pools))
        # NB need to chage this later on
        simplex_means = np.zeros((self.m/3, self.n_pools))
        batch = np.array(self.pool.map(self.sample_preperation,
                         range(0, self.n_pools)))
        batch = np.moveaxis(batch, 0, -1)
        self.batch_samples = batch[:, :self.m].reshape(
                            self.MC_samples/self.n_pools, self.m, self.n_pools)
        self.simplex_train = batch[:, self.m:].reshape(
                             self.MC_samples/self.n_pools, self.m/3,
                             self.n_pools)
        n, m, k = np.shape(self.batch_samples)
        # Combine all montecarlo
        for i in range(0, k):
            cov_model_temp = np.dot(self.batch_samples[:, :, i].T,
                                    self.batch_samples[:, :, i])/n
            cov_models[:, i] = cov_model_temp.flatten()
            simplex_means[:, i] = np.mean(self.simplex_train[:, :, i],
                                          axis=0).flatten()
        self.cov_model = np.sum(cov_models, axis=1)/self.n_pools
        self.simp_mean = np.sum(simplex_means, axis=1)/self.n_pools
        # Calculate the difference in mpdel correlation and emperical
        # correlation
        error_n = (-self.cov_model + self.emp_cov_train.flatten())
        error_s = (-self.simp_mean + self.rigidity.flatten())
        error_train = np.append(error_n, error_s)
        if self.save_loss:
            g_t = round(np.max(np.abs(error_train)), 6)
            self.train_error.append(g_t)
        if self.emp_cov_test is not None:
            error_test = (-self.cov_model + self.emp_cov_test.flatten())
            error_s_test = (-self.simp_mean + self.test_rigidity.flatten())
            error_test = np.append(error_test, error_s_test)
            if self.save_loss:
                g_te = round(np.max(np.abs(error_test)), 6)
                self.test_error.append(g_te)
        return error_train

    def __getstate__(self):
        """Use to make multi-processing work."""
        self_dict = self.__dict__.copy()
        del self_dict['pool']
        return self_dict
