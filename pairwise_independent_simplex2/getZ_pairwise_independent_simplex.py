"""Get the partition function. Written by Markus Ekvall 2018-07-09."""
from pairwise_independent_simplex_sampling import \
   pairwise_independent_simplex_sample
from get_simplex_train import get_simplex_train
import numpy as np
import scipy.sparse as sp


def getZ_pairwise_independent_simplex(M_samples, simplex, J, psi, betas):
    """
    Estimate the normalzation constant.

    ----------
    M_samples : int
        Number of samples used in Monte carlo simulations
    J : ndarray
        The dervied paramters
    betas : list
       list with vlaues between [0,1].

    """
    np.random.seed(seed=1)
    n = np.shape(J)[0]
    J_diag = J.copy()
    J_diag = np.diag(np.diag(J_diag))
    limit = 0.01
    import time
    start_time = time.time()
    print("---- INITIATE ANNEALED SAMPLING ----")
    samples = np.random.uniform(low=0.0, high=1.0, size=(M_samples, n))
    simplex_train = np.random.uniform(low=0.0, high=1.0, size=(M_samples, n/3))
    spike_probs = np.exp(-np.diag(J).T)/(1 + np.exp(-np.diag(J).T))
    samples = samples < np.tile(spike_probs, [M_samples, 1])
    log_prob_ratios = energy_diff(samples, simplex, betas[1], betas[0],
                                  J_diag, J, psi)
    for k in range(0, np.size(betas) - 2):
        J_k = (1-betas[k+1])*J_diag + betas[k+1]*J
        psi_k = psi*betas[k+1]
        simplex_train = sp.csc_matrix(simplex_train)
        samples = sp.csc_matrix(samples)
        samples, simplex_train = pairwise_independent_simplex_sample(
                                 samples, simplex_train, J_k, psi_k, n)
        log_prob_ratios = log_prob_ratios + energy_diff(
                    samples, simplex, betas[k+2], betas[k+1], J_diag, J, psi)
        if limit < float(k)/(np.size(betas) - 2):
            print("||Progress||: "+str(float(limit*100))+" %",
                  "||Time left||: %s s" % str(np.around(
                      float(time.time() - start_time)*(1/(float(k)/(
                         np.size(betas) - 2))-1), 2)))
            limit = float(k)/(np.size(betas) - 2)+0.1
    Z_0 = np.exp(np.sum(np.log(1 + np.exp(-np.diag(J)))))
    Z = Z_0 * np.mean(np.exp(log_prob_ratios))
    print("||Progress||: "+str(float(limit*100))+" %",
          "||Time left||: %s s" % str(np.around(
              float(time.time() - start_time)*(1/(float(k)/(
                 np.size(betas) - 2)) - 1), 2)))
    return Z


def energy_diff(samples, simplex, b_k, b_k_1, J_diag, J, psi):
    """Calcualte the energydifference."""
    simplex = get_simplex_train(samples)
    psi_term = -np.dot(simplex, psi)*(b_k - b_k_1)
    return np.sum(samples*samples.dot((b_k-b_k_1)*(J_diag-J)), 1)+psi_term
