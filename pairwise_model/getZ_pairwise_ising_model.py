"""Get the partition function. Written by Markus Ekvall 2018-07-09."""
from pairwise_ising_model_sampling import pairwise_ising_model_sampling
import numpy as np


def getZ_pairwise_ising_model(MC_samples, J, betas):
    """
    Estimate the normalzation constant.

    ----------
    MC_samples : int
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
    samples = np.random.uniform(low=0.0, high=1.0, size=(MC_samples, n))
    spike_probs = np.exp(-np.diag(J).T)/(1 + np.exp(-np.diag(J).T))
    samples = samples < np.tile(spike_probs, [MC_samples, 1])
    log_prob_ratios = delta_E(samples, betas[1], betas[0], J_diag, J)
    limit = 0.01
    import time
    start_time = time.time()
    print("---- INITIATE ANNEALED SAMPLING ----")
    for k in range(0, np.size(betas) - 2):
        J_k = (1-betas[k+1])*J_diag + betas[k+1]*J
        samples = pairwise_ising_model_sampling(samples, J_k, n)
        log_prob_ratios = log_prob_ratios + delta_E(samples, betas[k+2],
                                                    betas[k+1], J_diag, J)
        if limit < float(k)/(np.size(betas) - 2):
            print("||Progress||: "+str(float(limit*100))+" %",
                  "||Time left||: %s s" %
                  str(np.around(float(time.time()
                                - start_time)*(1/(float(k)/(np.size(betas)-2))
                                - 1), 2)))
            limit = float(k)/(np.size(betas) - 2)+0.1

    Z_0 = np.exp(np.sum(np.log(1 + np.exp(-np.diag(J)))))
    Z = Z_0 * np.mean(np.exp(log_prob_ratios))
    print("||Progress||: "+str(float(limit*100))+" %",
          "||Time left||: %s s" %
          str(np.around(float(time.time()
              - start_time)*(1/(float(k)/(np.size(betas) - 2))-1), 2)))
    return Z


def delta_E(samples, b_k, b_k_1, J_diag, J):
    """Calcualte the energydifference."""
    return np.sum(samples*samples.dot((b_k-b_k_1)*(J_diag-J)), 1)
