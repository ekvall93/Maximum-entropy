from pairwise_sampling import sample_pairwise
import numpy as np
from numpy import genfromtxt


def getZ_pairwise(M_samples, J, betas):
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
    #samples = np.array(genfromtxt('one_samp.csv', delimiter=','))
    samples =np.random.uniform(low=0.0, high=1.0, size=(M_samples,n))
    spike_probs = np.exp(-np.diag(J).T)/(1 + np.exp(-np.diag(J).T))
    samples = samples < np.tile(spike_probs, [M_samples, 1])
    log_prob_ratios = energy_diff(samples, betas[1], betas[0],J_diag,J)
    for k in range(0,np.size(betas) - 2):
        J_k = (1-betas[k+1])*J_diag + betas[k+1]*J
        samples = sample_pairwise(samples, J_k, n)
        log_prob_ratios = log_prob_ratios + energy_diff(samples, betas[k+2], betas[k+1],J_diag,J)
    Z_0 = np.exp(np.sum(np.log(1 + np.exp(-np.diag(J)))))
    Z = Z_0 * np.mean(np.exp(log_prob_ratios))
    return Z


def energy_diff(samples, b_k, b_k_1,J_diag,J):
    """ Calcualte the energydifference"""
    return np.sum(samples*samples.dot((b_k-b_k_1)*(J_diag-J)),1)
