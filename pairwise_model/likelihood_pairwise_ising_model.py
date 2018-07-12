"""Get likelihood. Written by Markus Ekvall 2018-07-09."""
from getZ_pairwise_ising_model import getZ_pairwise_ising_model
import numpy as np


def likelihood_pairwise_ising_model(data, MC_samples, J, betas):
    """
    Calculate the negative likelihood.

    ----------
    data : ndarray
        Spiketrains/simplextrains (#time_points,#neurons/simplex)
    MC_samples : int
        Number of samples used in Monte carlo simulations
    J : ndarray
        The dervied paramters
    betas : list
       list with vlaues between [0,1].

    """
    Z = getZ_pairwise_ising_model(MC_samples, J, betas)
    neg_log_L = (np.log(Z)
                 + np.mean(np.sum(data*(data.dot(J)), 1))) / np.shape(J)[0]
    return neg_log_L
