"""Get likelihood. Written by Markus Ekvall 2018-07-09."""
from getZ_pairwise_independent_simplex import getZ_pairwise_independent_simplex
from get_simplex_train import get_simplex_train
import numpy as np


def likelihood_pairwise_independent_simplex(data, M_samples, J, psi, betas):
    """
    Calculate the negative likelihood.

    ----------
    data : ndarray
        Spiketrains/simplextrains (#time_points,#neurons/simplex)
    M_samples : int
        Number of samples used in Monte carlo simulations
    J : ndarray
        The dervied paramters
    betas : list
       list with vlaues between [0,1].

    """
    simplex = get_simplex_train(data)
    Z = getZ_pairwise_independent_simplex(M_samples, simplex, J, psi, betas)
    psi_term = np.dot(simplex, psi)
    neg_log_L = (np.log(Z) + np.mean(np.sum(data*(data.dot(J)),
                 1)+psi_term)) / np.shape(J)[0]
    return neg_log_L
