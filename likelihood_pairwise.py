from getZ_pairwise import getZ_pairwise
import numpy as np
def likelihood_pairwise(data, M_samples, J, betas):
    """
    Calculate the negative likelihood
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
    Z = getZ_pairwise(M_samples, J, betas)
    neg_log_L = (np.log(Z) + np.mean(np.sum(data*(data.dot(J)), 1))) / np.shape(J)[0]
    return neg_log_L
