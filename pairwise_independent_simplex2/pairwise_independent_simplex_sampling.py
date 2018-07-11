"""Get pairwise sampling. Written by Markus Ekvall 2018-07-09."""
import numpy as np
import scipy.sparse as sp
from get_simplex_train import get_simplex_train_gibbs, set_row_csc


def pairwise_independent_simplex_sample(samples, simplex_train, J, psi,
                                        n_steps):
    """
    Monte carlo sampling for the pairwise model.

    Parameters
    ----------
    np_arr : ndarray
       Contains the mch-samples in dense format. For change values
       in data (faster than spare matrix)
    samples : ndarray
       Contains the mch-samples in sparse format. For dot poduct of data (
       fast than dense matrix).
    n_steps : int
       Number of MC iterations

    """
    np.random.seed(seed=1)
    J_offdiag = J.copy()
    np.fill_diagonal(J_offdiag, 0)
    neuron_id = 0
    s = 0
    [n, m] = np.shape(samples)
    rand = np.random.uniform(low=0.0, high=1.0, size=(n, n_steps))
    for j in range(0, n_steps):

        simplex_train = get_simplex_train_gibbs(neuron_id, s, samples,
                                                simplex_train)
        delta_E = J[neuron_id, neuron_id].astype(float)
        + 2*samples.dot(J_offdiag[:, neuron_id]).astype(float)
        + simplex_train[:, s].dot(psi[s].reshape(1,)).astype(float)

        p_spike = 1./(1+np.exp(delta_E))
        N = np.size(p_spike)
        p_spike = p_spike.reshape((N,)).astype(float)
        # If spin-state is more probeble keep it, in the end sve the most
        # probable spin-states.
        r = rand[:, neuron_id]
        x = r < p_spike
        if sp.isspmatrix_csc(samples):
            samples = set_row_csc(samples, neuron_id, x)
        else:
            samples[:, neuron_id] = rand[:, neuron_id] < p_spike

        # Old version
        neuron_id = neuron_id + 1
        if neuron_id % 3 == 0:
            s += 1
        if neuron_id == m:
            s = 0
            neuron_id = 0
    if sp.isspmatrix_csc(samples):
        return samples.toarray(), simplex_train.toarray()
    else:
        return samples.astype(int), simplex_train.toarray()
