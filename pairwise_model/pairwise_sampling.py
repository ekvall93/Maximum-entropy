"""Get pairwise sampling. Written by Markus Ekvall 2018-07-09."""
import numpy as np
import scipy.sparse as sp


def sample_pairwise(samples, J, n_steps, predict_spike_rate=False):
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
    [n, m] = np.shape(samples)
    rand = np.random.uniform(low=0.0, high=1.0, size=(n, n_steps))
    for j in range(0, n_steps):
        delta_E = J[neuron_id, neuron_id].astype(float) + 2*samples.dot(
                                         J_offdiag[:, neuron_id]).astype(float)
        p_spike = 1./(1+np.exp(delta_E))
        N = np.size(p_spike)
        p_spike = np.around(p_spike.reshape((N,)).astype(float), 6)
        # If spin-state is more probeble keep it, in the end sve the most
        # probable spin-states.
        if predict_spike_rate:
            x = 0.5 < p_spike
        else:
            x = rand[:, neuron_id] < p_spike
        if sp.isspmatrix_csc(samples):
            samples = set_row_csc(samples, neuron_id, x)
        else:
            samples[:, neuron_id] = rand[:, neuron_id] < p_spike
        # Old version
        neuron_id = neuron_id + 1
        if neuron_id == m:
            neuron_id = 0
    if sp.isspmatrix_csc(samples):
        return samples.toarray()
    else:
        return samples.astype(int)


def set_row_csc(D, col_idx, new_col):
    """
    Exchange a column in a sparse matrix.

    Parameters
    ----------
    D : csc_matrix
       The samples that should be changed
    col_ids : int
       Which column that should be changed
    new_col : ndarray
       The new column that will implemented

    """
    assert sp.isspmatrix_csc(D), "Array shall be a csc_matrix"
    assert col_idx < D.shape[1], \
        "The col index ({0}) shall be smaller than the"\
        "number of col in array ({1})" \
        .format(col_idx, D.shape[1])
    try:
        N_elements_new_col = len(new_col)
    except TypeError:
        msg = "Argument new_row shall be a list or numpy array, is now a {0}"\
           .format(type(new_col))
        raise AssertionError(msg)
    N_rows = D.shape[0]
    assert N_rows == N_elements_new_col, \
        "The number of elements in new col ({0}) must be equal to " \
        "the number of col in matrix D ({1})" \
        .format(N_elements_new_col, N_rows)

    idx_start_col = D.indptr[col_idx]
    idx_end_col = D.indptr[col_idx + 1]
    additional_nnz = N_rows - (idx_end_col-idx_start_col)
    D.data = np.r_[D.data[:idx_start_col], new_col, D.data[idx_end_col:]]
    D.indices = np.r_[D.indices[:idx_start_col], np.arange(N_rows),
                      D.indices[idx_end_col:]]
    D.indptr = np.r_[D.indptr[:col_idx + 1], D.indptr[(col_idx + 1):]
                     + additional_nnz]
    return D
