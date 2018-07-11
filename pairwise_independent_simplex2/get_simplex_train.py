"""Get simplex trains. Written by Markus Ekvall 2018-07-09."""
import numpy as np
import scipy.sparse as sp


def get_simplex_train_gibbs(neuron, simplex, samples, simplex_train):
    """
    Simplex train used for the Gibbs sampling.

    ----------
    neuron : int
       Which neurons that will fire
    simplex : int
       Which simplex the neurons belongs to
    samples : Array
       The spike-trians
    simplex_train : Array
       The simplex_train

    """
    index1 = (neuron+1) % 3
    index2 = (neuron+2) % 3
    [n, m] = np.shape(samples)

    simplex_start_index = np.zeros(n)
    # Source neuron is 100% active
    # index1 intermediate neurons and index2 sink
    if neuron % 3 == 0:
        # Spiketimes in bin after source
        # Intermediate spike
        # NB Markus
        first_spike_idx1 = samples[:, index1+simplex*3] == 1
        # Check in the next time-bin
        # Check in the second next time-bin, sink can be 7,5 ms after
        # sink.
        sink_1 = first_spike_idx1.nonzero()[0]
        sink_2 = sink_1+1
        sink_1 = sink_1[sink_1 < n]
        sink_2 = sink_2[sink_2 < n]
        sink_idx1 = samples[sink_1, index2+simplex*3] == 1
        sink_idx2 = samples[sink_2, index2+simplex*3] == 1
        # Get source neuron idx
        iX1 = sink_1[sink_idx1.nonzero()[0]]-1
        iX2 = sink_2[sink_idx2.nonzero()[0]]-2
    # Intermediate neurons will fire 100%.
    # index2 source neurons and index1 is the sink
    elif neuron % 3 == 1:
        # Source neurons
        # NB Markus
        first_spike_idx1 = samples[:, index2+simplex*3] == 1
        # Check sink spike-time
        sink_1 = first_spike_idx1.nonzero()[0]+1
        sink_2 = sink_1+1
        sink_1 = sink_1[sink_1 < n]
        sink_2 = sink_2[sink_2 < n]
        sink_idx1 = samples[sink_1, index1+simplex*3] == 1
        sink_idx2 = samples[sink_2, index1+simplex*3] == 1
        # Get source neuron idx
        iX1 = sink_1[sink_idx1.nonzero()[0]]-1
        iX2 = sink_2[sink_idx2.nonzero()[0]]-2
    # Sink neurons will fire 100%.
    # index1 source neurons and index2 is intermedier
    elif neuron % 3 == 2:
        # Source fire
        # NB Markus
        first_spike_idx1 = samples[:, index1+simplex*3] == 1
        intermediate = first_spike_idx1.nonzero()[0] + 1
        intermediate = intermediate[intermediate < n]
        intermediate_idx1 = samples[intermediate, index2+simplex*3] == 1
        # Get source neuron idx
        iX1 = list(intermediate[intermediate_idx1.nonzero()[0]]-1)
        iX2 = []
    # Get the correct timing for placement of simplex
    simplex_start_id = np.unique(list(iX1)+list(iX2))
    if np.size(simplex_start_id):
        simplex_start_index[simplex_start_id] = 1
    simplex_train = set_row_csc(simplex_train, simplex, simplex_start_index)
    return simplex_train


def get_simplex_train(samples):
    """
    The simplex train for 2 dimensional.

    ----------
    samples : array
       The spike trains

    """
    samples = sp.csc_matrix(samples)
    [n, m] = np.shape(samples)
    simplex_train = np.zeros((n, m/3))
    for simplex in range(0, m/3):
        # NB Markus
        first_spike_idx1 = samples[:, simplex*3] == 1
        intermediate = first_spike_idx1.nonzero()[0] + 1
        intermediate = intermediate[intermediate < n]
        intermediate_spike = samples[intermediate, 1+simplex*3] == 1
        sink_1 = intermediate[intermediate_spike.nonzero()[0]]
        sink_2 = intermediate[intermediate_spike.nonzero()[0]]+1
        sink_1 = sink_1[sink_1 < n]
        sink_2 = sink_2[sink_2 < n]
        sink_1_spike = samples[sink_1, 2+simplex*3] == 1
        sink_2_spike = samples[sink_2, 2+simplex*3] == 1
        iX1 = list(sink_1[sink_1_spike.nonzero()[0]]-1)
        iX2 = list(sink_2[sink_2_spike.nonzero()[0]]-2)
        iX3 = []
        simplex_start_id = np.unique(list(iX1)+list(iX2)+list(iX3))
        if np.size(simplex_start_id):
            simplex_train[simplex_start_id, simplex] = 1
    return simplex_train


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
        "The col index ({0}) shall be smaller than the number"\
        "of col in array ({1})" \
        .format(col_idx, D.shape[1])
    try:
        N_elements_new_col = len(new_col)
    except TypeError:
        msg = 'Argument new_row shall be a list or numpy array, is now a {0}'\
              .format(type(new_col))
        raise AssertionError(msg)
    N_rows = D.shape[0]
    assert N_rows == N_elements_new_col, \
        "The number of elements in new col ({0}) must be equal to " \
        "the number of col in matrix D ({1})" \
        .format(N_elements_new_col, N_rows)
    idx_start_col = D.indptr[col_idx]
    idx_end_col = D.indptr[col_idx + 1]
    additional_nnz = N_rows - (idx_end_col - idx_start_col)
    D.data = np.r_[D.data[:idx_start_col], new_col, D.data[idx_end_col:]]
    D.indices = np.r_[D.indices[:idx_start_col],
                      np.arange(N_rows), D.indices[idx_end_col:]]
    D.indptr = np.r_[D.indptr[:col_idx + 1],
                     D.indptr[(col_idx + 1):] + additional_nnz]
    return D
