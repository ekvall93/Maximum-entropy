"""Get partition function. Written by Markus Ekvall 2018-07-09."""
import numpy as np
import scipy.io as sio
import os
import matplotlib.pyplot as plt


def get_mat_data(experimental=True, stimulated=True, dimension=None):
    """
    Get the data.

    Parameters
    ----------
    experimental : bool
       If the data should be experiemental or not
    stimulated : bool
       Stimulated or not
    dimension : int
       Choose between which dimension of the simplex data (2-4) to use

    """
    if experimental:
        if stimulated:
            mat = sio.loadmat('../dataset/Arkiv/stimulated.mat')
            data = mat['x'].T
        else:
            mat = sio.loadmat('../dataset/Arkiv/unstimulated.mat')
            data = mat['sample']
    else:
        if dimension is None:
            assert "Need to choose dimension"

        mat = sio.loadmat('../dataset/Arkiv/simplex_train_'+str(dimension)
                          + 'D_sorted.mat')
        data = mat['x']
    return data


def get_spike_data():
    """Extract spike trians from experimental data."""
    neuron_spikes = []
    first = 0
    last = 0
    for i, file in enumerate(os.listdir("""/Users/markusekvall/Desktop/
                                        final_entropy_model/Spike_trains""")):
        if file.endswith(".mat"):
            if file[0] == ".":
                continue
            else:
                x = sio.loadmat(os.path.join("./Spike_trains", file))
                cluster_class = x["cluster_class"]
                # Set to ==2 if you want the second cluster
                idx = [cluster_class[:, 0] == 1]
                spike = cluster_class[idx[0], 1]
                if np.size(spike) != 0:
                    if max(spike) > last:
                        last = max(spike)
                    if min(spike) > last:
                        first = min(spike)
        neuron_spikes.append(spike)
    return neuron_spikes, first, last


def bin_the_data(neuron_spikes, first, last, bin_size):
    """
    Decide which bin size of the data.

    Parameters
    ----------
    experimental : bool
       If the data should be experiemental or not
    stimulated : bool
       Stimulated or not
    dimension : int
       Choose between which dimension of the simplex data (2-4) to use

    """
    neuron_activity = []
    timebins = range(first, int(last) + int(last) % bin_size, bin_size)
    for spike in neuron_spikes:
        activity = []
        spike_time = spike[0]
        i = 0
        for bin_size in timebins:
            k = 0
            while spike_time < bin_size:
                i += 1
                if i >= np.size(spike):
                    break
                spike_time = spike[i]
                k += 1
            activity.append(k)
        neuron_activity.append(activity)
    return neuron_activity, timebins


def set_neuron_spin(neuron_activity):
    """
    Set the spin of the neuron, 0 == inactive, 1 == active.

    Parameters
    ----------
    neuron_activity : list
       Set the state of the neuron in each bin

    """
    np.array(neuron_activity)
    spin = []
    for i, neuron in enumerate(neuron_activity):
        list2 = [0 if i == 0 else 1 for i in neuron]
        spin.append(list2)
    return spin


def plot_activity(neuron_spin, timebins):
    """
    Get raster plot.

    Parameters
    ----------
    neuron_spin : list
       spike train
    neuron_spin : int
       Size of bins

    """
    n, k = np.shape(neuron_spin)
    for i, neuron in enumerate(neuron_spin):
        Y = []
        X = []
        size = np.int(np.size(timebins)*0.001)
        for k, time in enumerate(timebins[:size]):
            if neuron[k] == 1:
                Y.append(i+1)
                X.append(time)

    plt.plot(X, Y, 'bo', markersize=1)
    plt.ylabel("Neurons")
    plt.xlabel("Time (ms)")
    plt.title('Raster plot of neural activity')
    plt.legend(loc='best')
    plt.show()


def get_data(bin_size, plot=False):
    """
    Extract soike trains from data.

    Parameters
    ----------
    bin_size : int
       Size of time bins
    plot : bool
       Get raster-plot

    """
    neuron_spikes, first, last = get_spike_data()
    neuron_activity, timebins = bin_the_data(neuron_spikes, first, last,
                                             bin_size)
    neuron_spin = set_neuron_spin(neuron_activity)
    if plot:
        plot_activity(neuron_spin, timebins)
    return np.asarray(neuron_spin)
