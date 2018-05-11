#from markus_montecarlo import *
#from monte_carlo import *
import numpy  as np
import scipy.io as sio
import matplotlib.pyplot as pp
import os

def get_mat_data():
    mat = sio.loadmat('/Users/markusekvall/Desktop/final_entropy_model/one_zero.mat')
    data = mat['sample']
    #mat = sio.loadmat('/Users/markusekvall/Desktop/final_entropy_model/dataset/simplex_train_3D.mat')
    #data = mat['x']
    return data


def get_spike_data():
    neuron_spikes = []

    first = 0
    last = 0
    for i, file in enumerate(os.listdir('/Users/markusekvall/Desktop/final_entropy_model/Spike_trains')):
        if file.endswith(".mat"):
            #print(file)
            if file[0] == ".":
                continue
            else:
                x = sio.loadmat(os.path.join("./Spike_trains", file))
                cluster_class = x["cluster_class"]
                #Set to ==2 if you want the second cluster
                idx = [cluster_class[:, 0] == 1]
                spike = cluster_class[idx[0], 1]
                if np.size(spike)!=0:
                    if max(spike) > last:
                        last = max(spike)
                    if min(spike) > last:
                        first = min(spike)
                    #Some of cluster 2 is zero, need to remove them
        neuron_spikes.append(spike)
    return neuron_spikes, first, last





def bin_the_data(neuron_spikes, first, last, bin_size):
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
    np.array(neuron_activity)
    spin = []

    for i, neuron in enumerate(neuron_activity):
        list2 = [0 if i == 0 else 1 for i in neuron]
        spin.append(list2)

    return spin

def plot_activity(neuron_spin, timebins):
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
    neuron_spikes, first, last = get_spike_data()
    neuron_activity, timebins = bin_the_data(neuron_spikes, first, last, bin_size)
    neuron_spin = set_neuron_spin(neuron_activity)
    if plot:
        plot_activity(neuron_spin, timebins)

    return np.asarray(neuron_spin)
