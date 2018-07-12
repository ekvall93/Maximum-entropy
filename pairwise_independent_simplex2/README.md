# Maximum-entropy
Python 2.7 and conda 4.3.30

Maximum entropy - Maximum for simplices/neural trains


Notice - The data for the simplices are not available since it's owned by the "Blue Brain Project". The data used in Jupyter notebook file is from the experimental data. Since the connectivity is not known, the simplices are not available. Hence, arbitrary neurons will be used as source, sink, and intermediate neuron to form 2-simplex. This will result in a worse likelihood than the Pairwise Ising model. But it will give an idea of how the algorithm works.

If one have known connectivity of the neurons, the data has to be prepared to be compatible for the algorithm. The algorithm iterate through the neural data matrix in triplets since there are three neuron per simplex. For example, if you have 2 simplices, with 6 neurons with their own spike train. For example, neuron 1 and 4 has the following spike trains:



Neuron ID 1: [0 0 0 1 0 0 1 ....].T

Neuron ID 4: [1 0 0 0 0 1 0 ....].T

The neural are connectivity is also known-


Neuron ID 1 (Source) -> Neuron ID 2 (intermediate) -> Neuron ID 3 (Sink)

Neuron ID 4 (Source) -> Neuron ID 5 (intermediate) -> Neuron ID 6 (Sink)

Then the data matrix should be sorted in the following way:

Neuron ID 1: [0 0 0 1 0 0 1 ....].T

Neuron ID 2: [0 0 0 0 0 0 0 ....].T

Neuron ID 3: [0 1 0 0 0 0 0 ....].T

Neuron ID 4: [0 0 0 0 0 1 0 ....].T

Neuron ID 5: [0 0 1 0 0 1 0 ....].T

Neuron ID 6: [0 0 0 1 0 0 0 ....].T

and not like for eaxample,

Neuron ID 1: [0 0 0 1 0 0 1 ....].T

Neuron ID 3: [0 1 0 0 0 0 0 ....].T

Neuron ID 4: [0 0 0 0 0 1 0 ....].T

Neuron ID 5: [0 0 1 0 0 1 0 ....].T

Neuron ID 2: [0 0 0 0 0 0 0 ....].T

Neuron ID 6: [0 0 0 1 0 0 0 ....].T



(#datapoints, #neurons)

It's very important that the neurons is sorted in the following way, otherwise simplex train will not be extracted correctly.
