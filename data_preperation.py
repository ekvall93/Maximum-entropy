import numpy as np
import matplotlib as mpl
mpl.use('Agg')
#from coniii import *
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def orderings_samples(sample_MCH, prob_sample):
    prob_MCH = state_probs(sample_MCH,allstates=None,weights=None,normalized=True)
    indices_str = [str(i) for i in range(0,np.size(prob_MCH[0]))]
    unordered_prob, unordered_indices_str = (list(t) for t in zip(*sorted(zip(prob_sample[0], indices_str))))
    ordered_prob = list(reversed(unordered_prob))
    ordered_indices_str = list(reversed(unordered_indices_str))
    ordered_indices = [int(i) for i in ordered_indices_str]
    ordererd_states = [prob_sample[1][i] for i in ordered_indices]
    prob_sample = [ordered_prob, ordererd_states]
    reordered_prob_MCH = []
    match = False
    for state_sample in prob_sample[1]:
        for i, state_MCH in enumerate(prob_MCH[1]):
            if list(state_sample) == list(state_MCH):
                reordered_prob_MCH.append(prob_MCH[0][i])
                match =True
                break
        if not match:
            reordered_prob_MCH.append(0)
        match = False

    return reordered_prob_MCH, prob_sample


def epsilon(sisj,sisj_estimated, corr_std):
    eps = []
    sum = 0
    for s_e, s_m, sd in zip(sisj,sisj_estimated, corr_std):
        eps.append((s_e-s_m)/sd)
    return eps



def split_data(data, ratio, seed=1):
    """split the dataset based on the split ratio."""
    # set seed
    np.random.seed(seed)
    # generate random indices
    num_row = np.shape(data)[1]
    indices = np.random.permutation(num_row)
    index_split = int(np.floor(ratio * num_row))
    index_train = indices[: index_split]
    index_test = indices[index_split:]
    # create split
    train = []
    test = []
    #Convert list into array
    data = np.asarray(data)
    train = data[:, index_train]
    test = data[:, index_test]
    #np.random.shuffle(train)
    return train, test

def correlation_matrix(test_samples):
    test = np.dot(test_samples, test_samples.T)/np.shape(test_samples)[1]
    #np.fill_diagonal(test, 0, wrap=False)
    mean = np.mean(test_samples, axis=1)
    C = []
    for i, element in enumerate(test):
        c = []
        for k, ele in enumerate(element):
            if i == k:
                c.append(0)
            else:
                c.append(ele - mean[i]*mean[k])
        C.append(c)
    return C

def correlation_grad(test_samples):
    test = np.dot(test_samples, test_samples.T)/np.shape(test_samples)[1]
    #np.fill_diagonal(test, 0, wrap=False)
    mean = np.mean(test_samples, axis=1)
    C = []
    for i, element in enumerate(test):
        c = []
        for k, ele in enumerate(element):
            if i == k:
                c.append(0)
            else:
                c.append(ele)
        C.append(c)
    return C


def error(h, J, test_samples):
    error_h = np.mean(abs(h-np.mean(test_samples, axis=1)))
    K = correlation_matrix(test_samples)
    C = np.cov(test_samples)
    #np.fill_diagonal(C, 0, wrap=False)

    error_cor = C - J
    np.asarray(error_cor)

    error_J = np.mean(abs(error_cor))/np.mean(abs(C))

    return error_h, error_J


def save_data_and_print(h, J, test, train, train_error, test_error,
                        total_test_error, total_train_error):
    error_h_test, error_J_test = error(h, J, test)
    error_h_train, error_J_train = error(h, J, train)
    print("training", error_h_train, error_J_train)
    print("Testing", error_h_test, error_J_test)
    print("Total train: ", error_h_train + error_J_train)
    print("Total test: ", error_h_test + error_J_test)
    train_error.append(error_J_train)
    test_error.append(error_J_test)
    total_test_error.append(error_h_test + error_J_test)
    total_train_error.append(error_h_train + error_J_train)

    return train_error, test_error, total_test_error, total_train_error

def learning_curve(iteration, train_error, test_error=None, total=False):
    #I = np.arange(1, np.size(iteration)+1)
    I = iteration
    fig = plt.figure()
    ax = fig.add_subplot(111)
    if test_error is not None:
        line2, = plt.plot(I, test_error, 'b--', label="Test error")
    else:
        line1, = plt.plot(I, train_error, 'r--', label="Training error")

    plt.ylabel("Average error:"r'$C_{ij}-J_{ij}$')
    plt.xlabel("Iteration")
    first_legend = plt.legend(handles=[line1], loc=1)
    ax = plt.gca().add_artist(first_legend)
    if test_error is not None:
        plt.legend(handles=[line2], loc=4)
    plt.title('Learning curve')
    if total:
        plt.savefig('figures/learning_curve/_total.png')
    else:
        plt.savefig('figures/learning_curve/learning_curve.png')


def correlation_graph_data(data,verbose,corr_matrix = False, model=True):


    fig = plt.figure()
    ax = fig.add_subplot(111)
    if corr_matrix:
        C= data
    else:
        C = np.cov(data)
    np.fill_diagonal(C, 0, wrap=False)
    tax = ax.matshow(C, cmap='hot', interpolation='nearest')
    fig.colorbar(tax)
    plt.ylabel("i")
    plt.xlabel("j")
    if model:
        plt.title('Correlation matrix for model: 'r'$C_{ij}$')
        plt.savefig('figures/correlation/C_m_'+str(verbose)+'.png')
    else:
        plt.title('Correlation matrix for emperical data: 'r'$C_{ij}$')
        plt.savefig('figures/correlation/C_e_'+str(verbose)+'.png')

def correlation_graph_J(J,verbose):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    #np.fill_diagonal(J, 0, wrap=False)
    cax = ax.matshow(J, cmap='hot', interpolation='nearest')
    fig.colorbar(cax)
    plt.ylabel("i")
    plt.xlabel("j")
    plt.title('Correlation matrix optimal: 'r'$J_{ij}$')
    plt.savefig('figures/correlation/J_'+str(verbose)+'.png')


def firing_rate_plot(firing_rate,verbose, model = False):
    neurons = np.arange(1, np.size(firing_rate)+1)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.xlabel("Neuron i")
    plt.plot(neurons, firing_rate, 'bo', label="Test error")
    if model:
        plt.title("Firing rate curve :"r'$h$')
        plt.ylabel("Firing rate:"r'$h_{i}$')
        plt.savefig('figures/activity.png')
    else:
        plt.title("Firing rate curve :"r'$r$')
        plt.ylabel("Firing rate:"r'$r_{i}$')
        plt.savefig('figures/firing_rate_data.png')

def firing_rate_plot_duo(h, emp_firing_rate, model_firing_rate,verbose):
    neurons = np.arange(1, np.size(emp_firing_rate)+1)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.xlabel("Neuron i")
    h, = plt.plot(neurons, h, 'ro',label="Activity"r'($h_{i}$)')
    r_e, = plt.plot(neurons, emp_firing_rate, 'bo',label="Mean emperical firing rate"r'($r_{i}$)')
    r_m, = plt.plot(neurons, model_firing_rate, 'go',label="Mean model firing rate"r'($r_{i}$)')
    plt.legend(handles=[h, r_e, r_m])
    plt.title("Activity plot")
    plt.ylabel("Firing rate spike/milliseconds")
    plt.savefig('figures/firing/fire_'+str(verbose)+'.png')

def distribution(model_prob,emperical_prob,verbose, ratio=100):
    if np.size(model_prob) < ratio:
        points = np.size(model_prob)
    else:
        points = ratio
    #points = int(np.size(model_prob)*0.2)
    histogram=plt.figure()
    a= [range(1,points+1)]
    plt.hist(a,np.size(a), weights=model_prob[:points],alpha=0.5, label='Model')
    plt.hist(a,np.size(a), weights=emperical_prob[0][:points],alpha=0.5, label='Emperical')
    plt.legend(loc='upper right')
    plt.ylabel("Probability")
    plt.xlabel("Spin-states")
    plt.savefig('figures/distribution/distribution_'+str(verbose)+'.png')

def comparison(observed, predicted,verbose,correlation=False):
    plt.figure()
    plt.plot(observed,predicted,'o')
    minimum = min(min(observed),min(predicted))
    maximum = max(max(observed),max(predicted))
    if correlation:
        plt.xlabel("Observed correlations")
        plt.ylabel("Model correlations")
    else:
        plt.xlabel("Observed means")
        plt.ylabel("Model means")

    plt.plot([minimum,maximum],[minimum,maximum],'k-')
    if correlation:
        plt.savefig('figures/comparison/com_cor_'+str(verbose)+'.png')
    else:
        plt.savefig('figures/comparison/com_mean'+str(verbose)+'.png')

def CHI2(eps,verbose):
    from scipy.stats import norm
    mu, std = norm.fit(eps)

    histogram=plt.figure()
    plt.hist(eps,normed=True)
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    plt.plot(x, p, 'k', linewidth=2)
    plt.legend(loc='upper right')
    plt.ylabel("probability")
    plt.xlabel("Eps")
    plt.savefig('figures/CHI2/eps_'+str(verbose)+'.png')

def energy_dist(dE_mod,dE_sample,verbose):
    histogram=plt.figure()
    n1 = plt.hist(dE_mod, alpha=0.5, label='Model',normed=True)
    n2 = plt.hist(dE_sample, alpha=0.5, label='Emperical',normed=True)
    plt.legend(loc='upper right')
    plt.ylabel("Probability")
    plt.xlabel("Energy")
    plt.savefig('/gpfs/bbp.cscs.ch/project/proj9/markus_test/maxent_2/figures/enerygy_dist/ed_'+str(verbose)+'.png')
