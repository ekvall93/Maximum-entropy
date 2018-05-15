from load_data import *
from fit_pairwise import *
from data_preperation import *
from likelihood_pairwise import likelihood_pairwise
from sklearn import preprocessing
import numpy
from numpy import random


#BPP

learning_rate = 1
#NB Chosse M-samples such that it can be divded equally beteen all processes.
M_samples = 400
# Number of cores for multiprocess.
n_jobs = 2;
iter = 2
betas = np.linspace(0,1,num=10)

likelihood_s_n = []
likelihood_n = []
#NB adding extra Parameters and simplex data.
for s in range(0,2):
    nr_simp = 10
    seed = s
    data_s_and_n, data_n = data_for_likelihood(nr_simp,seed)
    data_s_and_n = data_s_and_n.T
    data_n = data_n.T
    [m,n] = np.shape(data_s_and_n)
    print(m,n)
    #Recommended to have twice as many as neurons.
    gibbs_steps = n*2
    J0 = np.zeros((n,n))
    #Sim_and_n
    entropy = fit_pairwise(data_s_and_n,J0,learning_rate,M_samples,n_jobs,test=None,save_loss=True)
    J, emp_cov ,mod_cov,samples,train_error = entropy.fit_pairwise(iter,gibbs_steps);
    neg_log_L = likelihood_pairwise(data_s_and_n, M_samples, J, betas)
    print("Simplex and it's neurons",np.exp(-neg_log_L))
    likelihood_s_n.append(np.exp(-neg_log_L))



    #Only n
    entropy = fit_pairwise(data_n,J0,learning_rate,M_samples,n_jobs,test=None,save_loss=True)
    J, emp_cov ,mod_cov,samples,train_error = entropy.fit_pairwise(iter,gibbs_steps);
    neg_log_L = likelihood_pairwise(data_n, M_samples, J, betas)
    print("Same neurons+extra neurons",np.exp(-neg_log_L))
    likelihood_n.append(np.exp(-neg_log_L))
verbose = str(gibbs_steps)+"_"+str(iter)+"_"+str(M_samples)+"_"+str(learning_rate)

likelihhod_box_plot(likelihood_s_n,likelihood_n,verbose)
