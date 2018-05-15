from load_data import *
#from fit_pairwise import fit_pairwise
from fit_pairwise import *
from data_preperation import *
from likelihood_pairwise import likelihood_pairwise
#Experimental data
#data = get_mat_data()
#data = data[:40,:4]

#BPP data
train_s,train_n, test_s,test_n  = load_bpp_data(nr_seed=30,factor_test=0.1,split=True)
#Make it compatible with the code
train = train_s.T
test = test_s.T
#You have use the data you are going to use fpr the shape.
[m,n] = np.shape(train)
J0 = np.zeros((n,n))


learning_rate = 1
#NB Chosse M-samples such that it can be divded equally beteen all processes.
M_samples = 40000
# Number of cores for multiprocess.
n_jobs = 2
entropy = fit_pairwise(train,J0,learning_rate,M_samples,n_jobs,test=test,save_loss=True)
iter = 10;
#Recommended to have twice as many as neuronss.
gibbs_steps = 2*n

verbose = str(gibbs_steps)+"_"+str(iter)+"_"+str(M_samples)+"_"+str(learning_rate)

start_time = time.time()
J, emp_cov ,mod_cov,samples,train_error, test_error = entropy.fit_pairwise(iter,gibbs_steps);
print("--- %s seconds ---" % (time.time() - start_time))

train_error.pop(0)
test_error.pop(0)
iteration = range(0,np.size(train_error))

#Use it if the likelihoods are necessary.
#betas = np.linspace(0,1,num=10)
#neg_log_L = likelihood_pairwise(data, M_samples, J, betas)
#print(neg_log_L)

learning_curve(iteration, train_error,test_error=test_error)
correlation_graph_data(emp_cov,verbose)
correlation_graph_data(mod_cov,verbose,model=False)
comparison(emp_cov.flatten(), mod_cov.flatten(),verbose,correlation=True)
ins = np.tril_indices(n,k=-1)
under_emp = emp_cov[ins]
under_mod = mod_cov[ins]
under_emp = np.tril(emp_cov, -1)
under_mod = np.tril(mod_cov, -1)
eps = under_emp.flatten() - under_mod.flatten()
CHI2(eps,verbose)
correlation_graph_J(J,verbose)
