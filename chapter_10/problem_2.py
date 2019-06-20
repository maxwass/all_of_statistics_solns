from __future__ import print_function
import os
import time
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
#sys.path.append(os.path.dirname(os.path.abspath('../confidence_interval.py')))
#from confidence_interval import normal_95_confid_interval,pivotal_confid_interval,percentile_confid_interval


#Sampling from uniform distribution:
# X_1,...,X_10 ~ U[a,b] with a=1, b=3, n=10

#Estimating Tau = integral x dF(x) = mean of distribution, with
# MLE:                          tau_mle = (b_mle-a_mle)/2
# NonParam Plug In Estimator:   tau_np  = sum Xi / n

# Estimate MSE of tau_mle by simulation
# Find MSE of tau_np analytically
# Compare

#take many true samples of size sample_size to approximate the density
# of theta_n = max(Xi), i=1->sample_size
def check_density(sample_size,K, true_a, true_b):
    mean_mle_n  = np.zeros(K) #store estimates from each n-sample
    mean_pli_n  = np.zeros(K)
    for k in range(K):
        X = np.random.uniform(true_a,true_b,sample_size)
        mean_mle_n[k] = (np.amax(X) - np.amin(X))/2 + np.amin(X)
        mean_pli_n[k] = np.mean(X)

    print("Sampling Distribution of the Mean. {0:d} trials of {1:d} samples, Xi~U[a,b]".format(K,sample_size))
    print("Mean of Sampling Distribution:")
    print("MLE:     {0:f}".format(np.mean(mean_mle_n)))
    print("Plug in: {0:f}".format(np.mean(mean_pli_n)))
    print("Standard Error: Stdv of sampling distribution")
    print("MLE:     {0:f}".format(np.std(mean_mle_n)))
    print("Plug in: {0:f}".format(np.std(mean_pli_n)))
    bins = np.linspace(true_a-0.5, true_b+0.5,1000)
    plt.hist(mean_mle_n, bins, alpha=0.5,density=False, label='MLE')
    plt.hist(mean_pli_n, bins, alpha=0.5,density=False, label='Plug In Est')
    plt.legend(loc='upper right')
    plt.show()


def compare_param_nonparam_bootstrap(B,plug_in_estimate,T_boot,T_boot_p,est_std_err, est_std_err_p, conf_ints):

    bins = np.linspace(0,1.2,50)
    #Plot Histogram: boostrap theta_hat_n outcomes
    # Creates two subplots and unpacks the output array immediately
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    ax1.hist(T_boot, bins,density=True, label='sampled')
    #ax1.title("Bootstrapped theta_hat_n: est std_err:{0:f}".format(est_std_err))
    ax1.set_ylabel('#of occurences/{0:d})'.format(B))
    ax1.set_title('NonParametric Bootstrap theta_n')
    ax1.axvline(plug_in_estimate, color='k', linestyle='dashed', linewidth=1)
    _, max_ = plt.ylim()
    ax1.text(plug_in_estimate + plug_in_estimate/10, max_ - max_/10,'plug in estimate: {:.2f}'.format(plug_in_estimate))
    (l,r) = conf_ints[0]
    ax1.axvline(x=l, color = 'r')
    ax1.axvline(x=r, color = 'r')
    (l_piv,r_piv) = conf_ints[1]
    ax1.axvline(x=l_piv, color = 'b')
    ax1.axvline(x=r_piv, color = 'b')
    (l_per,r_per) = conf_ints[2]
    ax1.axvline(x=l_per, color = 'g')
    ax1.axvline(x=r_per, color = 'g')

    ax2.hist(T_boot_p, bins,density=True, label='sampled')
    ax2.set_ylabel('#of occurences/{0:d})'.format(B))
    ax2.set_title('Parametric Bootstrap theta_n')
    #plt.title("Bootstrapped theta_hat_n: est std_err:{0:f}".format(est_std_err_p))
    ax2.axvline(plug_in_estimate, color='k', linestyle='dashed', linewidth=1)
    _, max_ = plt.ylim()
    ax2.text(plug_in_estimate + plug_in_estimate/10, max_ - max_/10,'plug in estimate: {:.2f}'.format(plug_in_estimate))
    (cip_l, cip_r) = conf_ints[3]
    ax2.axvline(x=cip_l, color = 'y')
    ax2.axvline(x=cip_r, color = 'y')
    plt.show()
    time.sleep(5)


def main(arguments):
    from problem_2 import check_density
    true_a   = 1
    true_b   = 3
    true_tau = 2 # mean
    iterations   = 20000
    sample_size = 10#,100,200,500]
    mle_tau_error_squared = np.zeros(iterations,dtype=np.float)
    pli_tau_error_squared = np.zeros(iterations,dtype=np.float)

    #approximating MSE with plug in MSE: take many samples, plot how
    # fast it converges
    for i in range(iterations):
        X = np.random.uniform(true_a, true_b,sample_size)
        mle_tau = (1/2)*(np.amax(X)+np.amin(X))
        pli_tau = np.mean(X)
        mle_tau_error_squared[i] = (mle_tau-true_tau)**2
        pli_tau_error_squared[i] = (pli_tau-true_tau)**2

    s = np.arange(1,iterations,10)
    l = s.size
    mle_mse = np.zeros(l,dtype=np.float)
    pli_mse = np.zeros(l,dtype=np.float)
    for index,trials in enumerate(s):
        mle_mse[index] = np.mean(mle_tau_error_squared[0:trials])
        pli_mse[index] = np.mean(pli_tau_error_squared[0:trials])

    plt.scatter(s,mle_mse,label='MLE')
    plt.scatter(s,pli_mse,label='Plug in')
    mse_analytical = 1/(3*sample_size)
    plt.axhline(y=mse_analytical, color='r', linestyle='-')
    plt.gca().set_ylabel('MSE: mean squared error of sampling distribution)'.format())
    plt.gca().set_xlabel('number of 10-sample trials')
    plt.gca().set_title('MSE: Max Likeli Hood vs NonParam Plug in')
    plt.legend(loc='upper right')
    plt.show()

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
