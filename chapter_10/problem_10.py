from __future__ import print_function
import os
import time
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


def main(arguments):
    theta_true = 1
    iterations   = 100000
    n=50

    # X1, ... ,Xn ~ Uniform(0,theta = 1 unknown)
    # theta_mlw = max(X1,...,Xn)
    X = np.random.uniform(0,theta_true,n)
    theta_mle     = np.amax(X)
    theta_plug_in = np.amax(X)

    ##Estimating Variance of Sampling Distribution of theta

    #Method 1: Parametric Bootstrap
    theta_boot = np.zeros(iterations,dtype=np.float)
    for i in range(iterations):
        x_resample    = np.random.uniform(0,theta_mle,n)
        theta_boot[i] = np.amax(x_resample)
    ste_hat_boot   = np.std(theta_boot)

    #Method 2: NonParametric Bootstrapping
    theta_boot_nonparam = np.zeros(iterations,dtype=np.float)
    for i in range(iterations):
        x_np_resample          = np.random.choice(X,n,replace=True)
        theta_boot_nonparam[i] = np.amax(x_np_resample)
    ste_hat_boot_non_param   = np.std(theta_boot_nonparam)

    #Simulate True sampling distributien
    theta_sample_true = np.zeros(iterations,dtype=np.float)
    for i in range(iterations):
         x_sample         = np.random.uniform(0,theta_true,n)
         theta_sample_true[i] = np.amax(x_sample)
    ste_theta_true = np.std(theta_sample_true)

    #Analytical Sampling Distribution
    s = np.linspace(0,theta_true,100)
    sample_pdf = s**(n-1)*n/(theta_true**n)


    #compute 1-a confidence interval
    # intuition: If I were to repeat this experiment many times, the
    #             proportion of times the true parameter theta would
    #             be inside the confid interval is 1-a
    #95% confidence interval: theta_mle +/- 2*ste_hat
    z = 2
    ci_boot_l    = theta_mle - z*ste_hat_boot
    ci_boot_r    = theta_mle + z*ste_hat_boot
    ci_np_boot_l = theta_plug_in - z*ste_hat_boot_non_param
    ci_np_boot_r = theta_plug_in + z*ste_hat_boot_non_param
    ci_true_l    = theta_true - 2*ste_theta_true
    ci_true_r    = theta_true + 2*ste_theta_true

    ax1 = plt.subplot(3,1,1)
    bins = np.linspace(np.amin(theta_boot),np.amax(theta_boot),100)
    ax1.hist(theta_boot,bins,label='theta bootstrap')
    ax1.axvline(x=theta_mle, color='r', linestyle='dotted',label="theta_mle")
    ax1.axvline(x=ci_boot_l,color='b',linestyle='-',label='95% CI:se_boot={0:f}'.format(ste_hat_boot))
    ax1.axvline(x=ci_boot_r,  color='b', linestyle='-')
    #ax1.axvline(x=ci_delta_l, color='g', linestyle='-')
    #ax1.axvline(x=ci_delta_r,color='g',linestyle='-',label='95% CI:se_delta={0:f}'.format(ste_hat_delta))
    ax1.legend(loc='upper right')
    ax1.set_xlabel('approx of sampling distrib: parametric bootstrap')

    ax2 = plt.subplot(3,1,2, sharex=ax1)
    ax2.hist(theta_boot_nonparam,bins,label='theta nonparam bootstrap')
    ax2.axvline(x=theta_mle, color='m', linestyle='dotted',label="theta plug in")
    ax2.axvline(x=ci_np_boot_l,color='b',linestyle='-',label='95% CI:se_np_boot={0:f}'.format(ste_hat_boot_non_param))
    ax2.axvline(x=ci_np_boot_r,color='b',linestyle='-')
    ax2.legend(loc='upper right')
    ax2.set_xlabel('approx of sampling distrib: NON-parametric bootstrap')

    ax3 = plt.subplot(3,1,3, sharex=ax1)
    bins_ = np.linspace(np.amin(theta_sample_true),np.amax(theta_sample_true),100)
    ax3.hist(theta_sample_true,bins_,label='true sampling distrib theta')
    ax3.axvline(x=theta_true, color='m', linestyle='dotted',label="theta true")
    ax3.axvline(x=ci_true_l,color='b',linestyle='-',label='95% CI:se_true={0:f}'.format(ste_theta_true))
    ax3.axvline(x=ci_true_r,color='b',linestyle='-')
    ax3.legend(loc='upper right')
    ax3.set_xlabel('true sampling distribution of theta')




    plt.gcf().suptitle('Approx Sampling Distribs (n = {0:d})of theta=e^u, X~N(u,1):\nMLE (param boot) vs Plug In Estimator (nonparam boot) vs Direct Sampling From True Distrib'.format(n))
    plt.show()

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
