from __future__ import print_function
import os
import time
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


def main(arguments):
    n = 100
    mu = 5
    var = 1
    theta_true = np.exp(mu)
    iterations   = 100000

    # X1, ... ,Xn ~ Normal(mu unknown = 5, variance = 1)
    # theta       = exp(mu) = g(mu)

    X = np.random.normal(mu,var,n)
    mu_mle = np.mean(X)
    mu_pli = np.mean(X) #plug in estimate
    theta_mle = np.exp(mu_mle)
    theta_plug_in = np.exp(mu_pli)

    ##Estimating Variance of Sampling Distribution of theta

    #Method 1: Delta Method: se_hat = abs(g')*sqrt(1/In(mu_mle))
    #           where g = exp(mu), In(mu_mle) = fisher information
    #           of mu evaluated at mu = mu_mle
    fisher_info   = np.sqrt(var/n)
    ste_hat_delta = np.exp(mu_mle)*fisher_info

    #Method 2: Parametric Bootstrap
    theta_boot = np.zeros(iterations,dtype=np.float)
    for i in range(iterations):
        x_resample    = np.random.normal(mu_mle,var,n)
        mu_boot       = np.mean(x_resample)
        theta_boot[i] = np.exp(mu_boot)
    ste_hat_boot   = np.std(theta_boot)

    #Method 3: NonParametric Bootstrapping
    theta_boot_nonparam = np.zeros(iterations,dtype=np.float)
    for i in range(iterations):
        x_np_resample    = np.random.choice(X,n,replace=True)
        mu_np_boot       = np.mean(x_np_resample)
        theta_boot_nonparam[i] = np.exp(mu_np_boot)

    ste_hat_boot_non_param   = np.std(theta_boot_nonparam)

    #Simulate True sampling distribution
    theta_sample_true = np.zeros(iterations,dtype=np.float)
    for i in range(iterations):
         x_sample         = np.random.normal(mu,var,n)
         mu_sample        = np.mean(x_sample)
         theta_sample_true[i] = np.exp(mu_sample)
    ste_theta_true = np.std(theta_sample_true)

    #compute confidence interval
    #95% confidence interval: z +/- 1.64*ste_hat
    z = 2
    ci_delta_l   = theta_mle - z*ste_hat_delta
    ci_delta_r   = theta_mle + z*ste_hat_delta
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
    ax1.axvline(x=ci_delta_l, color='g', linestyle='-')
    ax1.axvline(x=ci_delta_r,color='g',linestyle='-',label='95% CI:se_delta={0:f}'.format(ste_hat_delta))
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




    plt.gcf().suptitle('Approx Sampling Distribs of theta=e^u, X~N(u,1):\nMLE (param boot) vs Plug In Estimator (nonparam boot) vs Direct Sampling From True Distrib')
    plt.show()

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
