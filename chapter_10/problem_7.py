from __future__ import print_function
import os
import time
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


def main(arguments):
    n_1 = n_2 = 200
    x_1,x_2   = 160, 148
    p_1_mle   = x_1/n_1
    p_2_mle   = x_2/n_2
    phi_mle   = p_1_mle-p_2_mle
    iterations   = 100000

    #find 90% confidence interval around phi_hat = p_1_hat-p_2_hat
    phi_boot = np.zeros(iterations,dtype=np.float)

    ##Estimating Variance of Sampling Distribution of Phi

    #Method 1: Multiparameter Delta Method
    term1 = p_1_mle*(1-p_1_mle)/n_1
    term2 = p_2_mle*(1-p_2_mle)/n_2
    ste_hat_delta = (term1+term2)**(0.5)

    #Method 2: Bootstrapping
    #Parametric Bootstrap
    for i in range(iterations):
        x_1_resample = np.random.binomial(n_1,p_1_mle,1)
        x_2_resample = np.random.binomial(n_2,p_2_mle,1)
        p_1_mle_boot = x_1_resample/n_1
        p_2_ml2_boot = x_2_resample/n_2
        phi_boot[i]  = p_1_mle_boot - p_2_ml2_boot

    ste_hat_boot   = np.std(phi_boot)

    #Method 3: NonParametric Bootstrapping
    data_1         = np.zeros(n_1)
    data_2         = np.zeros(n_2)
    data_1[:x_1-1] = 1 #'create' the data
    data_2[:x_2-1] = 1
    phi_plug_in    = np.count_nonzero(data_1)/n_1 - np.count_nonzero(data_2)/n_2
    phi_boot_nonparam = np.zeros(iterations,dtype=np.float)

    for i in range(iterations):
        data_1_resample = np.random.choice(data_1,n_1,replace=True)
        data_2_resample = np.random.choice(data_2,n_2,replace=True)
        p_1_est_boot    = np.count_nonzero(data_1_resample)/n_1
        p_2_est_boot    = np.count_nonzero(data_2_resample)/n_2
        phi_boot_nonparam[i]  = p_1_est_boot - p_2_est_boot

    ste_hat_boot_non_param   = np.std(phi_boot_nonparam)


    #compute confidence interval
    #90% confidence interval: z +/- 1.64*ste_hat
    z = 1.64
    ci_boot_l  = phi_mle-z*ste_hat_boot
    ci_boot_r  = phi_mle+z*ste_hat_boot
    ci_np_boot_l = phi_plug_in-z*ste_hat_boot_non_param
    ci_np_boot_r = phi_plug_in+z*ste_hat_boot_non_param
    ci_delta_l = phi_mle-z*ste_hat_delta
    ci_delta_r  = phi_mle+z*ste_hat_delta

    ax1 = plt.subplot(2,1,1)
    bins = np.linspace(np.amin(phi_boot),np.amax(phi_boot),100)
    ax1.hist(phi_boot,bins,alpha=0.5,label='phi bootstrap')
    ax1.axvline(x=phi_mle, color='r', linestyle='dotted',label="phi_mle")
    ax1.axvline(x=ci_boot_l,color='b',linestyle='-',label='90% CI:se_boot={0:f}'.format(ste_hat_boot))
    ax1.axvline(x=ci_boot_r,  color='b', linestyle='-')
    ax1.axvline(x=ci_delta_l, color='g', linestyle='-')
    ax1.axvline(x=ci_delta_r,color='g',linestyle='-',label='90% CI:se_delta={0:f}'.format(ste_hat_delta))
    ax1.legend(loc='upper right')
    ax1.set_xlabel('Phi from parametric bootstrap')

    ax2 = plt.subplot(2,1,2, sharex=ax1)
    ax2.hist(phi_boot_nonparam,bins,alpha=0.5,label='phi nonparam bootstrap')
    ax2.axvline(x=phi_mle, color='m', linestyle='dotted',label="phi plug in")
    ax2.axvline(x=ci_np_boot_l,color='b',linestyle='-',label='90% CI:se_np_boot={0:f}'.format(ste_hat_boot_non_param))
    ax2.axvline(x=ci_np_boot_r,color='b',linestyle='-')
    ax2.legend(loc='upper right')
    ax2.set_xlabel('Phi from NON-parametric bootstrap')

    plt.gcf().suptitle('MLE (param boot) & Plug In Estimator (nonparam boot): Sampling Distribution of Phi = p1-p2'.format())
    plt.show()

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
