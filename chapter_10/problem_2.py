from __future__ import print_function
import os
import time
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

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
def main(arguments):
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
