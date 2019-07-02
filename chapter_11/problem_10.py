from __future__ import print_function
import os
import time
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

#Hypothesis Testing

def main(arguments):
    #model: X1,...,Xn ~ Poisson(l)
    # H0: l  = 1
    # H1: l != 1/2
    # Define Statistics T = (l_mle - 1)/ste_est;
    #                   l_mle = sample_mean and ste_est = sqrt(l_mle/n)
    # Use CLT T ~ N(0,1) under H0 thus p_value = P(|T|>T_obs) = 2*std_norm_cdf(-|T_obs|)
    # Truth: l = 1

    l = 1
    n = 20
    alpha = 0.05
    count = 0
    #Perform many experiments with H0 being true. See how many times null
    # is rejected. How close is type I error rate (falseely reject null) to .05?
    iterations = 1000000
    p_vals = np.zeros(iterations)
    for i in range(iterations):
        X     = np.random.poisson(l,n)
        l_mle = np.mean(X)
        ste   = np.sqrt(l_mle/n) #using fischer information
        z     = (l_mle-l)/ste
        p_vals[i] = 2*stats.norm.cdf(-np.absolute(z))
        if(p_vals[i]<=alpha):
            count+=1
    print('number of rejections: {0:d}/{1:d}'.format(count,iterations))
    print('type I error: {0:f}'.format(count/iterations))
    print("type I error is consitantly close to 0.05. This suggets that our test statistic (lambda_mle - lamba_true)/standard_error is in fact ~= N(0,1) in distribution.")
    plt.hist(p_vals,20)
    plt.axhline(y=0.05*iterations, color='r', linestyle='-')
    plt.show()







if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
