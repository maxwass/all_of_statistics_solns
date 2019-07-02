from __future__ import print_function
import os
import time
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

#Hypothesis Testing

def main(arguments):
    #Hardcoded Data
    twain_essays = np.array([.225,.262,.217,.240,.230,.229,.235,.217])
    snodg_essays = np.array([.209,.205,.196,.210,.202,.207,.224,.223,.220,.201])
    iterations   = 100000
    print("twain: mean: {0:f}, var: {1:f}, len: {2:d}".format(np.mean(twain_essays),np.var(twain_essays),twain_essays.size))
    print("snodg: mean: {0:f}, var: {1:f}, len: {2:d}".format(np.mean(snodg_essays),np.var(snodg_essays),snodg_essays.size))

    #Large Sample Approach: Boostrap ste approximation

    #plug in estimator for difference of means
    d_plugin = np.mean(twain_essays) - np.mean(snodg_essays)

    X = np.zeros(iterations, dtype=np.float)
    for i in range(iterations):
        twain_samples = np.random.choice(twain_essays, twain_essays.size, replace=True)
        snodg_samples = np.random.choice(snodg_essays, snodg_essays.size, replace=True)
        X[i]          = np.mean(twain_samples) - np.mean(snodg_samples)

    ste_boot = np.std(X)
    var_analytic = np.var(twain_essays, dtype=np.float64)/twain_essays.size + np.var(snodg_essays,dtype=np.float64)/snodg_essays.size
    ste_analytic = np.sqrt(var_analytic)

    z_analytic = 10**5*((d_plugin - 0)/(ste_analytic*10**5))#multiply for numerical purposes
    z_boot     = (d_plugin - 0)/ste_boot
    p_value_analytic = 2*stats.norm.cdf(-np.absolute(z_analytic))
    p_value_boot     = 2*stats.norm.cdf(-np.absolute(z_boot))

    print('diff means plug in: {0:f}, \n\tste analytic: {1:f}, ste boot {2:f}'.format(d_plugin,ste_analytic, ste_boot))
    print('\n\n95% confidence interval: \n\t({0:f}, {1:f})'.format(d_plugin-1.96*ste_analytic,d_plugin+1.96*ste_analytic))
    print('p_value_analytic := 2*std_norm_cdf({0:f}) = {1:f}%'.format(z_analytic,100*p_value_analytic))
    print('p_value_boot     := 2*std_norm_cdf({0:f}) = {1:f}%'.format(z_boot,100*p_value_boot))

    print("Interpretation: a low p value shows the results are statistically significat. But the 95% confidence interval shows that while significant, the difference is quite small, leading us to question the conclusion that the authors are different.")


    #Small Sample Approach: Permutation Test
    #test_stat(X1,...,XN,Y1,...,YM) = |mean(X) - mean(Y)|
    T_obs      = np.absolute(np.mean(twain_essays) - np.mean(snodg_essays))
    T          = np.zeros(iterations, dtype=np.float)
    all_essays = np.concatenate((twain_essays, snodg_essays))
    count = 0
    for i in range(iterations):
        #sample from (N+M)! combinations
        perm = np.random.permutation(all_essays)
        #compute test statistics on this
        diff_mean = np.mean(perm[:twain_samples.size]) - np.mean(perm[twain_samples.size:])
        T[i]      = np.absolute(diff_mean)
        if(T[i]>=T_obs):
            count+=1
    p_val_perm = count/iterations
    print('\n\npermuation test: p_value = {0:f} %. Very small. Formally Reject H0 under an alpha .05'.format(100*p_val_perm))
    print('Interpretation: p-value is the probability of the test statistic T being larger than the observed value T_obs under H_0. Thus is the p_value is very small then either (i) a very unlikely outcome occured and H0 is still true or (ii) H0 is not true.')
if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
