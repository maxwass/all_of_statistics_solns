from __future__ import print_function
import os
import time
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
#comparing bootstrap confidence interval methods
#confidence interval file in higher folder
sys.path.append(os.path.dirname(os.path.abspath('../confidence_interval.py')))
from confidence_interval import normal_95_confid_interval,pivotal_confid_interval,percentile_confid_interval


#comparing the three confidence interval methods (Normal, Pivotal,
# and Percentile) to estimate the skew of the rv exp(Y), Y~N(0,1)

#Analytically:
# E[exp(Y)] =

def main(arguments):

    true_skew    = 6.184
    iterations   = 50
    sample_size_ = [50]#,100,200,500]
    B_           = [100000]#[1000,5000,10000] # number of bootstrap samples to take

    for sample_size in sample_size_:
        for B in B_:
            coverage = np.zeros(3) #number true is in 3 confidence intervals
            for j in range(1,iterations):
                Y = np.random.standard_normal(sample_size)
                X = np.exp(Y)
                data = X
                """
                bins = np.linspace(np.floor(np.amin(Y)), np.ceil(np.max(X)), 50)
                plt.hist(Y, bins, alpha=0.5, label='normal')
                plt.hist(X, bins, alpha=0.5, label='exp(normal)')
                plt.legend(loc='upper right')
                plt.show()
                """
                # estimating skewness of new dsitribution X
                plug_in_estimate = stats.skew(X)

                #Bootstrap Sampling
                # uniformly sample, with replacement, from indeces (0,1,2,...sample_size)
                #  Use selected indeces to pull out data from data vector. Feed selected
                #  data through T, our statistic, in this case correlation, and store
                #  in T_boot
                indeces = np.arange(sample_size)
                T_boot  = np.zeros(B, dtype=float)
                for i in range(B):
                    boot_indeces = np.random.choice(indeces,sample_size,replace=True)
                    boot_sample  = data[boot_indeces]
                    T_boot[i]    = stats.skew(boot_sample)

                est_std_err = np.std(T_boot)

                #Confidence Intervals
                #3 methods for 1-alpha = 95% confidence intervals:
                alpha = 0.05
                (l,r) = normal_95_confid_interval(plug_in_estimate, T_boot)
                #print("Normal based 95% confid interval: ({0:f}, {1:f})".format(l,r))
                if(true_skew>l and true_skew<r):
                    coverage[0]+=1
                (l_piv, r_piv) = pivotal_confid_interval(plug_in_estimate, T_boot, alpha)
                #print("Pivotal based 95% confid interval: ({0:f}, {1:f})".format(l_piv,r_piv))
                if(true_skew>l_piv and true_skew<r_piv):
                    coverage[1]+=1
                # Percentile Interval
                (l_per, r_per) = percentile_confid_interval(plug_in_estimate, T_boot, alpha)
                #print("Percentile based 95% confid interval: ({0:f}, {1:f})".format(l_per,r_per))
                if(true_skew>l_per and true_skew<r_per):
                    coverage[2]+=1
                #print("{0:d}th iteration".format(j))
            print("n = {0:d}, B = {1:d}".format(sample_size,B))
            cov_prob = coverage/float(iterations)
            print("\tNorm: {0:f}, Piv: {1:f}, Per: {2:f}".format(cov_prob[0],cov_prob[1],cov_prob[2]))
            #print('\n'.join('{}: {}'.format(*k) for k in enumerate(coverage/float(j))))
                #print("{0:d}th iteration: {1:f}".format(j,coverage/float(j)))
    """
    #Plot Histogram: boostrap correlation outcomes
    plt.hist(T_boot, bins='auto')
    plt.title("Bootsrrapped correlation coefficients: est std_err:{0:f}".format(est_std_err))
    plt.xlabel('bootstrapped correlation coefficient')
    plt.ylabel('number of occurences (out of {0:d})'.format(B))
    plt.axvline(plug_in_estimate, color='k', linestyle='dashed', linewidth=1)
    _, max_ = plt.ylim()
    plt.text(plug_in_estimate + plug_in_estimate/10, max_ - max_/10,'plug in estimate: {:.2f}'.format(plug_in_estimate))
    plt.axvline(x=l, color = 'r')
    plt.axvline(x=r, color = 'r')
    plt.axvline(x=l_piv, color = 'b')
    plt.axvline(x=r_piv, color = 'b')
    plt.axvline(x=l_per, color = 'g')
    plt.axvline(x=r_per, color = 'g')
    plt.show()
    """
if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
