from __future__ import print_function
import os
import time
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

#LSAT/GPA DATA
#Standard Error of the correlation between LSAT score and GPA using the boostrap
def main(arguments):
    #confidence interval file in higher folder
    sys.path.append(os.path.dirname(os.path.abspath('confidence_interval.py')))
    from confidence_interval import normal_95_confid_interval,pivotal_confid_interval,percentile_confid_interval

    #hardcode lsat/gpa data
    lsat = np.array([576,635,558,578,666,580,555,661,651,605,653,575,545,572,594])
    gpa  = np.array([3.39,3.30,2.81,3.03,3.44,3.07,3.00,3.43,3.36,3.13,3.12,2.74,2.76,2.88,3.96])
    data = np.column_stack((lsat,gpa))

    sample_size = len(lsat)

    #calculate sample correlation using definition
    cov = 0.0
    for i,point in enumerate(lsat):
        cov += (lsat[i]-np.mean(lsat)) * (gpa[i]-np.mean(gpa))
    corr = (1/len(lsat))*cov/(np.std(lsat)*np.std(gpa))
    print("calculated corr: {0:f}".format(corr))
    print("numpy library corr: {0:f}".format(np.corrcoef(data[:,0],data[:,1])[0,1]))


    plug_in_estimate = np.corrcoef(data[:,0],data[:,1])[0,1]

    #Bootstrap Sampling
    # uniformly sample, with replacement, from indeces (0,1,2,...sample_size)
    #  Use selected indeces to pull out data from data vector. Feed selected
    #  data through T, our statistic, in this case correlation, and store
    #  in T_boot
    indeces = np.arange(sample_size)
    B       = 10000 # number of bootstrap samples to take
    T_boot  = np.zeros(B, dtype=float)
    for i in range(B):
        boot_indeces = np.random.choice(indeces,sample_size,replace=True)
        boot_sample  = data[boot_indeces]
        T_boot[i] = np.corrcoef(boot_sample[:,0],boot_sample[:,1])[0,1]
    est_std_err = np.std(T_boot)

    #Confidence Intervals
    #3 methods for 1-alpha = 95% confidence intervals:
    alpha = 0.05
    # Normal: = Tn +/- z_a/2*est_std_err
    (l,r) = normal_95_confid_interval(plug_in_estimate, T_boot)
    print("Normal based 95% confid interval: ({0:f}, {1:f})".format(l,r))
    # Pivotal Intervals: (2*Tn - quantile(T_boot,0.9725), 2*Tn - quantile(T_boot,.025))
    (l_piv, r_piv) = pivotal_confid_interval(plug_in_estimate, T_boot, alpha)
    print("Pivotal based 95% confid interval: ({0:f}, {1:f})".format(l_piv,r_piv))
    # Percentile Interval
    (l_per, r_per) = percentile_confid_interval(plug_in_estimate, T_boot, alpha)
    print("Percentile based 95% confid interval: ({0:f}, {1:f})".format(l_per,r_per))

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

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
