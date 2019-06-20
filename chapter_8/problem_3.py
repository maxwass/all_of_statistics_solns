from __future__ import print_function
import os
import time
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

#takes in array of samples to create empirical cdf, then evaluated the
# empirical cdf at values
def ecdf(samples, values):
    samples = np.asarray(samples)
    values = np.asarray(values)
    samples_sorted = np.sort(samples)
    num_samples = samples.shape[0]

    #to make faster, impliment binary search to find the smallest
    # sample greater than v, then linear search to find first sample
    # of this value
    output = np.ones(len(values))*num_samples
    for j,v in enumerate(values):
        for i,s in enumerate(samples_sorted):
            if(v<s):
                output[j] = i # add 1 for each its greater than
                break

    output = output/float(num_samples)
    return output

def main(arguments):
    from problem_3 import ecdf
    #test ecdf
    """
    print([0,1/3,1,1])
    print(ecdf([1,3,5],[0,2,5,10]))
    print([0,1/2,1/2,1,1])
    print(ecdf([1,1,5,5],[0,1,2,5,10]))
    print([0,1,1])
    print(ecdf([1],[0,1,2]))
    time.sleep(10)
    """
    n = 100
    trials = 10000
    in_bound = 0
    samples = np.zeros(n)
    upper_band = np.zeros(n)
    lower_band = np.zeros(n)
    plot = True
    which_distrib = 'normal' #or cauchy
    for trial in range(1,trials+1,1):
        if(which_distrib == 'normal'):
            samples = np.random.normal(0, 1, n)
        else:
            samples = np.random.standard_cauchy(n)
        samples = np.sort(samples)

        #create vector of upper/lower bands
        #95% confidence F(x) is in this band for all x
        alpha = 0.05 #for 95% confidence band
        eps = np.sqrt(np.log(2/alpha)/(2*n))
        upper_band = np.minimum(ecdf(samples,samples)+ eps, np.ones(n)) #cdf <= 1
        lower_band = np.maximum(ecdf(samples,samples)- eps, np.zeros(n)) #cdf >= 0

        #we know true cdf. check if it is in fact in these bands
        if(which_distrib== 'normal'):
            true_cdf = stats.norm.cdf(samples)
        else:
            true_cdf = stats.cauchy.cdf(samples)
        a = np.less_equal(true_cdf   , upper_band)
        b = np.greater_equal(true_cdf, lower_band)
        c = np.logical_and(a,b)
        if(np.all(c)):
            #true cdf is within bounds
            in_bound +=1
        if(np.remainder(trial,100))==0:
            print("trials run: {0:d}\n".format(trial))
            print("% which true CDF in Fn(x)+/ band over all x: {0:f}\n".format(in_bound/float(trial)))

    if(plot):
        x = np.sort(samples)
        y = np.arange(len(x))/float(len(x))
        plt.plot(x, y, 'bo',x, lower_band, x, upper_band)
        plt.show()


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
