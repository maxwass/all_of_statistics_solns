from __future__ import print_function
import os
import time
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import re

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
    from fijiquakes import ecdf
    n = 1000
    plot = True
    path = '/Users/maxwasserman/Desktop/all_of_statistics/chapter_8/fijiquakes.dat'
    quakes_file = open(path,'r')
    samples = np.zeros(n, dtype=float)

    with open(path,'r') as f:
        for i, line in enumerate(f):
            if(i!=0):
                mag = line.split()[4]
                samples[i-1]=float(mag)

    samples = np.sort(samples)
    est_cdf = ecdf(samples, samples)

    #create vector of upper/lower bands
    #95% confidence F(x) is in this band for all x
    alpha = 0.05 #for 95% confidence band
    eps = np.sqrt(np.log(2/alpha)/(2*n))
    upper_band = np.minimum(est_cdf+ eps, np.ones(len(samples))) #cdf <= 1
    lower_band = np.maximum(est_cdf- eps, np.zeros(len(samples))) #cdf >= 0

    #compute a 95% confidence interval for F(4.9)-F(4.3)= P(4.3<X<4.9)
    # Use Normal Based Interval
    [a,b] = ecdf(samples,[4.3,4.9]) #evaluate ecdf at 4.3 and 4.9
    print('ecdf evaluated at 4.3: {0:f}, 4.9: {1:f}'.format(a,b))
    stand_error_a = a*(1-a)/n #this is actually squared stand_error_a
    stand_error_b = b*(1-b)/n

    print('std error(Fn_hat(4.3))^2 = {0:f}, std error(Fn_hat(4.9))^2 = {1:f}'.format(stand_error_a, stand_error_b))

    stand_error = np.sqrt(stand_error_a+stand_error_b)
    print('stand_error total = stand_error= {0:f}'.format(stand_error))

    l = (b-a) - 2*stand_error
    r = (b-a) + 2*stand_error
    print('interval: ({0:f}, {1:f})'.format(l,r))
    time.sleep(20)

    if(plot):
        x = samples
        y = est_cdf
        plt.plot(x, y, 'bo',x, lower_band, x, upper_band)
        plt.show()

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
