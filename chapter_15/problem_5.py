from __future__ import print_function
import os, sys
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats, linalg
from scipy.stats import multivariate_normal

# write function to generate observations from a multivariate normal(u,s) distribution
# The key to this is understanding how any multivariate normal can be decomposed into
#  a transformed multivariate standard normal(0,I) distribution
#  X ~ N(u,s) => X = u + s^(1/2)Z, where Z~N(0,I)
def multivar_normal(mean,cov,observations):
    k        = cov.shape[1]
    sqrt_cov = linalg.sqrtm(cov)
    samples = np.zeros((observations,2))
    for i in range(observations):
        Z        = np.random.normal(0, 1, k)
        # X is now a sample of N(mean,cov)
        X = mean + np.dot(sqrt_cov,Z)
        samples[i,:] = X
        #print(samples)
        if(False):
            print(mean); print(cov)
            print('std normal sample:',end=' '); print(Z)
            print('sqrt of cov:'); print(sqrt_cov)
            print('confirm sqrt computed correctly: sqrt_cov^2')
            print(sqrt_cov*sqrt_cov)
            X_ = np.dot(sqrt_cov,Z) #should be 2x1 col vector
            print(X_)
            print('transformed sample: X = mean + cov^(1/2)*Z = '); print(X)
    return samples

def main(arguments):

    mean = np.array([0,10])
    cov  = np.matrix('1 0;0 6')
    observations = 10000
    samples = multivar_normal(mean,cov,observations)
    sample_mean = np.mean(samples,axis=0)
    sample_cov  = np.cov(np.transpose(samples))
    print('sample_mean'); print(sample_mean)
    print('sample_cov'); print(sample_cov)
    plt.scatter(samples[:,0], samples[:,1])
    plt.xticks(np.arange(round(np.amin(samples[:,0])), np.amax(samples[:,0]), step=2))
    plt.yticks(np.arange(round(np.amin(samples[:,1])), np.amax(samples[:,1]), step=2))
    plt.axis('scaled')
    plt.show()


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
