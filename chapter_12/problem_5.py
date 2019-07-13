from __future__ import print_function
import os
import time
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

#Bayesian Inference

def main(arguments):
    # f(p|xn) ~ f(xn|p)*Beta(a,b) = L(p)*Beta(a,b)
    # 3) Bern Likelihood(p) * Beta Prior(a,b) => Beta Posterior
    #      => conjugate distributions through Bern likelihood
    #      => Beta=Uniform prior is conjug distrib for Bern likelihood
    #    f(p1|xn) = L(p)*Beta(a,b)
    #             = p^s * (1-p)^(n-s) * p*(a-1) * (1-p)^(b-1)
    #             = p^(s+a-1) * (1-p)^(n+b-s-1)
    #             = Beta(s+a, n+b-s), s<-# sucesses

    n = 10
    s = 2
    x = np.linspace(-0.1,1.1,120)

    # i) f(p) = Beta(0.5,0.5) -> f(p|xn) ~ Beta(s+0.5,n+0.5-s)
    a,b = 0.5, 0.5
    pdf_1 = stats.beta.pdf(x, s+a, n+b-s)
    # ii) f(p) = Beta(1,1) -> f(p|xn) ~ Beta(s+1,n+1-s)
    a,b = 1, 1
    pdf_2 = stats.beta.pdf(x, s+a, n+b-s)
    # iii) f(p) = Beta(10,10) -> f(p|xn) ~ Beta(s+10,n+10-s)
    a,b = 10,10
    pdf_3 = stats.beta.pdf(x, s+a, n+b-s)
    # iv) f(p) = Beta(100,100) -> f(p|xn) ~ Beta(s+100,n+100-s)
    a,b = 100, 100
    pdf_4 = stats.beta.pdf(x, s+a, n+b-s)

    fig, (ax1) = plt.subplots(1, 1, sharex='col')
    ax1.plot(x,pdf_1,label='1/2')
    ax1.plot(x,pdf_2,label='1')
    ax1.plot(x,pdf_3,label='10')
    ax1.plot(x,pdf_4,label='100')
    ax1.legend()
    plt.show()

    #As the paramater a in  Beta(a,a) increases it comes to dominate the posterior.
    # posterior = Beta(s+a, n+a-s)\n as lim posterior --(a->+inf)--> Beta(a,a)
    # data (s here) influences the posterior less as a becomes larger
    # when prior is more confident <=> higher concentration of prob mass then data affects posterior less


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
