from __future__ import print_function
import os
import time
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

#Bayesian Inference

def main(arguments):
    #model: X1,...,Xn ~ N(u,1)
    # f(u) ~ 1 (uniform prior)
    u, sigma, n = 5, 1, 100


    # a
    X = np.random.normal(u,sigma,n)

    # b
    U = np.linspace(3,7,100)
    L = np.zeros_like(U)

    #loop over a few selected mu's to plot likelihood
    #f(mu|xn) ~ prod i: N(xi|mu;mu,1) * 1
    #         = prod i: N(mu|xi;xi,1) * 1
    sample_mean = np.mean(X)
    s = -n/(2*sigma**2)
    for i,mu in enumerate(U):
        a    = (sample_mean-mu)**2
        l_i    = np.exp(s*a)
        L[i] = l_i

    norm_const = np.sum(L)
    fig, (ax1,ax2) = plt.subplots(2, 1, sharex='col')
    ax1.plot(U,L/norm_const)

    # c
    #Simulate 1000 draws from posterior distrib
    # f(u|xn) ~ exp(-1/(2sigma^2)/n *(sample_mean-u)^2)
    posterior_samples = np.random.normal(sample_mean,sigma/n,1000)
    ax2.hist(posterior_samples, density=True)

    # d
    # theta := exp(u)
    # find posterior density for theta analytically and
    # via simulation

    #analytically:
    # A_theta(z) := {u: exp(u) <= z} = (-inf,ln(z))
    # F_theta(z) := P(A_theta(z))
    #             = integral from -inf to ln(z) of N(u;sample_mean, sigma^2/n)
    # f_theta(z) := dF/dz = N(ln(z);sample_mean,sigma^2/n)*(1/z)
    # link to graph: https://www.wolframalpha.com/input/?i=1%2Fsqrt(2*pi)+*+exp((-1%2F2)*(ln(x)-5)%5E2)%2Fx,+for+x%3D0..500
    z = np.linspace(3,200,1000)
    ln_z = np.log(z)
    sigma_post = (sigma**2)/n
    mean_post  = sample_mean
    a          = (-1/(2*sigma_post**2))*(np.log(z)-sample_mean)**2
    f_theta    = (1/np.sqrt(2*np.pi*sigma_post**2))*np.exp(a)


    #simulation
    # transform draws from posterior:
    #  e^y1, e^y2, ...
    # they are now iid samples from f_theta
    exp_x = np.exp(posterior_samples)

    #e
    # find percentile via simulated draws
    z_025 = np.percentile(exp_x,2.5)
    z_975 = np.percentile(exp_x,97.5)
    print('percentile: ({0:f}, {1:f})'.format(z_025,z_975))

    fig1, (ax3,ax4) = plt.subplots(2, 1, sharex='col')
    ax3.plot(z,f_theta)
    #ax3.axvline(x=z_025)
    #ax3.axvline(x=z_975)

    ax4.hist(exp_x, density=True)

    #E) from thm 12.5 we know that the posterior for theta will
    # be N(theta_mle, se_hat_est^2) where
    #  theta_mle  = exp(u_mle)
    #  se_theta   = se_u|dexp(u)/du|
    #       se_u       = 1/sqrt(nI(u)), I(u) = 1/sigma^2
    #       dexp(u)/du = exp(u)
    # thus theta|xn ~ N(exp(u_mle), se_u(u_mle)*|exp(u_mle)|)
    #  (theta|xn - exp(u_mle) )/ste_theta ~ N(0,1)
    # thus 95% CI: exp(u_mle) +/- se_theta*1.96
    u_mle     = sample_mean
    theta_mle = np.exp(u_mle)
    I         = 1/(sigma**2)
    ste_u     = 1/np.sqrt(n*I)
    ste_theta = ste_u*np.absolute(np.exp(u_mle))
    z_l = theta_mle - 1.96*ste_theta
    z_r = theta_mle + 1.96*ste_theta
    print('95 CI: ({0:f}, {1:f})'.format(z_l,z_r))
    ax3.axvline(x=z_l)
    ax3.axvline(x=z_r)


    plt.show()






if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
