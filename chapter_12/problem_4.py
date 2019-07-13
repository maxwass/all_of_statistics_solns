from __future__ import print_function
import os
import time
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

#Bayesian Inference

def main(arguments):
    #
    placebo_size,    treat_size    = 50, 50
    placebo_improve, treat_improve = 30, 40
    p_mle_placebo = placebo_improve/placebo_size
    p_mle_treat   = treat_improve/treat_size

    # A & B
    # ste and 90% CI via delta method
    # tau = g(p1,p2) --> tau_mle = g(p1_mle,p2_mle)
    # for ste formulas: https://stats.stackexchange.com/questions/29641/standard-error-for-the-mean-of-a-sample-of-binomial-random-variables
    tau_mle = p_mle_treat - p_mle_placebo
    var_placebo = p_mle_placebo*(1-p_mle_placebo)/placebo_size

    #delta method
    var_treat   = p_mle_treat*(1-p_mle_treat)/treat_size
    ste_tau_delta = np.sqrt(var_treat + var_placebo)
    tau_90_l_delta = tau_mle - 1.645*ste_tau_delta
    tau_90_r_delta = tau_mle + 1.645*ste_tau_delta

    #bootstrap method
    B = 10000
    X = np.zeros(B)
    for i in range(B):
        treatment_resample = np.random.binomial(treat_size,   p_mle_treat)
        placebo_resample   = np.random.binomial(placebo_size, p_mle_placebo)
        X[i] = treatment_resample/treat_size - placebo_resample/placebo_size
    ste_tau_boot = np.std(X)
    tau_90_l_boot = tau_mle - 1.645*ste_tau_boot
    tau_90_r_boot = tau_mle + 1.645*ste_tau_boot

    print('tau := p2-p1')
    print("\ttau mle: {0:f}".format(tau_mle))
    print("\tz_.05: {0:f}".format(1.645))
    print('\tste: delta: {0:f}, boot: {1:f}'.format(ste_tau_delta, ste_tau_boot))
    print('\t90% CI delta: ({0:f}, {1:f})'.format(tau_90_l_delta,tau_90_r_delta))
    print('\t90% CI boot : ({0:f}, {1:f})'.format(tau_90_l_boot, tau_90_r_boot))

    # C
    # prior over joint distribution f(p1,p2) ~ 1
    #     use simulation to find the posterior mean and posterior
    #     90 interval for tau

    #analytically: https://www.wolframalpha.com/input/?i=integral+%5B0,1%5D+integral+%5B0,1%5D+(y-x)(x%5E30)*(1-   x)%5E(50-30)*(y%5E40)*(1-y)%5E(50-40)%2F(7.9415*10%5E-28)+dx+dy
    # tau_mean = integral [0,1] integral [0,1] (p2-p1)*L(p1)*L(p2)/norm_const dp1 dp2= 0.1923
    # integral: https://www.wolframalpha.com/input/?i=integral+%5B0,1%5D+integral+%5B0,1%5D+(y-x)(x%5E30)*(1-x)%5E(50-30)*(y%5E40)*(1-y)%5E(50-40)%2F(7.9415*10%5E-28)+dx+dy


    # 1) f(p1,p2|xn,ym) ~ L(p1)*L(p2) = f(p1|xn)*f(p2|ym) <=> independent
    # 2) Uniform[0,1]   = Beta[1,1]
    # 3) Bern Likelihood(p1) * Beta Prior(a,b) => Beta Posterior
    #      => conjugate distributions through Bern likelihood
    #      => Beta=Uniform prior is conjug distrib for Bern likelihood
    #    f(p1,p2|xn,ym)=f(p1|xn)*f(p2|ym)
    #                  = L(p1)*U[0,1]       * L(p2)*U[0,1]
    #                  = L(p1)*Beta(1,1)    * L(p2)*Beta(1,1)
    #                  = Beta(s_n+1,s_n+1-sn)*Beta(1+s_m, 1+m-s_m)
    #                        where s_i := # of sucesses
    s_n, s_m = placebo_improve, treat_improve
    n, m     = placebo_size   , treat_size
    TAU = np.zeros(B)
    for i in range(B):
        p1_sample  = np.random.beta(1+s_n, 1+n-s_n)
        p2_sample  = np.random.beta(1+s_m, 1+m-s_m)
        TAU[i] = p2_sample-p1_sample
    simulation_mean = np.mean(TAU)
    print("\tanalytical mean: 0.1923, sim mean: {0:f}".format(simulation_mean))

    # 90% interval for tau
    # we apply transformation to our posterior = density for joint of p1 & p2
    # A(z):= {(p1,p2): (p2-p1)<= z} =
    # = integral [a,b] f(p1,p2,) ... very hard... use simulation

    #simulation:
    tau_90_l = np.percentile(TAU,5)
    tau_90_r = np.percentile(TAU,95)
    print('\t90% (posterior mass) interval: ({0:f}, {1:f})'.format(tau_90_l,tau_90_r))

    # D
    p_treat_odds   =  p_mle_treat  /(1-p_mle_treat)
    p_placebo_odds =  p_mle_placebo/(1-p_mle_placebo)
    phi_mle = np.log(p_placebo_odds/p_treat_odds)

    # delta : link to calculation below
    # https://www.wolframalpha.com/input/?i=%7B%7B25%2F6,-25%2F4%7D%7D.%7B%7B9,0%7D,%7B0,10%7D%7D.%7B%7B25%2F6%7D,%7B-25%2F4%7D%7D%2F1875
    ste_phi_delta = 0.54
    phi_90_l_delta = phi_mle - 1.645*ste_phi_delta
    phi_90_r_delta = phi_mle + 1.645*ste_phi_delta

    # bootstrap
    PHI = np.zeros(B)
    for i in range(B):
        p1_sample  = np.random.beta(1+s_n, 1+n-s_n) # placebo
        p2_sample  = np.random.beta(1+s_m, 1+m-s_m) # treatment
        p1_odds    = p1_sample/(1-p1_sample)
        p2_odds    = p2_sample/(1-p2_sample)
        PHI[i] = np.log(p1_odds/p2_odds)
    ste_phi_boot= np.std(PHI)

    phi_90_l_boot = phi_mle - 1.645*ste_phi_boot
    phi_90_r_boot = phi_mle + 1.645*ste_phi_boot

    print('phi=log odds ratio:')
    print('\tphi_mle: {0:f}'.format(phi_mle))
    print('\tste: delta: {0:f}, boot: {1:f}'.format(ste_phi_delta, ste_phi_boot))
    print('\t90% CI delta: ({0:f}, {1:f})'.format(phi_90_l_delta,phi_90_r_delta))
    print('\t90% CI boot : ({0:f}, {1:f})'.format(phi_90_l_boot,phi_90_r_boot))
    # e) posterior mean and 90% interval for phi
    phi_mean = np.mean(PHI)
    print("\tanalytical mean: ____, sim mean: {0:f}".format(phi_mean))
    #simulation:
    phi_90_l = np.percentile(PHI,5)
    phi_90_r = np.percentile(PHI,95)
    print('\t90% (posterior) interval: ({0:f}, {1:f})'.format(phi_90_l,phi_90_r))


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
