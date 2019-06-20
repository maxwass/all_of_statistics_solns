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
# and Percentile) to estimate the parameter theta of Uniform[0,theta]

#define estimator theta_hat = max{Xi},i=1->n
#analytic pdf of theta_hat_n =
# { (n/theta^n)*z^(n-1), z in [0,theta]
# {     0              , otherwise


#take many true samples of size sample_size to approximate the density
# of theta_n = max(Xi), i=1->sample_size
def check_density(sample_size,K, true_theta):
    theta_hat_n = np.zeros(K) #store estimates from each n-sample
    for k in range(K):
        X = np.random.uniform(0,true_theta,sample_size)
        theta_hat_n[k] = np.amax(X)

    u            = np.linspace(0,true_theta,sample_size,endpoint=True)
    pdf_theta_n  = sample_size/(true_theta^sample_size)*(u**(sample_size-1))
    bins = np.linspace(0, true_theta+0.5, 25)
    plt.hist(theta_hat_n, bins, alpha=0.5,density=True, label='sampled')
    plt.plot(u, pdf_theta_n,'o',label='true pdf theta_n')
    plt.legend(loc='upper right')
    plt.show()

def compare_param_nonparam_bootstrap(B,plug_in_estimate,T_boot,T_boot_p,est_std_err, est_std_err_p, conf_ints):

    bins = np.linspace(0,1.2,50)
    #Plot Histogram: boostrap theta_hat_n outcomes
    # Creates two subplots and unpacks the output array immediately
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    ax1.hist(T_boot, bins,density=True, label='sampled')
    #ax1.title("Bootstrapped theta_hat_n: est std_err:{0:f}".format(est_std_err))
    ax1.set_ylabel('#of occurences/{0:d})'.format(B))
    ax1.set_title('NonParametric Bootstrap theta_n')
    ax1.axvline(plug_in_estimate, color='k', linestyle='dashed', linewidth=1)
    _, max_ = plt.ylim()
    ax1.text(plug_in_estimate + plug_in_estimate/10, max_ - max_/10,'plug in estimate: {:.2f}'.format(plug_in_estimate))
    (l,r) = conf_ints[0]
    ax1.axvline(x=l, color = 'r')
    ax1.axvline(x=r, color = 'r')
    (l_piv,r_piv) = conf_ints[1]
    ax1.axvline(x=l_piv, color = 'b')
    ax1.axvline(x=r_piv, color = 'b')
    (l_per,r_per) = conf_ints[2]
    ax1.axvline(x=l_per, color = 'g')
    ax1.axvline(x=r_per, color = 'g')

    ax2.hist(T_boot_p, bins,density=True, label='sampled')
    ax2.set_ylabel('#of occurences/{0:d})'.format(B))
    ax2.set_title('Parametric Bootstrap theta_n')
    #plt.title("Bootstrapped theta_hat_n: est std_err:{0:f}".format(est_std_err_p))
    ax2.axvline(plug_in_estimate, color='k', linestyle='dashed', linewidth=1)
    _, max_ = plt.ylim()
    ax2.text(plug_in_estimate + plug_in_estimate/10, max_ - max_/10,'plug in estimate: {:.2f}'.format(plug_in_estimate))
    (cip_l, cip_r) = conf_ints[3]
    ax2.axvline(x=cip_l, color = 'y')
    ax2.axvline(x=cip_r, color = 'y')
    plt.show()
    time.sleep(5)


def main(arguments):
    from problem_7 import check_density
    true_theta   = 1
    iterations   = 100
    sample_size_ = [50]#,100,200,500]
    B_           = [10000]#[1000,5000,10000] # number of bootstrap samples to take
    for sample_size in sample_size_:
        for B in B_:
            coverage = np.zeros(4) #number true is in 1 parametric + 3 confidence intervals
            widths   = np.zeros(4)
            for j in range(iterations):
                #check_density(sample_size,100,true_theta)

                X = np.random.uniform(0,true_theta,sample_size)
                plug_in_estimate = np.amax(X)

                #NonParametric Bootstrap Sampling
                # uniformly sample, w/ replacement, from indeces (0,1,...sample_size)
                #  Use selected indeces to pull out data from data vector. Feed selected
                #  data through T, our statistic, in this case max{Xi*}, and store
                #  in T_boot
                indeces = np.arange(sample_size)
                T_boot  = np.zeros(B, dtype=float)
                for i in range(B):
                    boot_indeces = np.random.choice(indeces,sample_size,replace=True)
                    boot_sample  = X[boot_indeces]
                    T_boot[i]    = np.amax(boot_sample)
                est_std_err = np.std(T_boot)

                #Parametric Bootstrap
                # Draw n-sample from pdf(plug_in_estimate)
                T_boot_p  = np.zeros(B, dtype=float)
                delta_p   = np.zeros(B, dtype=float)
                for i in range(B):
                    boot_sample_p = np.random.uniform(0,plug_in_estimate,sample_size)
                    T_boot_p[i]   = np.amax(boot_sample_p)
                    delta_p[i]    = T_boot_p[i]-plug_in_estimate
                est_std_err_p = np.std(T_boot_p)

                #Confidence Intervals
                alpha = 0.05

                #Parametric Confidence Intervals
                p_025 = np.percentile(delta_p, alpha/2)
                p_95  = np.percentile(delta_p, 1-(alpha/2))
                cip_l = plug_in_estimate+p_025 #not sure if correct
                cip_r = plug_in_estimate-p_95
                #print("plug in estimate: {0:f}".format(plug_in_estimate))
                #print(".025 percentile:{0:f}, .95 percentile {1:f}".format(p_025,p_95))
                #print('Param 95% conf int: ({0:f},{1:f})'.format(cip_l, cip_r))
                if(true_theta>cip_l and true_theta<cip_r):
                     coverage[3]+=1
                     widths[3] += cip_r-cip_l

                #NonParametric Confidence Intervals
                #3 methods for 1-alpha = 95% confidence intervals:
                (l,r) = normal_95_confid_interval(plug_in_estimate, T_boot)
                print("Normal based 95% confid interval: ({0:f}, {1:f})".format(l,r))
                if(true_theta>l and true_theta<r):
                    coverage[0]+=1
                    widths[0]  +=r-l

                (l_piv,r_piv)=pivotal_confid_interval(plug_in_estimate, T_boot, alpha)
                print("Piv 95% confid interval: ({0:f}, {1:f})".format(l_piv,r_piv))
                if(true_theta>l_piv and true_theta<r_piv):
                    coverage[1]+=1
                    widths[1]  +=r_piv-l_piv
                # Percentile Interval
                (l_per,r_per)=percentile_confid_interval(plug_in_estimate,T_boot,alpha)
                print("Perc 95% confid interval: ({0:f}, {1:f})".format(l_per,r_per))
                if(true_theta>l_per and true_theta<r_per):
                    coverage[2]+=1
                    widths[2]  +=r_per-l_per
                conf_ints = [(l,r), (l_piv, r_piv), (l_per, r_per), (cip_l,cip_r)]
                #compare_param_nonparam_bootstrap(B,plug_in_estimate,T_boot,T_boot_p,est_std_err, est_std_err_p, conf_ints)
                print("{0:d}th iteration".format(j))

            print("n = {0:d}, B = {1:d}".format(sample_size,B))
            cov_prob = coverage/float(iterations)
            ave_widths = np.true_divide(widths,coverage)

            print("\t:   % correct, ave_width")
            print("\tNorm  : {0:f}, {1:f}".format(cov_prob[0],ave_widths[0]))
            print("\tPiv   : {0:f}, {1:f}".format(cov_prob[1],ave_widths[1]))
            print("\tPer   : {0:f}, {1:f}".format(cov_prob[2],ave_widths[2]))
            print("\tParam : {0:f}, {1:f}".format(cov_prob[3],ave_widths[3]))
            print("\n\n")

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
