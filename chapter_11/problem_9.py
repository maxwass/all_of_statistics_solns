from __future__ import print_function
import os
import time
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

#Hypothesis Testing
#not implimented: odd ratio or FDR

def main(arguments):
    #Hardcoded Data
    placebo = np.array([80,45])
    chloro  = np.array([75,26])
    dimen   = np.array([85,52])
    pen_100 = np.array([67,35])
    pen_150 = np.array([85,37])

    #model: X1,...,Xn ~ Bern(p) where X = 1 if nasaeus
    # H0: d := p - p_placebo = 0 (no effect)
    # H1: d != 0
    # Define Statistics T = ( d_hat - 0)/ste,where ste is using mle bern
    # Use CLT T ~ N(0,1) under H0 thus p_value = P(|T|>T_obs) = 2*std_norm_cdf(-|T_obs|)

    #plug in estimator for difference of means
    placebo_mle = placebo[1]/placebo[0]
    chloro_mle  = chloro[1]/chloro[0]
    dimen_mle   = dimen[1]/dimen[0]
    pen_100_mle = pen_100[1]/pen_100[0]
    pen_150_mle = pen_150[1]/pen_150[0]

    placebo_var = placebo_mle*(1-placebo_mle)/placebo[0]
    chloro_ste  = np.sqrt(chloro_mle*(1-chloro_mle)/chloro[0]    + placebo_var)
    dimen_ste   = np.sqrt(dimen_mle*(1-dimen_mle)/dimen[0]       + placebo_var)
    pen_100_ste = np.sqrt(pen_100_mle*(1-pen_100_mle)/pen_100[0] + placebo_var)
    pen_150_ste = np.sqrt(pen_150_mle*(1-pen_150_mle)/pen_150[0] + placebo_var)

    d_chloro     = chloro_mle-placebo_mle
    d_chloro_z   = d_chloro/chloro_ste
    d_chloro_p_val = 2*stats.norm.cdf(-np.absolute(d_chloro_z))
    print("chloro: d_mle: {0:f}, z_val {1:f}, p_val: {2:f}".format(d_chloro, d_chloro_z, d_chloro_p_val))

    d_dimen       = dimen_mle-placebo_mle
    d_dimen_z     = d_dimen/dimen_ste
    d_dimen_p_val = 2*stats.norm.cdf(-np.absolute(d_dimen_z))
    print("dimen:  d_mle: {0:f}, z_val {1:f}, p_val: {2:f}".format(d_dimen, d_dimen_z, d_dimen_p_val))

    d_pen_100       = pen_100_mle-placebo_mle
    d_pen_100_z     = d_pen_100/pen_100_ste
    d_pen_100_p_val = 2*stats.norm.cdf(-np.absolute(d_pen_100_z))
    print("pen_100:  d_mle: {0:f}, z_val {1:f}, p_val: {2:f}".format(d_pen_100, d_pen_100_z, d_pen_100_p_val))


    d_pen_150       = pen_150_mle-placebo_mle
    d_pen_150_z     = d_pen_150/pen_150_ste
    d_pen_150_p_val = 2*stats.norm.cdf(-np.absolute(d_pen_150_z))
    print("pen_150:  d_mle: {0:f}, z_val {1:f}, p_val: {2:f}".format(d_pen_150, d_pen_150_z, d_pen_150_p_val))



    #Bonferoni Method (multiple testing adjustment)
    m = 4
    print("p_value <? bonferoni adjusted threshold (alpha/m) {0:f}/{1:d}={2:f}".format(0.05,m,0.05/m))
    print("Chloro:   {0:f} <? {1:f}".format(d_chloro_p_val, 0.05/m))
    print("Dimen:    {0:f} <? {1:f}".format(d_dimen_p_val, 0.05/m))
    print("Pen_100:  {0:f} <? {1:f}".format(d_pen_100_p_val, 0.05/m))
    print("Pen_150:  {0:f} <? {1:f}".format(d_pen_150_p_val, 0.05/m))

    print("Interpretation: Under single hypothesis testing, the only treatment which has enough evidence to reject hypotheses that it is the same as the placebo is the Chlorpromazine (p = 0.005703 < 0.05). This is the same as when corrected by Bonferoni Method (Chlor is only rejected null)")



if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
