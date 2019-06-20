from __future__ import print_function
import os
import time
import numpy as np
from scipy import stats

def normal_95_confid_interval(plug_in_estimate, T_boot):
    est_std_err = np.std(T_boot)
    left_normal  = plug_in_estimate-2*est_std_err
    right_normal = plug_in_estimate+2*est_std_err
    return (left_normal, right_normal)

def pivotal_confid_interval(plug_in_estimate, T_boot, alpha):
    left_piv  = 2*plug_in_estimate-np.quantile(T_boot, 1 - (alpha/2))
    right_piv = 2*plug_in_estimate-np.quantile(T_boot, alpha/2)
    return (left_piv,right_piv)

def percentile_confid_interval(plug_in_estimate, T_boot, alpha):
    left_percentile  = np.quantile(T_boot, alpha/2)
    right_percentile = np.quantile(T_boot, 1 - (alpha/2))
    return (left_percentile,right_percentile)
