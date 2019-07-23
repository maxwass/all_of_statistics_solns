from __future__ import print_function
import os
import time
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import unittest
from problem_7 import model_subset,run_regression,backward_search,forward_search,exhaustive_search,print_model_info,mallows_cp,subset_to_covar_names

def main(arguments):

    # run with: 'python3 test_p7.py'. If no errors thrown, working.



    data_ = [ [0,1,2], [0,0,1], [-1,-1,0], [1,1,0] ]
    data = np.asarray(data_)
    print(data)
    fieldnames = np.asarray(("a","b","c"))
    num_cov = data.shape[1]

    # multiple regression
    outcome = "c"
    covariates_mask  = np.ones(num_cov, dtype=bool)
    covariates_names = fieldnames #0th is model name
    covariates_mask[np.where(covariates_names == outcome)] = False
    outcome_mask    = np.logical_not(covariates_mask)
    covariates_names = covariates_names[covariates_mask]

    X = data[:,covariates_mask]
    Y = data[:,outcome_mask]
    print('Outcome Variable   : {0:s}'.format(outcome))
    print(Y)
    print('Covarites Variables: ',end=' '); print(covariates_names)
    print(X)
    n = X.shape[0]
    num_cov = X.shape[1]
    print('data points: {0:d}, covariates: {1:d}'.format(n,num_cov))

    #test each case for each algorithm
    # 3 cases: regress on a,b, or both
    full_model   = LinearRegression()
    full_model.fit(X,Y)
    predict = full_model.predict(X)
    R_tr    = n*mean_squared_error(Y, predict)
    var_est = R_tr * (1/(n-X.shape[1]))
    cov_used= X.shape[1]
    cp      = mallows_cp(R_tr, range(cov_used), var_est)
    print("Score: {0:.2f} = Train Error+2|S|var_est = {1:.2f}+2*{2:d}*{3:.2f}".format(cp,R_tr,cov_used,var_est))

    c_a_model = LinearRegression()
    a = X[:,[True,False]]
    c_a_model.fit(a,Y)
    predict = c_a_model.predict(a)
    R_tr    = n*mean_squared_error(Y, predict)
    cov_used= a.shape[1]
    cp      = mallows_cp(R_tr, range(cov_used), var_est)
    print("Score: {0:.2f} = Train Error+2|S|var_est = {1:.2f}+2*{2:d}*{3:.2f}".format(cp,R_tr,cov_used,var_est))

    c_b_model = LinearRegression()
    b = X[:,[False,True]]
    c_b_model.fit(b,Y)
    predict = c_b_model.predict(b)
    R_tr    = n*mean_squared_error(Y, predict)
    cov_used= b.shape[1]
    cp      = mallows_cp(R_tr, range(cov_used), var_est)
    print("Score: {0:.2f} = Train Error+2|S|var_est = {1:.2f}+2*{2:d}*{3:.2f}".format(cp,R_tr,cov_used,var_est))

    #the best model for mallows cp score is full model

    tc = unittest.TestCase('__init__')
    #exhaustive search
    #confirm exhaustive search finds that best model is ['a','b']
    covariates_names = fieldnames[covariates_mask]
    exh_model = exhaustive_search(Y,X,mallows_cp, var_est,covariates_names)
    out   = subset_to_covar_names(exh_model.covariate_subset, covariates_names)
    tc.assertTrue(set(out)==set(['a','b']))
    test_intercept= np.absolute(exh_model.params[0]-full_model.intercept_)
    test_coef     = np.absolute(exh_model.params[1][0]-full_model.coef_[0])
    tc.assertTrue(test_intercept<0.0001)
    tc.assertTrue(test_intercept<0.0001)
    # forward search: confirm forward search finds...
    #   1) best model with 2 cov is ['a','b']
    #   2) best model with 1 cov is ['b']
    num_cov_model   = 2
    fwd_model       = forward_search(Y,X,mallows_cp, var_est,covariates_names, num_cov_model)
    out   = subset_to_covar_names(fwd_model.covariate_subset, covariates_names)
    tc.assertTrue(set(out)==set(['a','b']))
    test_intercept= np.absolute(fwd_model.params[0]-full_model.intercept_)
    test_coef     = np.absolute(fwd_model.params[1][0]-full_model.coef_[0])
    tc.assertTrue(test_intercept<0.0001)
    tc.assertTrue(test_intercept<0.0001)

    num_cov_model = 1
    fwd_model     = forward_search(Y,X,mallows_cp, var_est,covariates_names, num_cov_model)
    out           = subset_to_covar_names(fwd_model.covariate_subset, covariates_names)
    tc.assertTrue(set(out)==set(['b']))
    test_intercept= np.absolute(fwd_model.params[0]-c_b_model.intercept_)
    test_coef     = np.absolute(fwd_model.params[1][0]-c_b_model.coef_[0])
    tc.assertTrue(test_intercept<0.0001)
    tc.assertTrue(test_intercept<0.0001)

    # backward search: confirm forward search finds...
    #   1) best model with >=2 cov is ['a','b']
    #   2) best model with >=1 cov is ['a', b']
   # backward search
    num_cov_model = 2
    bwd_model     = backward_search(Y,X,mallows_cp, var_est,covariates_names, num_cov_model)
    out           = subset_to_covar_names(bwd_model.covariate_subset, covariates_names)
    tc.assertTrue(set(out)==set(['a','b']))
    test_intercept= np.absolute(bwd_model.params[0]-full_model.intercept_)
    test_coef     = np.absolute(bwd_model.params[1][0]-full_model.coef_[0])
    tc.assertTrue(test_intercept<0.0001)
    tc.assertTrue(test_intercept<0.0001)



    num_cov_model = 1
    bwd_model     = backward_search(Y,X,mallows_cp, var_est,covariates_names, num_cov_model)
    out           = subset_to_covar_names(bwd_model.covariate_subset, covariates_names)
    tc.assertTrue(set(out)==set(['a','b']))
    test_intercept= np.absolute(bwd_model.params[0]-full_model.intercept_)
    test_coef     = np.absolute(bwd_model.params[1][0]-full_model.coef_[0])
    tc.assertTrue(test_intercept<0.0001)
    tc.assertTrue(test_intercept<0.0001)



if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
