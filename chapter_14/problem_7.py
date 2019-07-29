from __future__ import print_function
import os
import time
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import csv
import itertools
from collections import OrderedDict
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import mean_squared_error
#Fit MPG from HP
class model_subset():
    def __init__(self,score,params,covariate_changed,covariate_subset):
        self.score             = score
        self.covariate_changed = covariate_changed
        self.covariate_subset  = covariate_subset
        self.params            = params

def run_regression(X,Y,score_func, var_est,num_cov,cov_subset, cov_changed):
    covariate_data_subset = get_subset_cov(cov_subset,num_cov,X)
    model       = LinearRegression()
    model.fit(covariate_data_subset,Y)
    predict     = model.predict(covariate_data_subset)
    R_tr        = X.shape[0]*mean_squared_error(Y, predict)
    score       = score_func(R_tr, cov_subset, var_est)
    params      = [model.intercept_, model.coef_]
    model_info  = model_subset(score,params,cov_changed,cov_subset)
    return model_info

#We are outputting a model whose parameters are a subset of full model
 # parameters. Thus must specify how many params one wants in the out-
 # putted model. We continue until score stops decreasing OR reach #
 # covariates specified
def backward_search(Y,X,score_func, var_est, covariates_names, num_cov_output):
    num_cov, n     = X.shape[1], X.shape[0]
    all_cov = set()
    all_cov.update(range(num_cov))
    optimal_model = run_regression(X,Y,score_func,var_est,num_cov,all_cov,np.nan)
    if(num_cov_output>num_cov):
        print("less total covariates {0:d} than amount requested {1:d}".format(num_cov,num_cov_output))
        return optimal_model
    debug = True
    if(debug):
        print('**BACKWARD SEARCH for model with >= {0:d} parameters'.format(num_cov_output))
        print('Full model has {0:d} parameters'.format(num_cov))

    for subset_size in range(num_cov-1,num_cov_output-1,-1):
        print('SEARCHING SUBSET SIZE: {0:d}'.format(subset_size))
        if(debug):
            print('\ncurrent best subset:')
            print_model_info(optimal_model,covariates_names)
        potential_models = []
        #see which of the covariates, when removed, produces smallest score
        for cov in optimal_model.covariate_subset:
            temp_subset = optimal_model.covariate_subset.copy()
            temp_subset.remove(cov)
            #print('regression on subsets')
            #print(temp_subset)
            md_info     = run_regression(X,Y,score_func,var_est,num_cov,temp_subset,cov)
            potential_models.append(md_info)
        #pretty output showing scores for each remaining cov
        print('\t', end=' ')
        for md_info in potential_models:
            cov_name = covariates_names[md_info.covariate_changed]
            print('| {0:s}: {1:.2f} |'.format(cov_name,md_info.score), end=' ')
        print('\n')
        improve = False
        for md_info in potential_models:
            if(md_info.score < optimal_model.score):
                optimal_model = md_info
                improve       = True
                print('\t->better model found')
                print_model_info(optimal_model,covariates_names)
        if(improve==False):
            print('score did not improve, returning best found thus far')
            return optimal_model
    return optimal_model
#We are outputting a model whose parameters are a subset of full model
# parameters. Thus must specify how many params one wants in the out-
# putted model. We continue until score stops decreasing OR reach #
# covariates specified
def forward_search(Y,X,score_func, var_est, covariates_names, num_cov_output):
    num_cov, n     = X.shape[1], X.shape[0]
    #initialize optimal model to be empty set
    unused_covariates = set()
    unused_covariates.update(range(num_cov))
    optimal_model = model_subset(np.inf,np.nan,np.nan,set())
    if(num_cov_output<1):
        print('requested {0:d} covariates. Must have >= 1.'.format(num_cov_output))
    if(num_cov_output>num_cov):
        print("total cov's {0:d} < cov's requested {1:d}".format(num_cov,num_cov_output))
        print("setting requested covs to total covs: {0:d}->{1:d}".format(num_cov_output,num_cov))
        num_cov_output = num_cov
    debug = True
    if(debug):
        print('***FORWARD SEARCH for model with <= {0:d} parameters'.format(num_cov_output))
        print('Full model has {0:d} parameters'.format(num_cov))

    print('- - - - - - - - - - - - - - - - - - ')
    for subset_size in range(1,num_cov_output+1,1):
        print('SEARCHING SUBSET SIZE: {0:d}'.format(subset_size))
        if(debug):
            print('\ncurrent best subset:')
            print_model_info(optimal_model,covariates_names)
            print(' current unused covs:')
            print(unused_covariates)
        potential_models = []
        #see which of the covariates, when removed, produces smallest score
        for cov in unused_covariates:
            temp_subset = optimal_model.covariate_subset.copy()
            temp_subset.add(cov)
            #print('regression on subsets')
            #print(temp_subset)
            md_info     = run_regression(X,Y,score_func,var_est,num_cov,temp_subset,cov)
            potential_models.append(md_info)
        #pretty output showing scores for each remaining cov
        print('\t', end=' ')
        for md_info in potential_models:
            cov_name = covariates_names[md_info.covariate_changed]
            print('| {0:s}: {1:.2f} |'.format(cov_name,md_info.score), end=' ')
        print('\n')
        improve = False
        for md_info in potential_models:
            if(md_info.score < optimal_model.score):
                optimal_model = md_info
                improve       = True
                print('\t->better model found')
                print_model_info(optimal_model,covariates_names)
        if(improve==False):
            print('score did not improve, returning best found thus far')
            print('- - - - - - - - - - - - - - - - - - ')
            return optimal_model
        #update unused covariates
        all_cov = set()
        all_cov.update(range(num_cov))
        unused_covariates = all_cov.difference(optimal_model.covariate_subset)
        print('- - - - - - - - - - - - - - - - - - ')
    return optimal_model

def exhaustive_search(Y,X,score_func, var_est, covariates_names):
    num_cov, n     = X.shape[1], X.shape[0]
    #initialize optimal model to be empty set
    optimal_model = model_subset(np.inf,np.nan,np.nan,set())
    debug = True
    if(debug):
        print('***EXHAUSTIVE SEARCH for model w/ ({0:d}, {1:d}) params'.format(0,num_cov))
    for subset_size in range(1,num_cov+1,1):
        for md in itertools.combinations(range(num_cov),subset_size):
            temp_subset = set()
            temp_subset.update(md)
            print(temp_subset)
            md_info     = run_regression(X,Y,score_func,var_est,num_cov,temp_subset,np.nan)
            print_model_info(md_info,covariates_names)
            if(md_info.score < optimal_model.score):
                optimal_model = md_info
                improve       = True
                print('\t->better model found')
                print_model_info(optimal_model,covariates_names)

    return optimal_model

def mallows_cp(R_tr, subset, var_est):
    return R_tr + 2*len(subset)*var_est

#data already has the outcome variable column removed
# subset indeces must correspond to order in data
# 0-Vol, 1-HP, 2-SP, 3-WT
def get_subset_cov(subset,size_full_set, data):
    bool_select_cov = np.zeros(size_full_set, dtype=bool)
    for j in subset:
        bool_select_cov[j] = True
    covariate_subset = data[:,bool_select_cov]
    return covariate_subset

#subset can be set(), array, or tuple
def subset_to_covar_names(subset,covariate_names):
    if(subset==None):
        return['']
    if(len(subset)==0):
        return []
    covariate_names_ = np.asarray(covariate_names)
    subset_          = np.asarray(list(subset))
    return covariate_names[subset_]

def print_model_info(model_subset,covariate_names):
    print('\tscore: {0:f}'.format(model_subset.score))
    print('\tsubset: ', end=' ')
    subset = model_subset.covariate_subset
    print(subset,end=' '); print('<==>',end=' ')
    print(subset_to_covar_names(subset,covariate_names))
    #print('\tparameters: ', end=' ')
    #print(model_subset.params)

def main(arguments):
    C = 28 # number of nondata lines in .dat file
    n = 82 # data points
    plot = False

    # Read in Raw txt file and clean it up
    # Note: manually deleted one line between feildnames and data in txt file
    path = '/Users/maxwasserman/Desktop/all_of_statistics/chapter_14/carmileage.txt'
    path_cleaned = '/Users/maxwasserman/Desktop/all_of_statistics/chapter_14/carmileage_cleaned.dat'

    # open file and bring file pointer back to beginning
    f_clean = open(path_cleaned,'r+')
    f_clean.seek(0, 0)

    data = np.zeros((n,5))
    fieldnames = ("MAKE/MODEL","VOL","HP","MPG","SP","WT")
    reader = csv.DictReader(itertools.islice(f_clean, 0, None), delimiter=',')
    reader.fieldnames = fieldnames #"MAKE","MODEL","VOL","HP","MPG","SP","WT"

    for i,row in enumerate(reader):
        a = np.array(list(row.values()))[1:]
        data[i,:] = a.astype(float)

    #number of covariates:
    num_cov = data.shape[1]

    # multiple regression

    #VOL    HP    MPG    SP    WT
    outcome = "MPG"
    covariates_mask  = np.ones(num_cov, dtype=bool)
    covariates_names = np.asarray(fieldnames[1:]) #0th is model name
    covariates_mask[np.where(covariates_names == outcome)] = False
    outcome_mask    = np.logical_not(covariates_mask)
    covariates_names = covariates_names[covariates_mask]
    print('Outcome Variable   : {0:s}'.format(outcome))
    print('Covarites Variables: ',end=' '); print(covariates_names)

    X = data[:,covariates_mask]
    Y = data[:,outcome_mask]

    # full model
    full_model   = LinearRegression()
    full_model.fit(X,Y)
    full_predict = full_model.predict(X)

    #training error = n * mse
    # Residual: epsilon_i = Y_i - Y_predict_i
    # Residual Sum of Squares: RSS = sum{i} epsilon_i^2
    # Mean Square Error: mse = (1/n) *  sum{i} (Y_i-Y_predict_i)^2
    # Thus RSS = n * mse
    # unbiased estimate of sigma^2 = (1/(n-k)) * sum{i} epsilon_i^2
    #                              = (1/(n-k)) * (n * mse)
    R_tr              = n*mean_squared_error(Y, full_predict)
    var_est  = R_tr * (1/(n-X.shape[1]))
    cp = mallows_cp(R_tr, range(num_cov), var_est)
    #print("Training Error:  {0:f}".format(R_tr))
    #print('variance est:    {0:f}'.format(var_est))
    #print('mallows cp stat: {0:f}'.format(cp))

    # exhaustive search
    exh_model       = exhaustive_search(Y,X,mallows_cp, var_est,covariates_names)
    print('\nExhaustive Search:')
    print_model_info(exh_model, covariates_names)
    print('\n\n\n')

    num_cov_model   = 2
    # forward search
    #TODO copy and past backward_search to edit relevent sections
    fwd_model       = forward_search(Y,X,mallows_cp, var_est,covariates_names, num_cov_model)
    print('\nForward Search Result:')
    print_model_info(fwd_model, covariates_names)
    print('\n\n\n')

    # backward search
    bwd_model = backward_search(Y,X,mallows_cp, var_est,covariates_names, num_cov_model)
    print('\nBackward Search Result:')
    print_model_info(bwd_model, covariates_names)
    print('\n\n\n')

    clf = Lasso(alpha=0.5)
    clf.fit(X,Y)
    print(clf.intercept_)
    print(clf.coef_)
    mse  = mean_squared_error(Y, clf.predict(X))
    R_tr = X.shape[0]*mse
    print("MSE: {0:.2f}, Training Error: {1:.2f}".format(mse, R_tr))
    #score       = score_func(R_tr, cov_subset, var_est)

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
