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
from sklearn.linear_model import LinearRegression
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
    debug = False
    if(debug):
        print('Backward Search for model with >= {0:d} parameters'.format(num_cov_output))
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
        time.sleep(3)
        improve = False
        for md_info in potential_models:
            if(md_info.score < optimal_model.score):
                optimal_model = md_info
                improve       = True
                print('better model found')
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
    optimal_score, optimal_params, optimal_subset = np.inf, np.nan, set()
    model          = LinearRegression()
    debug = True
    if(debug):
        print('Backward Search for model with <= {0:d} parameters'.format(num_cov_output))
        print('Full model has {0:d} parameters'.format(num_cov))

    forward_subset, remain_cov = set(), set(range(num_cov))

    for j in range(1,num_cov_output+1,1):
        if(debug):
            print('\tcurrent forward subset  : ', end=' '); print(forward_subset,end=' ')
            print('| current remaining subset: ', end=' '); print(remain_cov)
        scores = []
        #see which of the remaining covariates, when added, produces the smallest score
        for cov in remain_cov:
            temp_subset = set.union(forward_subset,{cov})
            #pull out columns from data matrix corresponding to subset
            covariate_data_subset = get_subset_cov(temp_subset,num_cov,X)
            model.fit(covariate_data_subset,Y)
            predict = model.predict(covariate_data_subset)
            R_tr    = n*mean_squared_error(Y, predict)
            scores.append([cov,score_func(R_tr, temp_subset, var_est)])

        #find which of these produced lowest score
        min_score , min_score_cov = np.inf, -1
        for s in scores:
            if(s[1]<min_score):
                min_score     = s[1]
                min_score_cov = s[0]
        #pretty output showing scores for each remaining cov
        print('\t', end=' ')
        for s in scores:
            cov, cov_name, cov_score  = s[0], covariates_names[s[0]], s[1]
            if(cov==min_score_cov):
                print('|* {0:s}: {1:.2f} *|'.format(cov_name,cov_score), end=' ')
            else:
                print('| {0:s}: {1:.2f} |'.format(cov_name,cov_score), end=' ')
        #update if better model found
        if(min_score<optimal_score):
            print('\n\t\t* found new best score* ', end=' ')
            forward_subset.add(min_score_cov)
            remain_cov.remove(min_score_cov)
            print('| forward subset->', end=' '); print(forward_subset, end=' ')
            print('| remain subset-> : ', end=' '); print(remain_cov)
            optimal_score = min_score
            optimal_params = (model.intercept_, model.coef_)
            #may need to sort and make array
            optimal_subset = forward_subset
            print('\n\n')
        else:
            print('score did not improve, returning best found thus far')
            return optimal_score, optimal_params, optimal_subset
        #time.sleep(5)

def exhaustive_search(Y,X,score_func, var_est, covariates_names):
    num_cov, n     = X.shape[1], X.shape[0]
    optimal_score  = np.inf
    optimal_params = np.nan
    optimal_subset = ()
    model          = LinearRegression()
    debug = False
    if(debug):
        print('input set size: {0:d}'.format(num_cov))
    for r in range(1,num_cov+1,1):
        if(debug):
            print('subset size: {0:d}'.format(r))
        for subset in itertools.combinations(range(num_cov),r):
            #pull out columns from data matrix corresponding to subset
            #convert subset into logical array
            covariate_subset = get_subset_cov(subset,num_cov,X)
            model.fit(covariate_subset,Y)
            predict = model.predict(covariate_subset)
            R_tr    = n*mean_squared_error(Y, predict)
            score = score_func(R_tr, subset, var_est)
            if(debug):
                print('\t{0:d}: '.format(i), end =" ")
                print(subset_to_covar_names(subset,covariates_names))
                print('\t\tscore: {0:f}'.format(score))
            if(score<optimal_score):
                optimal_score = score
                optimal_subset = subset
                optimal_params = (model.intercept_, model.coef_)
                if(debug):
                    print('\t\t^^new best subset found')

    return optimal_score, optimal_params, optimal_subset

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
    covariate_names_ = np.asarray(covariate_names)
    subset_          = np.asarray(list(subset))
    return covariate_names[subset_]
'''
def print_search_result(score,covar_names,parameters):
    print('\tscore: {0:f}'.format(score))
    print('\tparameters: ', end=' ')
    print(parameters)
    print('\tsubset: ', end=' ')
    print(covar_names)
'''
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

    f_clean = open(path_cleaned,'r+')
    ''' #only do this 1 time to clean data file
    f       = open(path,'r')
    # skip non data lines
    for _ in range(C):
        print(f.readline())

    print('SKIPPED NON DATA LINES')
    f_clean.seek(0, 0)

    #read header and data lines
    for i,line in enumerate(f):
        print(str(i)+': '+','.join(line.split())+'\n')
        f_clean.write(','.join(line.split()) + '\n')

    print('\n\n\n\n\n')
    time.sleep(2)
    '''

    # bring file pointer back to beginning
    f_clean.seek(0, 0)

    data = np.zeros((n,5))
    fieldnames = ("MAKE/MODEL","VOL","HP","MPG","SP","WT")
    reader = csv.DictReader(itertools.islice(f_clean, 0, None), delimiter=',')
    reader.fieldnames = fieldnames #"MAKE","MODEL","VOL","HP","MPG","SP","WT"

    for i,row in enumerate(reader):
        a = np.array(list(row.values()))[1:]
        data[i,:] = a.astype(float)


    vol = data[:,0]
    hp  = data[:,1]
    mpg = data[:,2]
    sp  = data[:,3]
    wt  = data[:,4]

    #number of covariates:
    num_cov = data.shape[1]

    #test combinations function
    '''
    for subset in itertools.combinations(S,2):
        print(subset)
    '''
    #combinations funciton will return all size r subsets of input iterable
    '''
    toy_data = np.array([ [0,1,2,3,4],[0,1,2,3,4],[0,1,2,3,4] ])
    print('toy_data')
    print(toy_data)
    print(toy_data.shape)
    print(data.shape)
    #print([0,1,2,3,4])
    #print(fieldnames[1:])
    print('input set size: {0:d}'.format(num_cov))
    i = 1
    for r in range(num_cov+1):
        print('subset size: {0:d}'.format(r))
        for subset in itertools.combinations(range(num_cov),r):
            print('\t{0:d}: '.format(i), end =" ")
            print(subset)
            i=i+1
            #pull out columns from data matrix corresponding to subset
            #convert subset into logical array
            covariate_subset = get_subset_cov(subset,num_cov,toy_data)
            print(covariate_subset)
    '''
    # multiple regression

    #VOL    HP    MPG    SP    WT
    outcome = "SP"
    covariates_mask  = np.ones(num_cov, dtype=bool)
    covariates_names = np.asarray(fieldnames[1:]) #0th is model name
    covariates_mask[np.where(covariates_names == outcome)] = False
    outcome_mask    = np.logical_not(covariates_mask)
    covariates_names = covariates_names[covariates_mask]

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
    print("Training Error:  {0:f}".format(R_tr))
    print('variance est:    {0:f}'.format(var_est))
    print('mallows cp stat: {0:f}'.format(cp))

    # exhaustive search
    '''
    exh_score, exh_params, exh_subset = exhaustive_search(Y,X, mallows_cp, var_est,covariates_names)
    exh_covar_names               = subset_to_covar_names(exh_subset,covariates_names)
    print('Exhaustive Search:')
    print_search_result(exh_score, exh_covar_names, exh_params)
    '''
    # forward search
    #TODO copy and past backward_search to edit relevent sections
    num_cov_model = 2
    '''
    fwd_score, fwd_params, fwd_subset = forward_search(Y,X,mallows_cp, var_est, covariates_names, num_cov_model)
    fwd_covar_names  = subset_to_covar_names(exh_subset,covariates_names)
    print('Forward Search')
    print_search_result(fwd_score, fwd_covar_names, fwd_params)
    '''

    # backward search
    bwd_model = backward_search(Y,X,mallows_cp, var_est,covariates_names, num_cov_model)
    print_model_info(bwd_model, covariates_names)



if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
