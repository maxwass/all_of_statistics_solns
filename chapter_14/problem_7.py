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
    log_mpg = np.log(mpg)

    #number of covariates:
    num_cov = data.shape[1]

    #test combinations function
    S = range(num_cov)
    '''
    for subset in itertools.combinations(S,2):
        print(subset)
    '''
    #combinations funciton will return all size r subsets of input iterable
    print([0,1,2,3,4])
    print(fieldnames[1:])
    print('input set size: {0:d}'.format(num_cov))
    i = 1
    for r in range(num_cov+1):
        print('subset size: {0:d}'.format(r))
        for subset in itertools.combinations(S,r):
            print('\t{0:d}: '.format(i), end =" ")
            print(subset)
            i=i+1

    # multiple regression
                        #VOL	HP    MPG    SP    WT
    select_covariates = [True, True, False, True, True]
    select_outcome    = np.logical_not(select_covariates)

    covariates = data[:,select_covariates]
    outcome    = data[:,select_outcome]

    multi_model   = LinearRegression()
    multi_model.fit(covariates,outcome)
    multi_predict = model.predict(covariates)

    #training error = n * mse
    R_tr = n*mean_squared_error(outcome, multi_predict)
    print("Training Error: {0:f}".format(R_tr))



if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
