from __future__ import print_function
import os
import time
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import chi2
import csv
import itertools
from collections import OrderedDict
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def main(arguments):
    n = 209 # data points
    plot = True

    # Read in Raw txt file and clean it up
    # Note: manually deleted one line between feildnames and data in txt file
    path = '/Users/maxwasserman/Desktop/all_of_statistics/chapter_16/montana.dat'
    path_cleaned = '/Users/maxwasserman/Desktop/all_of_statistics/chapter_16/montana_cleaned.dat'

    f_clean = open(path_cleaned,'r+')
    # bring file pointer back to beginning
    #f_clean.seek(0, 0)

    fieldnames = ("AGE", "SEX", "INC", "POL", "AREA", "FIN", "STAT")
    data = np.zeros((n,len(fieldnames)-1))
    #reader = csv.DictReader(itertools.islice(f_clean, 0, None))#, delimiter=' ')
    reader = csv.DictReader(f_clean, delimiter='\t')
    reader.fieldnames = fieldnames #"MAKE","MODEL","VOL","HP","MPG","SP","WT"
    age, fin = [], [] #np.zeros(n), np.zeros(n)
    for i,row in enumerate(reader):
        a = np.array(list(row.values()))
        #print('{0:d}: '.format(i), end=' '); print(a)
        #data[i,:] = a.astype(float)
        if(a[0]=='*' or a[5]=='*'):
            print('{0:d} row has *'.format(i))
            continue
        age.append(float(a[0]))
        fin.append(float(a[5]))
        #age[i], fin[i] = a[0].astype(float), a[5].astype(float)
        print('{0:d}: age: {1:.0f}, financial status: {2:.0f}'.format(i,float(a[0]),float(a[5])))

    #age,sex,inc,pol,area,fin,stat = data[:,0],data[:,1],data[:,2],data[:,3],data[:,4],data[:,5],data[:,6]
    age_ = np.array(age)
    fin_  = np.array(fin)
    print(age_.shape)
    print(age_)
    print(fin_.shape)
    print(fin_)

    # AGE is a RV: {1,2,3}
    # FIN is a RV: {1,2,3}
    # We now must create counts X(i,j)
    #   where X(i,j) := the number of occurences where AGE=i and FIN=j
    X = np.zeros((3,3))
    for j in range(len(age_)):
        age_index = int(age_[j])-1 #shift index
        fin_index = int(fin_[j])-1
        X[age_index,fin_index] += 1
        #print('{0:d}: age {1:.0f}  fin {2:.0f}'.format(j,age_[j],fin_[j]))
        #print('index: (age,fin): ( {0:.0f}, {1:.0f} )'.format(age_index, fin_index))
        #print(X)
        #time.sleep(5)
    print(X)
    N = len(age_)
    #sum across rows
    X_age = np.sum(X, axis=1)
    X_fin = np.sum(X, axis=0)

    T = 0.0
    U = 0.0
    for age_val in range(1,3+1,1):
        for fin_val in range(1,3+1,1):
            print('age: {0:d}, financial status: {1:d}'.format(age_val,fin_val))
            age_index = age_val-1
            fin_index = fin_val-1
            X_ij = X[age_index,fin_index]
            X_i  = X_age[age_index]
            X_j  = X_fin[fin_index]
            T += X_ij * np.log( X_ij*N / (X_i*X_j) )

            E_ij = X_age[age_index]*X_fin[fin_index]/N #expected # counts under Independence assumption
            U += ((X_ij-E_ij)**2) / E_ij
    T = 2*T
    print('chi-squared df = (I-1)(J-1) = 4')
    df = 4
    #stats.chisquare(f_obs, f_exp=[], ddof=k)
    p_value_T = 1 - chi2.cdf(T, df)
    print('Test Statistic T = {0:.2f} -> p_value = {1:.7f}'.format(T,p_value_T))
    p_value_U = 1 - chi2.cdf(U, df)
    print('Test Statistic U = {0:.2f} -> p_value = {1:.7f}'.format(U,p_value_U))

    #Both p-values are very small -> strong evidence that random variables are
    #  NOT INDEP => AGE and FINANCIAL STATUS are ASSOCIATED


    if(plot):
        fig, ax = plt.subplots(1, 1)
        x = np.linspace(chi2.ppf(0.01, df), chi2.ppf(0.9999999, df), 100)
        ax.plot(x, chi2.pdf(x, df), 'r-', lw=5, alpha=0.6, label='chi2 pdf')
        ax.set_ylim([0,.2])
        ax.set_ylabel('Chi-Squared PDF, DOF= {0:d}'.format(df))
        ax.set_xlabel('Outcome of Statistic')
        ax.set_title("Independence Testing")
        plt.axvline(x=T)
        plt.axvline(x=U, color = 'red')
        plt.show()

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
