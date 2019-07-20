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

#Fit MPG from HP
def main(arguments):
    C = 28 # number of nondata lines in .dat file
    n = 82 # data points
    plot = True

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
        data[i,:] = a

    mpg     = np.zeros(n, dtype=float)
    hp      = np.zeros(n, dtype=float)
    mpg     = data[:,2]
    hp      = data[:,1]
    log_mpg = np.log(mpg)

    # raw data
    model     = LinearRegression()
    model.fit(hp[:,np.newaxis],mpg[:,np.newaxis])
    x = np.linspace(np.amin(hp)-50, np.amax(hp)+50,num=100)
    predict   = model.predict(x[:,np.newaxis])

    intercept = model.intercept_
    slope     = model.coef_
    R2       = model.score(hp[:,np.newaxis],mpg[:,np.newaxis])
    s = "mpg = {} * hp + {}\n R^2: {}".format(slope[0],intercept, R2)

    # log(mpg)
    model_2     = LinearRegression()
    model_2.fit(hp[:,np.newaxis],log_mpg[:,np.newaxis])
    predict_2   = model_2.predict(x[:,np.newaxis])
    predict_2_e = np.exp(predict_2)
    intercept_2 = model_2.intercept_
    slope_2     = model_2.coef_
    R2_2       = model_2.score(hp[:,np.newaxis],log_mpg[:,np.newaxis])
    s_2 = "log(mpg) = {} * hp  + {}\n mpg= {}*exp({}*hp) \n R^2: {}".format(slope_2[0],intercept_2,intercept_2, slope_2[0], R2_2)


    if(plot):
        f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
        ax1.plot(hp,mpg,'bo', x, predict,'r')
        ax1.set_xlabel('Horsepower')
        ax1.set_ylabel('MPG')
        ax1.set_title("HP vs MPG")
        ax1.text(0.3,0.7,s,fontsize=15,transform=ax1.transAxes)
        #ax1.text(0.8, 0.8, s, fontsize=12)

        ax2.plot(hp,log_mpg,'bo', x, predict_2,'r')
        ax2.plot(hp,mpg, 'bo', x, predict_2_e, 'r')
        ax2.set_xlabel('Horsepower')
        ax2.set_ylabel('LOG(MPG)')
        ax2.set_title("HP vs LOG(MPG)")
        ax2.text(0.3,0.5, s_2, fontsize=15, transform=ax2.transAxes)
        plt.show()

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
