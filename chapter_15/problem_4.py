from __future__ import print_function
import os, sys
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# write function to generate observations from a multinomial(n,p) distribution
def multinom_sample(n,p):
    ball_colors   = len(p)
    elements      = range(0,ball_colors,1) # 0,1,...,ball_colors-1
    probabilities = p
    draws = np.random.choice(elements, n, p=probabilities)

    if(False):
        print('prob vector  : ', end=' '); print(p)
        print('num draws n  : {0:d}'.format(ball_colors))
        print('len(p): {0:d}'.format(ball_colors))
        print('draws....')
    print(draws)

def main(arguments):

    multinom_sample(5,[0.25,.25,0.5])


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
