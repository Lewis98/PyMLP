import matplotlib.pyplot as plt
import numpy as np
import random
import csv

np.random.seed(0)

import perceptron as p

import neuralNet as n


def create_DS(points, classes):
    X = np.zeros((points*classes, 2))
    Y = np.zeros(points*classes, dtype='uint8')
    
    for class_no in range(classes):
        ix = range(points*class_no, points*(class_no+1))
        r = np.linspace(0, 1, points)
        t = np.linspace(class_no*4, (class_no+1)*4, points) + np.random.randn(points)*0.2
        X[ix] = np.c_[r*np.sin(t*2.5), r*np.cos(t*2.5)]
        Y[ix] = class_no
    return X, Y



# https://www.ais.uni-bonn.de/download/datasets.html

if __name__ == "__main__":
 
    X,Y = create_DS(100, 3)
    
    N = n.network(X, Y)
    
    X_x = []
    X_y = []
    Y = []
    for i in range(1000):
        X_x.append(random.randint(1,500))
        X_y.append(random.randint(1,500))

        if X_x[i] + 5 > X_y[i]:
            Y.append(1)
        else:
            Y.append(-1)

    X = [X_x, X_y]

    #P = p.Perceptron()
    
    #P.train(X, Y)

#
#    
#
#    input("Press Enter to continue . . .")