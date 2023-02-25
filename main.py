import matplotlib.pyplot as plt
import random
import csv


import perceptron as p

# https://www.ais.uni-bonn.de/download/datasets.html

if __name__ == "__main__":
 
    
    X_x = []
    X_y = []
    Y = []
    for i in range(1000):
        X_x.append(random.randint(1,500))
        X_y.append(random.randint(1,500))

        if X_x[i] > X_y[i]:
            Y.append(1)
        else:
            Y.append(-1)

    X = [X_x, X_y]

    P = p.Perceptron()
    
    P.train(X, Y)

#
#    
#
#    input("Press Enter to continue . . .")