import math
import numpy as np


np.random.seed(0)


def ActReLU (inputs):
        # Rectified Linear Unit Activation Function
        return np.maximum(0, inputs) # Returns input x if x > 0 else returns 0

def ActSoftMax (inputs):
    #Softmax Activation Function
    expVals = np.exp(inputs) - np.max(inputs, axis=1, keepdims=True)
    return expVals / np.sum(expVals, axis=1, keepdims=True) # Returns normalised probability distribution of inputs



class layer:
    def __init__(self, n_inputs, n_neurons, activationFnc='ReLU'):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1,n_neurons))
        self.aFnc = activationFnc
        
        self.aFncs = {
            'ReLU' : ActReLU,
            'SoftMax' : ActSoftMax   
        }
        
        
        
    def out (self, inputs):
        output = np.dot(inputs, self.weights) + self.biases # Compute sum of weighted inputs plus bias
        return (self.aFncs[self.aFnc](output)) # Return output passed through activation function
    
    
    def lossCCE (self, yPred, yTrue):
        yPredC = np.clip(yPred, 1e-7, 1-1e-7) # Clip to remove 'div by 0' flaw where yPred is 0
        
        # EG :
        # yPred = [
        #       [0.24, 0.63, 0.13],
        #       [0.17, 0.51, 0.32],
        #       [0.86, 0.12, 0.02]
        #     ]
        
        if (len(yTrue.shape) == 1):  # Case : Scalar (I.E: class 2 target = 2)
            Correct = yPredC[range(len(yPred)), yTrue]
            
            # EG :
            # yTrue = [0, 1, 1]
            # Correct = [0.24, 0.51, 0.12] (Returns predictions at index of each yTrue value)
            
        elif (len(yTrue.shape) == 2):  # Case : One Hot Encoding (I.E: class 2 target = [0,0,1,0])
            Correct = np.sum(yPredC*yTrue, axis=1)
            
            # EG : 
            # yTrue = [
            #       [1, 0, 0],
            #       [0, 1, 0],
            #       [0, 1, 0]
            #    ]
            
            # Correct multiplies the matricies yTrue and yPred to nullify 'non-predicted' values
            # I.E : (0.24 * 1) + (0.63 * 0) + (0.13 * 0) = 0.24
            
        return np.mean(-np.log(Correct))

            

class network:
    def __init__ (self, inputs, targets):        

        print(inputs)

        l1 = layer(2,3, 'ReLU')
        l2 = layer(3,3, 'SoftMax')
        
        l1Out = l1.out(inputs)
        l2Out = l2.out(l1Out)
        print(l1Out)
        print(l2Out)
        
        lossOut = l2.lossCCE(l2Out, targets)
        print(lossOut)
        
        