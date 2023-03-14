import sys


class Perceptron:
    def __init__(self, numInputs=0, bias=0):
        self.numInputs = numInputs
        self.weights = [.5] * self.numInputs
        self.output = 0
        self.learnRate = .1 # Default learning rate
        self.bias = bias

    
    def Activation(self, i):
        if (i >= 0):
            return 1
        else:
            return -1
    
    
    def predict(self, inputsArr):
        
        outArr = []
        for i in len(inputsArr):
            outArr.append(self.calculateOut(inputsArr[i]))
        
        return outArr
                

    def calculateOut(self, inputs):
        if len(inputs) != self.numInputs:
            errMsg = 'Invalid number of inputs: {} provided | {} required'.format(len(inputs),self.numInputs)
            raise Exception(errMsg)

        for i in range(len(inputs)):
            self.output += inputs[i] * self.weights[i]
        self.output += self.bias
        self.output = self.Activation(self.output)
        return self.output
    
    def trainOne(self, inputArray, targetArray):

        if (self.numInputs == 0):
            self.numInputs = len(inputArray)
            self.weights = [.5] * self.numInputs

        guesses = []
        for inp in range(len(targetArray)):
            target = targetArray[inp]

            inputs = []
            for i in range(len(inputArray)):
                inputs.append(inputArray[i][inp])

            guess = self.calculateOut(inputs)
            error = target - guess

            guesses += [guess]

            for i in range(len(inputs)):
                self.weights[i] += error * inputs[i] * self.learnRate

        return guesses
    
    def train(self, X, Y, rounds=5000):
        trained = False
        iter = 0
        while not trained:

            iter += 1
            guesses = self.trainOne(X, Y)

            right = 0
            for gu in range(len(Y)):
                if guesses[gu] == Y[gu]:
                    right += 1
                
            
            if iter > rounds-1 or right >= len(Y) - (0):
                input(f"Trained '{iter}' rounds | '{right}' Correct out of '{len(Y)}' | Accuracy : {(right / len(Y)) * 100}% ")
                trained = True
                


