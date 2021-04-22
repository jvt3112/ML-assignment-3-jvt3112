from autograd import numpy as np, elementwise_grad

def sigmoid(x):
    return 1/(1+np.exp(-x))

def relu(x):
    return np.maximum(x,0)

def identity(x):
    return x

def mseLoss(y_pred, y_true):
    return np.mean(np.power(y_true-y_pred, 2))

class MyNeuralNetworkLayer():
    def __init__(self,inputNeurons,outputNeurons,activation=None):
        #initialising neural network layer
        self.input = None 
        self.output = None
        self.activation = activation
        self.withActivationInput = None 
        self.weigtsMatrix = np.random.rand(inputNeurons, outputNeurons) 
        self.biasMatrix = np.random.rand(1, outputNeurons) 
    
    def myForwardPass(self, inputData):
        self.input = inputData
        if(self.activation==None): #if the layer doesn't contains activation function
            self.output = np.dot(self.input,self.weigtsMatrix)+self.biasMatrix #calculating forward output for the current layer
            return self.output
        else:
            self.withActivationInput = self.activation(np.dot(self.input,self.weigtsMatrix)+self.biasMatrix) #calculating forward output for the current layer with activation involved
            self.output = self.withActivationInput 
            return self.output 


    def myBackwardPass(self, output_error, learningRate):
        if(self.activation!=None): #if activation fucntion applied using autograd to calculate the gradient descent for activation
            agrad = elementwise_grad(self.activation)
            output_error = agrad(self.withActivationInput)*output_error #first part is the local gradient and the output_error here is the upstream gardient that we got from previous layer
        inputError = np.dot(output_error,self.weigtsMatrix.T)  #calculating input  error from WX+B with respect to X we will get upstreamGradient*localGradiet(W in these case)
        weightsError = np.dot(self.input.T,output_error)  #calculating weights error from WX+B with respect to W we will get upstreamGradient*localGradiet(X in these case)
        self.weigtsMatrix -= learningRate*weightsError  #updating weight matrix
        self.biasMatrix -= learningRate*output_error #updating bias matrix
        return inputError


class NeuralNetwork():
    def __init__(self, loss=None):
        #initialising the neural network
        self.listOfLayers = []
        self.lossFunction = loss

    def addLayer(self, layer):
        #adding layer
        self.listOfLayers.append(layer)
    
    def predict(self, inputData):
        myAnswers = []
        for i in range(len(inputData)):
            outputOfCurrentLayer = inputData[i]
            for layer in self.listOfLayers:
                outputOfCurrentLayer = layer.myForwardPass(outputOfCurrentLayer) #here outputOfcUrrentLayer calculates forward of each layer thus iterating over all the layers
            myAnswers.append(outputOfCurrentLayer)
        return myAnswers 

    def fit(self, xTrain, yTrain, epochs, learningRate):
        for i in range(epochs):
            err = 0
            for j in range(len(xTrain)):
                outputOfCurrentLayer = xTrain[j]
                for layer in self.listOfLayers:
                    outputOfCurrentLayer = layer.myForwardPass(outputOfCurrentLayer) #here outputOfcUrrentLayer calculates forward of each layer thus iterating over all the layers
                err += self.lossFunction(outputOfCurrentLayer, yTrain[j]) #this calculates the err we have for the input 
                #calculating gradient for the loss function for updating weights in reverse format
                agrad = elementwise_grad(self.lossFunction)
                error = agrad(outputOfCurrentLayer,yTrain[j])
                #error forms the upstream gardient for the previous layer in reverse format
                for layer in reversed(self.listOfLayers):
                    #calculting grdient for each layer and passing it as upstream gardeint in the previous layer of the network
                    #thus step by step updating weights of the network
                    error = layer.myBackwardPass(error, learningRate)
            err /= len(xTrain)
            print('Epoch %d/%d   error=%f' % (i+1, epochs, err))

     