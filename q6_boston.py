import numpy as np
import neuralNetwork.network
from neuralNetwork.network import NeuralNetwork, MyNeuralNetworkLayer
import math
from sklearn.preprocessing import MinMaxScaler    
from sklearn.datasets import load_boston
from metric import *
from sklearn.model_selection import KFold
x_train = load_boston().data
y_train = load_boston().target
scalar = MinMaxScaler()
scalar.fit(x_train)
x_train = scalar.transform(x_train)
x_train = x_train.reshape(x_train.shape[0], 1, 13)
x_train = x_train.astype('float32')

kf = KFold(n_splits=3)
avgRMSE = 0
for trainData, testData in kf.split(x_train):
    Xtrain, Xtest = x_train[trainData], x_train[testData]
    ytrain, ytest = y_train[trainData], y_train[testData]
    net = NeuralNetwork(loss=neuralNetwork.network.mseLoss)
    net.addLayer(MyNeuralNetworkLayer(13, 13, activation=neuralNetwork.network.sigmoid))            
    net.addLayer(MyNeuralNetworkLayer(13, 13, activation=neuralNetwork.network.relu))                  
    net.addLayer(MyNeuralNetworkLayer(13, 1, activation=neuralNetwork.network.identity))                  
    net.fit(Xtrain, ytrain, epochs=10, learningRate=0.001)
    out = net.predict(Xtest)
    # print(out[:10],y_train[:10])
    print(rmse(np.array(out),np.array(ytest)))
    avgRMSE+=rmse(np.array(out),np.array(ytest))
print("Overall MSE error: ", avgRMSE/3)


