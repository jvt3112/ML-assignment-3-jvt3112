import numpy as np
import neuralNetwork.network
from neuralNetwork.network import NeuralNetwork, MyNeuralNetworkLayer
from keras.utils import np_utils
from sklearn.datasets import load_digits
from metric import *
from sklearn.model_selection import KFold
def calArgMax(out):
    ans = []
    for i in out:
        ans.append(np.argmax(i))
    return np.array(ans)

x_train = load_digits().data
y_train = load_digits().target
x_train = x_train.reshape(x_train.shape[0], 1, 64)
x_train = x_train.astype('float32')
y_train = np_utils.to_categorical(y_train)

kf = KFold(n_splits=3)
avgAccuracy = 0
for trainData, testData in kf.split(x_train):
    Xtrain, Xtest = x_train[trainData], x_train[testData]
    ytrain, ytest = y_train[trainData], y_train[testData]
    net = NeuralNetwork(loss=neuralNetwork.network.mseLoss)
    net.addLayer(MyNeuralNetworkLayer(64, 64, activation=neuralNetwork.network.sigmoid))          
    net.addLayer(MyNeuralNetworkLayer(64, 32, activation=neuralNetwork.network.relu))                   
    net.addLayer(MyNeuralNetworkLayer(32, 10, activation=neuralNetwork.network.sigmoid))                
    net.fit(Xtrain, ytrain, epochs=35, learningRate=0.9)
    out = net.predict(Xtest)
    Y = calArgMax(out)
    yHat = calArgMax(ytest)
    print(accuracy(yHat,Y))
    avgAccuracy+= accuracy(yHat,Y)
print("Overall Accuracy: ",avgAccuracy/3)



