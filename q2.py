import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from logisticRegression.logisticRegression import LogisticRegression
from metric import *
from sklearn.preprocessing import MinMaxScaler         
from sklearn.model_selection import KFold
scaler = MinMaxScaler()
np.random.seed(42)

from sklearn.datasets import load_breast_cancer


def __sigmoid(z):
        return 1/(1+np.exp(-z))
def predictEstimates(X,theta):
        return __sigmoid(X.dot(theta[1:])+theta[0])
    
def predict(X,theta):
    return np.where(predictEstimates(X,theta)>0.5,1,0)
    

X = load_breast_cancer().data
y = load_breast_cancer().target
scalar = MinMaxScaler()
scalar.fit(X)
X = scalar.transform(X)


print('\n----------Autograd Fit L1 regularised------------')
LR = LogisticRegression(learningRate = 0.1, maxIterations = 100, regularization='l1', invRegLambda = 0.1)
LR.fit_autograd(X, y)
y_hat = LR.predict(X)
print(accuracy(y_hat, y))

print('\n----------Autograd Fit L2 regularised---------------')
LR = LogisticRegression(learningRate = 0.1, maxIterations = 100, regularization='l2', invRegLambda = 0.1)
LR.fit_autograd(X, y)
y_hat = LR.predict(X)
print(accuracy(y_hat, y))


X = load_breast_cancer().data
y = load_breast_cancer().target
scalar = MinMaxScaler()
scalar.fit(X)
X = scalar.transform(X)

print('\n---------L1 Hyperparamter Testing---------')
kfolding = KFold(3, True, 1)
lambdaValue = []
accuracyValue = []
for train, test in kfolding.split(X):
    maximum = 0
    bestTheta = 0
    bestLambda = 0
    for k in range(1,12):
        for trainData, testData in kfolding.split(train):
            trainSetData, testSetData= X[trainData], X[testData]
            ytrainData, ytestData= y[trainData], y[testData]
            LR = LogisticRegression(learningRate = 0.1, maxIterations = 100, regularization='l1', invRegLambda = k/100)
            theta = LR.fit_autograd(trainSetData, ytrainData)
            yHat = LR.predict(testSetData)
            accu = accuracy(yHat, ytestData)
            if(maximum<accu):
                maximum = accu
                bestTheta = theta 
                bestLambda = k/100
            maximum = max(maximum,accu)
    finalTestSet = X[test]
    yFinalTest = y[test]
    yHat = predict(finalTestSet,bestTheta)
    Accuracy = accuracy(yHat, yFinalTest)
    accuracyValue.append(Accuracy)
    lambdaValue.append(bestLambda)
print(accuracyValue,lambdaValue)

X = load_breast_cancer().data
y = load_breast_cancer().target
scalar = MinMaxScaler()
scalar.fit(X)
X = scalar.transform(X)

print('\n---------L2 Hyperparamter Testing---------')
kfolding = KFold(3, True, 1)
lambdaValue = []
accuracyValue = []
for train, test in kfolding.split(X):
    maximum = 0
    bestTheta = 0
    bestLambda = 0
    for k in range(1,12):
        for trainData, testData in kfolding.split(train):
            trainSetData, testSetData= X[trainData], X[testData]
            ytrainData, ytestData= y[trainData], y[testData]
            LR = LogisticRegression(learningRate = 0.1, maxIterations = 100, regularization='l2', invRegLambda = k/100)
            theta = LR.fit_autograd(trainSetData, ytrainData)
            yHat = LR.predict(testSetData)
            accu = accuracy(yHat, ytestData)
            if(maximum<accu):
                maximum = accu
                bestTheta = theta 
                bestLambda = k/100
            maximum = max(maximum,accu)
    finalTestSet = X[test]
    yFinalTest = y[test]
    yHat = predict(finalTestSet,bestTheta)
    Accuracy = accuracy(yHat, yFinalTest)
    accuracyValue.append(Accuracy)
    lambdaValue.append(bestLambda)
print(accuracyValue,lambdaValue)



print('----------Checking for important features--------------')

X = load_breast_cancer().data
y = load_breast_cancer().target
scalar = MinMaxScaler()
scalar.fit(X)
X = scalar.transform(X)
lambdas = []
thetasAll = []
for lambdaVal in range(1,10):
    LR = LogisticRegression(learningRate = 1e-8, maxIterations = 100, regularization='l1', invRegLambda = lambdaVal/1e-1)
    theta = LR.fit_autograd(X, y)
    lambdas.append(lambdaVal/1e-1)
    thetasAll.append(theta)
    print(lambdaVal, "corresponding thetas\n", theta) 