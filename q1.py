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

X = load_breast_cancer().data
y = load_breast_cancer().target
scalar = MinMaxScaler()
scalar.fit(X)
X = scalar.transform(X)

print('\n---------Unregularised Normal Fit----------')
LR = LogisticRegression(learningRate = 0.1, maxIterations = 1000)
LR.fit(X,y)
y_hat = LR.predict(X)
print(accuracy(y_hat, y))


print('\n---------Unregularised Autograd Fit----------')
LR = LogisticRegression(learningRate = 0.1, maxIterations = 1000)
LR.fit_autograd(X,y)
y_hat = LR.predict(X)
print(accuracy(y_hat, y))


print('\n---------3 KFold----------')
Kfolding = KFold(3, True, 1)
avgAccuracy = 0
for trainData, testData in Kfolding.split(X):
    trainSetData, testSetData= X[trainData], X[testData]
    ytrainData, ytestData= y[trainData], y[testData]
    LR = LogisticRegression(learningRate = 0.1, maxIterations = 1000)
    LR.fit(trainSetData, ytrainData)
    yHat = LR.predict(testSetData)
    acc = accuracy(yHat, ytestData)
    avgAccuracy += acc
print("Overall accuracy of Breast Cancer Dataset on 3 Fold: ", (avgAccuracy/3)*100)


X = load_breast_cancer().data
y = load_breast_cancer().target
scalar = MinMaxScaler()
scalar.fit(X)
X = scalar.transform(X)

print('\n--------- Fit with 2 features----------')
LR = LogisticRegression()
X = X[:,[2,7]]       #[1,15], [1,5], .... [12,14]
thetas = LR.fit(X, y)
y_hat = LR.predict(X)
print(accuracy(y_hat, y))
LR.plot_decision_boundary(X, y,thetas)