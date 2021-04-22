import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from logisticRegression.logisticRegression import LogisticRegression
from metric import *
import matplotlib.colors as cma

from sklearn.datasets import load_digits
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix, ConfusionMatrixDisplay

def oneHotEncoding(y, n_labels, dtype):
    matrix = np.zeros((len(y), n_labels))
    for i, val in enumerate(y):
        matrix[i, int(val)] = 1
    return matrix.astype(dtype)

def predict_multi(X, theta):
    bias = np.ones((X.shape[0], 1))
    X = np.append(bias, X, axis=1)
    sigmoidZ = 1/(1+np.exp(-(X.dot(theta))))			
    myAnswers = []
    for i in sigmoidZ:
        myAnswers.append(np.argmax(i))
    return myAnswers


X = load_digits().data
y = load_digits().target
scalar = MinMaxScaler()
scalar.fit(X)
X = scalar.transform(X)

print('\n----------MultiClass Normal Fit------------')
yEncoded = oneHotEncoding(y,10,dtype="float")
LR = LogisticRegression(learningRate = 6e-2, maxIterations = 60, regularization=None)
LR.fitMulticlass(X,yEncoded)
y_hat = LR.predictMulticlass(X)
print(multiaccuracy(y_hat, yEncoded))

print('\n----------MultiClass Autograd Fit------------')
LR = LogisticRegression(learningRate = 6e-2, maxIterations = 60, regularization=None)
LR.fitMulticlassAutograd(X,yEncoded)
y_hat = LR.predictMulticlass(X)
print(multiaccuracy(y_hat, yEncoded))


kfolding = KFold(4, True, 1)
for train, test in kfolding.split(X):
    maximum = 0
    bestTheta = 0
    for trainData, testData in kfolding.split(train):
        trainSetData, testSetData= X[trainData], X[testData]
        ytrainData, ytestData= y[trainData], y[testData]
        ytestData = oneHotEncoding(ytestData, 10, dtype="float")
        ytrainData = oneHotEncoding(ytrainData, 10, dtype="float")
        LR = LogisticRegression(learningRate=0.01,  maxIterations = 100, regularization=None)
        theta = LR.fitMulticlass(trainSetData, ytrainData)
        testSetData = np.array(testSetData)
        y_hat = LR.predictMulticlass(testSetData)
        accu = multiaccuracy(y_hat, ytestData)
        if(maximum<accu):
            maximum = accu
            bestTheta = theta 
    finalTestSet = X[test]
    y_final_test = y[test]
    yHat = predict_multi(finalTestSet,bestTheta)
    yHat = np.array(yHat)
    acc = accuracy(yHat, y_final_test)

yHat = predict_multi(X,bestTheta)
print('\n----------Confusion Matrix------------')
matrix = confusion_matrix(y, yHat)
print(matrix)

disp = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=np.array([0,1,2,3,4,5,6,7,8,9]))
disp.plot()
plt.savefig('q3_c_ConfusionMatrix.png') 
plt.show()

print('\n----------PCA Analysis------------')
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X = pca.fit_transform(load_digits().data)
plt.scatter(X[:,0],X[:, 1], c=load_digits().target, cmap="Paired")
plt.colorbar()
plt.savefig('q3_d_PCA_scatter_plot.png')
plt.show()