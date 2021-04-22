# import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors
from scipy.special import expit
import autograd.numpy as np
from autograd import elementwise_grad, grad
from autograd.numpy import exp, log
import cma
np.random.seed(42)

class LogisticRegression():
    def __init__(self, learningRate = 0.3, maxIterations = 100, regularization = None, invRegLambda = 0.1):
        self.learningRate = learningRate 
        self.maxIterations = maxIterations 
        self.regularization = regularization 
        self.invRegLambda = invRegLambda 
        self.theta = []
        self.X = None 
        self.y = None 
    
    def fit(self, X, y):
        self.X = np.array(X) #converting X to np array 
        self.y = np.array(y) #converting y to np array
        self.X = np.append(np.ones((self.X.shape[0],1)),self.X,axis=1) #appending columns of ones
        self.theta = np.zeros(self.X.shape[1])
        for iterationNum in range(self.maxIterations):
            predictionY = self.__sigmoid(self.X.dot(self.theta)) #ND
            errorY =  predictionY - self.y #N
            self.theta -= (self.learningRate*(np.dot(self.X.T,errorY))) / (self.X.shape[0]) 
        self.theta
    

    def fit_autograd(self, X, y):
        self.X = np.array(X) #converting X to np array 
        self.y = np.array(y) #converting y to np array
        self.X = np.append(np.ones((self.X.shape[0],1)),self.X,axis=1) #appending columns of ones
        self.theta = np.random.rand(self.X.shape[1])#np.ones(self.X.shape[1])#
        agrad = elementwise_grad(self.costFunctionUnregularised)
        agrad1 = elementwise_grad(self.costFunctionL1Regularised)
        agrad2 = elementwise_grad(self.costFunctionL2Regularised)
        for iterationNum in range(self.maxIterations):
            if self.regularization == 'l1':
                self.theta -= (self.learningRate*(agrad1(self.theta, self.X, self.y))) /(self.X.shape[0])
            elif self.regularization == 'l2':
                self.theta -= (self.learningRate*(agrad2(self.theta, self.X, self.y))) /(self.X.shape[0])
            else: 
                self.theta -= (self.learningRate*(agrad(self.theta, self.X, self.y))) /(self.X.shape[0])
        return self.theta

    def costFunctionUnregularised(self,theta,X,y):
        yHat = np.dot(X,theta)
        yHatZ = 1/(1 + np.exp(-yHat))
        return -np.sum(y*(log(yHatZ)) + (1-y)*log(1 - yHatZ))
    
    def costFunctionL1Regularised(self,theta,X,y):
        yHat = np.dot(X,theta)
        yHatZ = 1/(1 + np.exp(-yHat))
        return -np.sum(y*(log(yHatZ)) + (1-y)*log(1 - yHatZ)) + self.invRegLambda*(abs(theta))
    
    def costFunctionL2Regularised(self,theta,X,y):
        yHat = np.dot(X,theta)
        yHatZ = 1/(1 + np.exp(-yHat))
        return -np.sum(y*(log(yHatZ)) + (1-y)*log(1 - yHatZ)) +self.invRegLambda*(theta.T*theta)

    def predictEstimates(self,X):
        return self.__sigmoid(X.dot(self.theta[1:])+self.theta[0])
    
    def predict(self,X):
        return np.where(self.predictEstimates(X)>0.5,1,0)
    

    def __sigmoid(self, z):
        return 1/(1+np.exp(-z))
    
    def plot_decision_boundary(self, X, y, model):
        bias = self.theta[0]
        weight1, weight2 = self.theta[1], self.theta[2]
        c = -bias/weight2 #calculating intercept
        m = -weight1/weight2 #calculating slope
        xmin, xmax = -1, 2
        ymin, ymax = -1, 2.5
        xd = np.array([xmin, xmax])
        yd = m*xd + c
        plt.plot(xd, yd, 'k', lw=1, ls='--')
        plt.fill_between(xd, yd, ymin, color='tab:blue', alpha=0.2)
        plt.fill_between(xd, yd, ymax, color='tab:orange', alpha=0.2)

        plt.scatter(*X[y==0].T, s=8, alpha=0.5)
        plt.scatter(*X[y==1].T, s=8, alpha=0.5)
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)
        plt.ylabel(r'$x_2$')
        plt.xlabel(r'$x_1$')
        plt.savefig('q1_decision_boundry.png')
        plt.show()

    def fitMulticlass(self,X,y):
        self.X = np.array(X) #converting X to np array 
        self.y = np.array(y) #converting y to np array
        self.X = np.append(np.ones((self.X.shape[0],1)),self.X,axis=1)
        self.totalFeatures = self.X.shape[1]
        self.theta= np.zeros((self.totalFeatures,y.shape[1]))
        for i in range(self.maxIterations):
            softP = self.softmax(self.X, self.theta)
            for j in range(self.y.shape[1]):
                val = np.exp(self.X.dot(self.theta[:,j]))/softP   
                error = -(self.y[:,j]-val)
                self.theta[:,j] = self.theta[:,j]- (self.learningRate * error.dot(self.X))/self.X.shape[0]
        return self.theta

    def softmax(self,X,theta):
        softP = 0
        for i in range(self.y.shape[1]):
            softP+=np.exp(X.dot(theta[:,i]))
        return softP

    def fitMulticlassAutograd(self,X,y):
        self.X = np.array(X) #converting X to np array 
        self.y = np.array(y) #converting y to np array
        self.X = np.append(np.ones((self.X.shape[0],1)),self.X,axis=1)
        self.totalFeatures = self.X.shape[1]
        self.theta= np.zeros((self.totalFeatures,y.shape[1]))
        agrad = elementwise_grad(self.costFunctionAutogradMulticlass, 0)
        for i in range(self.maxIterations):
            self.theta = self.theta- (self.learningRate * agrad(self.theta,self.X, self.y))/self.X.shape[0]
        return self.theta

    
    def costFunctionAutogradMulticlass(self,theta, X, y):
        costVal = 0
        for i in range(self.X.shape[0]):
            maxiSample = np.argmax(y[i])
            val1 = np.sum(np.exp(np.dot(X[[i],:],theta)))
            val2 = np.exp(np.dot(X[[i],:],theta[:,maxiSample]))
            costVal+= log(val2/val1)
        return -costVal
        
    def predictMulticlass(self,X):
        Z = 1/(1+np.exp(-(X.dot(self.theta[1:] )+self.theta[0])))
        b = np.zeros_like(Z)		
        b[np.arange(len(Z)), Z.argmax(1)] = 1
        return b
