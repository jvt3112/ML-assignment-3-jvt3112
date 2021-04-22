import math  
import pandas as pd
import numpy as np
def accuracy(y_hat, y):
    """
    Function to calculate the accuracy
    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    Output:
    > Returns the accuracy as float
    """
    """
    The following assert checks if sizes of y_hat and y are equal.
    Students are required to add appropriate assert checks at places to
    ensure that the function does not fail in corner cases.
    """
    assert(y_hat.size == y.size)
    totalTruth = 0
    for i in range(y_hat.size):
        if(y_hat[i]==y[i]):
            totalTruth+=1 
    return (totalTruth/y_hat.size)

def multiaccuracy(y_hat, y):
    assert(y_hat.size == y.size)
    totalTruth = 0
    for i in range(y_hat.shape[0]):
        if(y_hat[i]==y[i]).all():
            totalTruth+=1 
    return (totalTruth/y_hat.shape[0])

def precision(y_hat, y, cls):
    """
    Function to calculate the precision
    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    > cls: The class chosen
    Output:
    > Returns the precision as float
    """
    assert(y_hat.size == y.size)
    clsElementInPrediction = 0 
    preciseValueInPrediction = 0 
    for i in range(y_hat.size):
        if(y_hat[i]==cls):
            clsElementInPrediction+=1 
            if(y_hat[i]==y[i]):
                preciseValueInPrediction+=1 
    if(clsElementInPrediction==0):
        return 0
    return (preciseValueInPrediction/clsElementInPrediction)

def recall(y_hat, y, cls):
    """
    Function to calculate the recall
    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    > cls: The class chosen
    Output:
    > Returns the recall as float
    """
    assert(y_hat.size == y.size)
    clsElementInGroundTruth = 0 
    recallValueInPrediction = 0 
    for i in range(y_hat.size):
        if(y[i]==cls):
            clsElementInGroundTruth+=1 
            if(y_hat[i]==y[i]):
                recallValueInPrediction+=1 
    if(clsElementInGroundTruth==0):
        return 0
    return (recallValueInPrediction/clsElementInGroundTruth)

def rmse(y_hat, y):
    """
    Function to calculate the root-mean-squared-error(rmse)
    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    Output:
    > Returns the rmse as float
    """

    assert(y_hat.size == y.size)
    meanSquaredError = 0 
    for i in range(y_hat.size):
        meanSquaredError+=(y_hat[i]-y[i])**(2)
    return math.sqrt(meanSquaredError/y.size)

def mae(y_hat, y):
    """
    Function to calculate the mean-absolute-error(mae)
    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    Output:
    > Returns the mae as float
    """
    assert(y_hat.size == y.size)
    meanAbsoluteError = 0 
    for i in range(y_hat.size):
        meanAbsoluteError+=abs(y_hat[i]-y[i])
    return meanAbsoluteError/y.size
