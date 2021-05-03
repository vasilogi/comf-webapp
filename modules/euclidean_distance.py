# goodness of fit measures

# Third party imports
import pandas as pd
import numpy as np


def ssRes(y,yfit):
    # calculate the sum of square residuals (ideal = 0)
    residuals = y - yfit
    ss_res    = np.sum(residuals**2.0)
    return ss_res

def rSquared(y,yfit):
    # calculate the determination coefficient for linear fits (ideal = 1)
    residuals = y - yfit
    ss_res    = np.sum(residuals**2.0)
    ss_tot    = np.sum((y-np.mean(y))**2.0)
    return 1.0 - (ss_res / ss_tot)

def MSE(y,yfit):
    # calculate the mean square error (ideal = 0)
    residuals = y - yfit
    ss_res    = np.sum(residuals**2.0)
    return (1.0/len(y))*ss_res

def resEr(y,yfit):
    # calculate the residuals error (ideal = 0)
    residuals = y - yfit
    return np.sum(residuals)

def resAEr(y,yfit):
    # calculate the residuals absolute error (ideal = 0)
    residuals = np.abs(y - yfit)
    return np.sum(residuals)

def resREr(y,yfit):
    # calculate the residuals relative error (ideal = 0)
    residuals = np.abs(y - yfit)/np.abs(y)
    return np.sum(residuals)

# euclidean distance in the context of model fitting method (COMF)
# https://www.mdpi.com/2073-4352/10/2/139
def d(p,q):
    return np.sqrt((0.0-p)**2.0 + (0.0-q)**2.0)

def convergenceData(df):
    # arguments: a dataframe that contains at least the columns | models | R2 - alpha | R2 - integral
    # returns  : a pandas dataframe containing the model with the fitting error (Euclidean Distance)
    
    # filter the negative determination coefficient values
    numDf = df._get_numeric_data()
    numDf[numDf < 0] = 0

    # calculate the Euclidean distance (fitting error)
    p = numDf['R2 - alpha'].to_numpy()
    q = numDf['R2 - integral'].to_numpy()
    dist = d(p,q)
    data = {'model': df['model'].to_list(), 'fitting error': dist}
    return pd.DataFrame(data)