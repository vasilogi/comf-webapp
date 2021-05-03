# Third party imports

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from scipy.interpolate import interp1d

# Local application imports
from modules.reaction_models import Model
from modules.euclidean_distance import resREr

def simpleLinearFit(time,k):
    return k*time

def conversionRegression(time,conversion,model):
    # perform Non-Linear Regression
    # fit the experimental conversion (conversion)
    # calculate the Arrhenius rate constant (k)

    x          = time
    y          = conversion
    popt, pcov = curve_fit(model.alpha,x,y,p0=0.1)          # p0 : initial guess
    # popt: optimal values for the parameters so that the sum of the squared residuals of f(xdata, *popt) - ydata is minimized.
    k          = popt[0]                                    # Arrhenius rate constant
    yfit       = np.array([model.alpha(t,k) for t in time]) # simulated conversion fraction

    # calculate the determination coefficient
    r_squared = resREr(y,yfit)

    return yfit, k, r_squared

def integralRateRegression(time,conversion,model):
    # perform Non-Linear Regression
    # fit the experimental integral rate conversion (g)
    # calculate the Arrhenius rate constant (k)

    x          = time
    y          = np.array([model.g(a) for a in conversion])
    popt, pcov = curve_fit(simpleLinearFit,x,y,p0=0.1)          # p0 : initial guess
    # popt: optimal values for the parameters so that the sum of the squared residuals of f(xdata, *popt) - ydata is minimized.
    k          = popt[0]                                        # Arrhenius rate constant
    yfit       = np.array([simpleLinearFit(t,k) for t in time]) # simulated conversion fraction

    # calculate the determination coefficient
    r_squared = resREr(y,yfit)

    return yfit, k, r_squared

def comprehensiveRegressor(time,conversion,models):
    # arguments
    # time       (numpy array)
    # conversion (numpy array)
    # models (list of strings)

    # returns
    # a dataframe containg the fitting information

    rate_constant_alpha         = []
    rate_constant_integral      = []
    determination_coef_alpha    = []          # the residuals relative error (ideal = 0)
    determination_coef_integral = []          # the residuals relative error (ideal = 0)
    
    # loop over the models
    for modelIndx, modelName in enumerate(models):

        # pick up the model
        model = Model(modelName)

        # integral rate regression
        yfit, k_integral, r_squared_integral = integralRateRegression(time, conversion, model)

        # if the determination coefficient is negative then the fit is wrong by default
        # and the particular model will be discarded
        # however for a nonlogarithmim visualization of the euclidean distance we need values close to 0
        if r_squared_integral < 0.0:
            r_squared_integral = 0.0

        rate_constant_integral.append(k_integral)
        determination_coef_integral.append(r_squared_integral)

        # conversion regression
        if modelName not in ['D2','D4']:
            yfit, k_alpha, r_squared_alpha = conversionRegression(time, conversion, model)
        else:
            r_squared_alpha = r_squared_integral
            k_alpha         = k_integral

        if r_squared_alpha < 0.0:
            r_squared_alpha = 0.0
        
        rate_constant_alpha.append(abs(k_alpha)) # bug: RuntimeWarning: invalid value encountered in sqrt
        determination_coef_alpha.append(r_squared_alpha)
    
    # pass the data to a dictionary
    data = {'model': models,
            'rate_constant - alpha': rate_constant_alpha,
            'rate_constant - integral': rate_constant_integral,
            'R2 - alpha': determination_coef_alpha,
            'R2 - integral': determination_coef_integral}

    # dictionary to dataframe
    df = pd.DataFrame(data)

    return df


def arrheniusEnthalpy(rate_constant,temperature):
    # arguments 
    # rate_constant (numpy array)
    # temperature (numpy array)

    # returns
    # the activation enthalpy (Ea) in a list like [activation enthalpy, mean square error]
    # and the frequency factor (A)
    # and the prediction

    # gas constant 
    R = 8.31446261815324 # J K-1 mol-1

    x = 1.0/temperature
    y = np.log(rate_constant)

    x = x.reshape((-1, 1))

    # linear regression for the logarithmic arrenius equation
    regr   = LinearRegression()
    regr.fit(x, y)
    y_pred = regr.predict(x)

    Ea  = -regr.coef_[0]*R # in J mol-1
    A   = np.exp(regr.intercept_)
    MSE = mean_squared_error(y, y_pred)
    R2   = r2_score(y_pred , y)

    return [Ea,MSE], A

def isocEnthalpy(time,temperature):
    # arguments 
    # time (numpy array)
    # temperature (numpy array)

    # returns
    # the activation enthalpy (Ea) in a list like [activation enthalpy, mean square error]
    # and the ln[g(a)A] factor

    # gas constant 
    R = 8.31446261815324 # J K-1 mol-1

    x = 1.0/temperature
    y = np.log(time)

    x = x.reshape((-1, 1))

    # linear regression for the logarithmic arrenius equation
    regr   = LinearRegression()
    regr.fit(x, y)
    y_pred = regr.predict(x)

    Ea  = regr.coef_[0]*R*1.0e-3 # in kJ mol-1
    gA  = regr.intercept_
    MSE = mean_squared_error(y, y_pred)
    R2   = r2_score(y_pred , y)

    return [Ea,MSE], gA

def interpolateTime(time,conversion,interConversion):
    # arguments
    # time (numpy array)
    # conversion (numpy array)
    # npoints (integer) - the number of the interpolation points
    
    # returns
    # the time for the interpolated points
    y               = time
    x               = conversion
    interpol        = interp1d(x,y,kind='nearest',fill_value="extrapolate")
    return interpol(interConversion)