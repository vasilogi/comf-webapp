# Standard library imports
import sys

# Third party imports
import numpy as np

class Model:

    """A class that contains the conversion, integral
       and differential reaction rates for each model.
       
       name       = the name of the model                   (string)
       conversion = the experimentally recorded conversion  (single value)
       time       = the experimentally recorded time        (single value)
       k          = the Arrhenius constant                  (float)
       
       """
    def __init__(self,modelname):
        self.name   = modelname
    
    def g(self,a):
        
        """Return the integral reaction rate for each different model"""

        if (self.name == 'A2'):
            if a == 0.0:
                print('A2 model')
                print('the input conversion value is equal to one and the logarithm argument is zero')
                print('the logarithm is not defined')
                print('the recommended conversion range is 0.05-0.95')
                sys.exit('Program Stopped')
            else:
                rate = np.sqrt(-np.log(1.0-a))
        elif (self.name == 'A3'):
            if a == 0.0:
                print('A3 model')
                print('the input conversion value is equal to one and the logarithm argument is zero')
                print('the logarithm is not defined')
                print('the recommended conversion range is 0.05-0.95')
                sys.exit('Program Stopped')
            else:
                rate = (-np.log(1.0-a))**(1.0/3.0)
        elif (self.name == 'A4'):
            if a == 0.0:
                print('A4 model')
                print('the input conversion value is equal to one and the logarithm argument is zero')
                print('the logarithm is not defined')
                print('the recommended conversion range is 0.05-0.95')
                sys.exit('Program Stopped')
            else:
                rate = (-np.log(1.0-a))**(1.0/4.0)
        elif (self.name == 'D1'):
            rate = a**2.0
        elif (self.name == 'D2'):
            if a == 0.0:
                print('D2 model')
                print('the input conversion value is equal to one and the logarithm argument is zero')
                print('the logarithm is not defined')
                print('the recommended conversion range is 0.05-0.95')
                sys.exit('Program Stopped')
            else:
                rate = (1.0-a)*np.log(1.0-a)+a
        elif (self.name == 'D3'):
            rate = (1.0-(1.0-a)**(1.0/3.0))**2.0
        elif (self.name == 'D4'):
            rate = 1.0 - (2.0/3.0)*a - (1.0-a)**(2.0/3.0)
        elif (self.name == 'F0'):
            rate = a
        elif (self.name == 'F1'):
            if a == 0.0:
                print('F1 model')
                print('the input conversion value is equal to one and the logarithm argument is zero')
                print('the logarithm is not defined')
                print('the recommended conversion range is 0.05-0.95')
                sys.exit('Program Stopped')
            else:
                rate = -np.log(1.0-a)
        elif (self.name == 'F2'):
            rate = (1.0/(1.0-a))-1.0
        elif (self.name == 'F3'):
            rate = 0.5*((1.0-a)**(-2.0) - 1.0)
        elif (self.name == 'P2'):
            rate = a**0.5
        elif (self.name == 'P3'):
            rate = a**(1.0/3.0)
        elif (self.name == 'P4'):
            rate = a**0.25
        elif (self.name == 'R2'):
            rate = 1.0-(1.0-a)**0.5
        elif (self.name == 'R3'):
            rate = 1.0-(1.0-a)**(1.0/3.0)
        else:
            print('Wrong model choice')
            print('Instead, choose one from the models list: "A2-A4","D1-D4","F0-F3","P2-P4","R2-R3"')
            sys.exit('Program Stopped')
        
        return rate
    
    def f(self,a):
        
        """Return the differential reaction rate for each different model"""
        if (self.name == 'A2'):
            if a == 0.0:
                print('A2 model')
                print('the input conversion value is equal to one and the logarithm argument is zero')
                print('the logarithm is not defined')
                print('the recommended conversion range is 0.05-0.95')
                sys.exit('Program Stopped')
            else:
                rate = 2.0*(1.0-a)*np.sqrt(-np.log(1.0-a))
        elif (self.name == 'A3'):
            if a == 0.0:
                print('A3 model')
                print('the input conversion value is equal to one and the logarithm argument is zero')
                print('the logarithm is not defined')
                print('the recommended conversion range is 0.05-0.95')
                sys.exit('Program Stopped')
            else:
                rate = 3.0*(1.0-a)*(-np.log(1.0-a))**(2.0/3.0)
        elif (self.name == 'A4'):
            if a == 0.0:
                print('A4 model')
                print('the input conversion value is equal to one and the logarithm argument is zero')
                print('the logarithm is not defined')
                print('the recommended conversion range is 0.05-0.95')
                sys.exit('Program Stopped')
            else:
                rate = 4.0*(1.0-a)*(-np.log(1.0-a))**(3.0/4.0)
        elif (self.name == 'D1'):
            rate = 1.0/(2.0*a)
        elif (self.name == 'D2'):
            if a == 0.0:
                print('D2 model')
                print('the input conversion value is equal to one and the logarithm argument is zero')
                print('the logarithm is not defined')
                print('the recommended conversion range is 0.05-0.95')
                sys.exit('Program Stopped')
            else:
                rate = -1.0/np.log(1.0-a)
        elif (self.name == 'D3'):
            if a <= 1.0:
                rate = (3.0*(1.0-a)**(2.0/3.0))/(2.0*(1.0-(1.0-a)**(1.0/3.0)))
            else: # bug: RuntimeWarning: invalid value encountered in double_scalars
                rate = 0.0
        elif (self.name == 'D4'):
            rate = (3.0/2.0)*((1.0-a)**(-1.0/3.0)-1.0)**(-1.0)
        elif (self.name == 'F0'):
            rate = 1.0
        elif (self.name == 'F1'):
            rate = 1.0-a
        elif (self.name == 'F2'):
            rate = (1.0-a)**2.0
        elif (self.name == 'F3'):
            rate = (1.0-a)**3.0
        elif (self.name == 'P2'):
            rate = 2.0*(a**(1.0/2.0))
        elif (self.name == 'P3'):
            rate = 3.0*(a**(2.0/3.0))
        elif (self.name == 'P4'):
            rate = 4.0*(a**(3.0/4.0))
        elif (self.name == 'R2'):
            rate = 2.0*((1.0-a)**(1.0/2.0))
        elif (self.name == 'R3'):
            if a <= 1.0:
                rate = 3.0*((1.0-a)**(2.0/3.0))
            else: # bug: RuntimeWarning: invalid value encountered in double_scalars
                rate = 0.0
        else:
            print('Wrong model choice')
            print('Instead, choose one from the models list: "A2-A4","D1-D4","F0-F3","P2-P4","R2-R3"')
            sys.exit('Program Stopped')
            
        return rate

    def alpha(self,t,k):

        # arguments
        # t : time
        # k : arrenius rate constant

        # bug: RuntimeWarning: invalid value encountered in sqrt

        # it has been noticed that when regressing the conversion curve
        # especially with an A2-A4 model, the rate constant results negative
        # that is why the sqrt takes a negative argument.
        # however, most of the times the negative value is exactly the correct if
        # we turn the sign which we do to correct the bug

        k = abs(k)
        
        """Return the simulated conversion for each different model"""

        if (self.name == 'A2'):
            rate = 1.0-np.exp(-k*k*t*t)
        elif (self.name == 'A3'):
            rate = 1.0-np.exp(-k*k*k*t*t*t)
        elif (self.name == 'A4'):
            rate = 1.0-np.exp(-k*k*k*k*t*t*t*t)
        elif (self.name == 'D1'):
            rate = (k*t)**0.5
        elif (self.name == 'D2'):
            rate = False
            print('This model does not have an analytical expression for conversion')
            sys.exit('Program Stopped')
        elif (self.name == 'D3'):
            rate = 1.0 - (1.0-(k*t)**0.5)**3.0
        elif (self.name == 'D4'):
            rate = False
            print('This model does not have an analytical expression for conversion')
            sys.exit('Program Stopped')
        elif (self.name == 'F0'):
            rate = k*t
        elif (self.name == 'F1'):
            rate = 1.0-np.exp(-k*t)
        elif (self.name == 'F2'):
            rate = (k*t)/(1.0 + k*t)
        elif (self.name == 'F3'):
            rate = (1.0 + 2.0*k*t - (1.0 + 2.0*k*t)**0.5)/(1.0 + 2.0*k*t)
        elif (self.name == 'P2'):
            rate = k*k*t*t
        elif (self.name == 'P3'):
            rate = k*k*k*t*t*t
        elif (self.name == 'P4'):
            rate = k*k*k*k*t*t*t*t
        elif (self.name == 'R2'):
            rate = k*t*(2.0 - k*t)
        elif (self.name == 'R3'):
            rate = k*t*(3.0 - 3.0*k*t + (k*t)**2.0 )
        else:
            print('Wrong model choice')
            print('Instead, choose one from the models list: "A2-A4","D1-D4","F0-F3","P2-P4","R2-R3"')
            sys.exit('Program Stopped')
        
        return rate