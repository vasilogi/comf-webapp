# Standard library imports
import sys

# Third party imports
import numpy as np

def rateConstant(frequency,enthalpy,temperature):
    # S.I.
    # the units of frequency are identical to those of the rate constant k
    # e.g. for a first-order reaction it's s-1
    R    = 8.31446261815324 # J K-1 mol-1
    beta = 1.0/(R*temperature)
    return frequency*np.exp(-beta*enthalpy)

def mass2conv(m0,mt,minf):
    # arguments

    # m0   (float): initial sample weight
    # mt   (float): sample weight at time t
    # minf (float): final sample weight

    # returns

    # the extent of conversion

    if m0 - minf == 0.0:
        print('The denominator in conversion fraction is zero')
        print('Check your input data')
        sys.exit('Program Stopped')
    else:
        return (m0 - mt)/(m0 - minf)

def conv2mass(m0,minf,alpha):
    # arguments

    # m0    (float): initial sample weight
    # minf  (float): final sample weight
    # alpha (float): extent of conversion at time t

    # returns

    # the sample mass at time t

    return m0 - alpha*(m0-minf)