# Third party imports
import pandas as pd
import numpy as np

# Local application imports
from modules.arrhenius import mass2conv

def read_filtrated_datafile(CSV,low,high):
    # CSV:  absolute path of the csv file
    # low:  lower limit of conversion
    # high: higher limit of conversion
    df           = pd.read_csv(CSV)

    # Check if conversion columns exists in the CSV file

    if 'conversion' in df:
        # filter data
        rangeFilter  = (df['conversion']>= low) & (df['conversion']<= high)
        df           = df[rangeFilter]
    else:
        # convert mass to conversion
        mass = df['mass'].to_numpy()
        m0   = mass[0]    # initial weight
        minf = mass[-1]   # final weight
        conversion = np.array([mass2conv(m0,mt,minf) for mt in mass])
        df['conversion'] = conversion
        # filter data
        rangeFilter  = (df['conversion']>= low) & (df['conversion']<= high)
        df           = df[rangeFilter]

    conversion   = df['conversion'].to_numpy()
    time         = df['time'].to_numpy()
    temperature  = df['temperature'].to_numpy()[0]
        
    return conversion, time, temperature

def read_units(CSV):
    # CSV:  absolute path of the csv file
    df        = pd.read_csv(CSV)
    timeUnits = df['time units'].to_list()[0]
    massUnits = df['mass units'].to_list()[0]
    tempUnits = df['temperature units'].to_list()[0]

    if tempUnits == 'Kelvin':
        tempUnits = 'K'
        
    return timeUnits, massUnits, tempUnits