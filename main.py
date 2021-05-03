# Standard library imports
import os

# Third party imports
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import plotly.express as px
import numpy as np
import pandas as pd

# Local application imports
from modules.file_handlers import read_filtrated_datafile, read_units
from modules.regressors import comprehensiveRegressor, arrheniusEnthalpy, interpolateTime, isocEnthalpy
from modules.euclidean_distance import convergenceData
from modules.reaction_models import Model
from modules.arrhenius import rateConstant

# DIRECTORIES
MAIN_DIR      = os.getcwd()                            # current working directory
DATA          = os.path.join(MAIN_DIR,'data')          # data directory
MODEL_FITTING = os.path.join(MAIN_DIR,'model-fitting') # data directory

# models names supported in this code
modelNames = ["A2","A3","A4","D1","D2","D3","D4","F0","F1","F2","F3","P2","P3","P4","R2","R3"]

# get the data from the csv files
DataNames = os.listdir(DATA)
Data      = [os.path.join(DATA,i) for i in DataNames]

# get units globally
timeUnits, massUnits, tempUnits = read_units(Data[0])

# define the options list for the files dropdown
filesOptionsList = [{'label': x, 'value': os.path.join(DATA,x)} for x in DataNames]

# define the options list for the models dropdown
validModelNames = ["A2","A3","A4","D1","D3","F0","F1","F2","F3","P2","P3","P4","R2","R3"]
modelsOptionsList = [{'label': x, 'value': x} for x in validModelNames]

# define the options list for the models dropdown
modelsArrheniusOptionsList = [{'label': x, 'value': x} for x in modelNames]

# import external CSS stylesheets
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

# start the app
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div(children=[
    
    # HEADER
    html.Div(children = 'A web application for solid-state reaction kinetics modelling',
            style={"text-align": "center", "font-size": "100%"}),

    # INPUT DATA GRAPH
    html.Div([
        dcc.Graph(id='input-scatter-plot'),
        html.Div(id='range-slider-text',children='conversion range'),
        dcc.RangeSlider(
            id='input-data-range-slider',
            min=0.0,
            max=1.0,
            step=0.01,
            marks={
            0.0: '0.0',
            0.1: '0.1',
            0.2: '0.2',
            0.3: '0.3',
            0.4: '0.4',
            0.5: '0.5',
            0.6: '0.6',
            0.7: '0.7',
            0.8: '0.8',
            0.9: '0.9',
            1.0: '1.0',
            },
            value=[0.05, 0.95]
        )
    ]),

    # OUTPUT CONVERGENCE PLOT    
    html.Div([
        html.Div(id='dropdown-html',children='choose one data file'),
        dcc.Dropdown(
            id='files-dropdown',
            options=filesOptionsList,
            value=filesOptionsList[0]['value'],
            multi=False
        ),
        html.Div(id='output-congergence-graph-html',children='the model with minimum fitting error is the best candidate'),
        dcc.Graph(id='output-congergence-graph')
    ]),

    # FITTING GRAPH
    html.Div([
        html.Div(id='output-fitting-graph-html',children='choose multiple models to compare'),
        dcc.Dropdown(
            id='models-dropdown',
            options=modelsOptionsList,
            value=modelsOptionsList[0]['value'],
            placeholder='Select a model',
            multi=True
        ),
        dcc.Graph(id='output-fitting-graph'),
    ]),

    # ARRHENIUS PLOT WITH THE KINETIC TRIPLET

    html.Div([
        html.Div(id='output-arrhenius-graph-html',children='choose your model to predict the rate constant over temperature'),
        dcc.Dropdown(
            id='models-dropdown-arrhenius',
            options=modelsArrheniusOptionsList,
            value=modelsArrheniusOptionsList[0]['value'],
            placeholder='Select a model',
            multi=False
        ),
        dcc.Graph(id='output-arrhenius-graph'),
        html.Div(id='output-arrhenius-info')
    ]),

    # ARRHENIUS PLOT WITH THE KINETIC TRIPLET

    html.Div([
        dcc.Graph(id='output-isoconversional-graph')
    ])

])

# callback for the conversion range slider
@app.callback(
    dash.dependencies.Output('input-scatter-plot', 'figure'),
    [dash.dependencies.Input('input-data-range-slider', 'value')])
def update_input_graph(slider_range):
    low, high = slider_range
    
    # filter the datafile based on the conversion range
    data = []

    # load all datafiles
    for case in Data:

        # read a data file
        conversion, time, temperature = read_filtrated_datafile(case,low,high)
        # save the trace
        trace = go.Scatter(
            x = time,
            y = conversion,
            mode = 'markers',
            name = str(temperature) + tempUnits,
            hovertemplate='conversion: %{y:.2f}'
        )
        data.append(trace)
    
    xlabel = 'time [' + timeUnits + ']'
    figure={
            'data': data,
            'layout': go.Layout(
                title = 'input data',
                xaxis = {'title': xlabel},
                yaxis = {'title': 'conversion'}
            )         
    }

    return figure

# callback for the convergence bar graph
@app.callback(
    dash.dependencies.Output('output-congergence-graph', 'figure'),
    [dash.dependencies.Input('input-data-range-slider', 'value'),
     dash.dependencies.Input('files-dropdown','value')])
def regression(slider_range,filename):
    
    # get the edge values from the slider
    low, high = slider_range

    # load the particular csv as chosen by the files-dropdown
    case = filename

    # read a data file
    conversion, time, temperature = read_filtrated_datafile(case,low,high)

    # perform non-linear regression and return the fitting information
    df = comprehensiveRegressor(time, conversion, modelNames)

    # calculate the convergence criterion
    data = convergenceData(df)

    fig = px.bar(data, x='model', y='fitting error', hover_data={'model':True,'fitting error':':.2f'})

    return fig

# callback for the fitting line chart
@app.callback(
    dash.dependencies.Output('output-fitting-graph', 'figure'),
    [dash.dependencies.Input('input-data-range-slider', 'value'),
     dash.dependencies.Input('files-dropdown','value'),
     dash.dependencies.Input('models-dropdown','value')])
def updateModelFitting(slider_range,filename,modelnames):

    # get the edge values from the slider
    low, high = slider_range

    # load the particular csv as chosen by the files-dropdown
    case = filename

    # read a data file
    conversion, time, temperature = read_filtrated_datafile(case,low,high)
    
    # add the input data trace
    trace = go.Scatter(x=time, y=conversion, mode='markers', name = str(temperature) + 'K')
    data = [trace]

    if not isinstance(modelnames,list):
        modelnames = [modelnames] # because modelnames has to be a list for the comprehensiveRegressor call
    
    # perform non-linear regression and return the fitting information
    df = comprehensiveRegressor(time, conversion, modelnames)

    # define a more continuous time fot the fitting models
    time = np.linspace(np.min(time),np.max(time),2000)

    # create a datafroma containing the time and the solution for each
    # model
    for indx, k in enumerate(df['rate_constant - alpha']):
        # pick up a single model
        model = Model(df['model'][indx])
        # calculate the specific conversion
        yfit  = np.array([model.alpha(t,k) for t in time])
        # add the particular fit to the figure
        trace = go.Scatter(x=time, y=yfit, mode='lines', name=df['model'][indx])
        data.append(trace)

    xlabel = 'time [' + timeUnits + ']'
    figure={
        'data': data,
        'layout': go.Layout(
            title = 'model fitting',
            xaxis = {'title': xlabel},
            yaxis = {'title': 'conversion'}
        )
    }
        
    return figure

# callback for the arrhenius chart
@app.callback(
    [
        dash.dependencies.Output('output-arrhenius-graph', 'figure'),
        dash.dependencies.Output('output-arrhenius-info', 'children')
    ],
    [
        dash.dependencies.Input('input-data-range-slider', 'value'),
        dash.dependencies.Input('models-dropdown-arrhenius','value')
    ]
    )
def updateArrheniusFitting(slider_range,modelname):

    # get the edge values from the slider
    low, high = slider_range

    # load all datafiles and perform regression using the model chosen by the dropdown
    x = []
    y = []

    for case in Data:

        # read a data file
        conversion, time, temperature = read_filtrated_datafile(case,low,high)

        if not isinstance(modelname,list):
            modelname = [modelname] # because modelnames has to be a list for the comprehensiveRegressor call
    
        # perform non-linear regression and return the fitting information
        df = comprehensiveRegressor(time, conversion, modelname)

        if modelname in ['D2','D4']:
            k = df['rate_constant - integral'].values[0]
        else:
            k = df['rate_constant - alpha'].values[0]

        y.append(k)
        x.append(temperature)

    x = np.array(x) # temperature
    y = np.array(y) # rate constant

    activation, A = arrheniusEnthalpy(y,x)

    # compute a continuous fit
    temperature = np.linspace(np.min(x),np.max(x),2000)
    temperature = np.array(temperature)

    k_fit = [rateConstant(A,activation[0],T) for T in temperature]
    k_fit = np.array(k_fit)

    data = []
    trace = go.Scatter(x=x, y=y, mode='markers', name=modelname[0]+' prediction')
    data.append(trace)
    trace = go.Scatter(x=temperature, y=k_fit, mode='lines', name='Arrhenius fit')
    data.append(trace)

    xlabel = 'T [' + tempUnits + ']'
    figure={
        'data': data,
        'layout': go.Layout(
            title = 'Arrhenius validation',
            xaxis = {'title': xlabel},
            yaxis = {'title': 'rate constant'}
        )
    }

    # display text
    displayText = 'The activation enthalpy is' + str(round(activation[0]*1.0e-3,2)) + 'kJ mol-1'
        
    return figure, displayText

# callback for the isoconversional method
@app.callback(
    dash.dependencies.Output('output-isoconversional-graph', 'figure'),
    [dash.dependencies.Input('input-data-range-slider', 'value')])
def update_input_graph(slider_range):
    # select conversion range
    low, high = slider_range
    
    # determine the number of interpolated points
    npoints = 10

    # compute npoints of interpolated conversion
    interConversion = np.linspace(low,high,npoints)

    # initialize the dictionary containing the interpolated conversion
    isocData = {'interConversion': interConversion}

    for case in Data:
        # read a data file given the conversion range from the dropdown
        conversion, time, temperature = read_filtrated_datafile(case,low,high)
        # calculate the interpolated time
        interTime                     = interpolateTime(time,conversion,interConversion)
        # get the temperature as a header and the time as column values for each data set
        isocData.update({str(temperature):interTime})

    # interConversion | time for temperature 1 | time for temperature 2 ...
    df = pd.DataFrame(isocData)

    # prepare data for linear regression
    temperature = [float(i) for i in df.columns.values if i != 'interConversion']
    temperature = np.array(temperature)

    Ea        = [] # Activation energy (kJ/mol)
    intercept = [] # Intercept ln[A g(a)]
    MSE       = [] # Standard deviation
    R2        = [] # Determination coefficient

    # the size of the dataframe
    dfSize = df.shape[0]
    for i in range(dfSize):
        # get the time for each temperature set
        time = df.iloc[i,1::].to_numpy()
        # perform linear regression
        activation, gA = isocEnthalpy(time,temperature)
        Ea.append(activation[0])
        MSE.append(activation[1])
        intercept.append(gA)
    
    isocFitData = {'activation enthalpy': Ea, 'std': MSE, 'intercept': intercept, 'conversion': df['interConversion'].to_numpy()}
    df = pd.DataFrame(isocFitData)

    # get the graph of the activation energy versus conversion
    figure = px.bar(df, x="conversion", y="activation enthalpy", color="activation enthalpy", error_y="std", labels={'activation enthalpy':'activation enthalpy (kJ/mol)'})

    return figure

if __name__ == '__main__':
    app.run_server(debug=True)