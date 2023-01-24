# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import math
from random import random

import numpy
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import os
import numpy as np
from dash import Dash, html, dcc,Input, Output, callback, dash_table
import dash_bootstrap_components as dbc
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import datetime
from scipy import signal


def test():
    df = pd.DataFrame([{'row': 'A',
                        'start_time': '2020-08-04 06:00:01',
                        'end_time': '2020-08-04 06:06:01',
                        'status': 'succeeded'},
                       {'row': 'A',
                        'start_time': '2020-08-04 07:00:01',
                        'end_time': '2020-08-04 07:05:01',
                        'status': 'failed'},
                       {'row': 'A',
                        'start_time': '2020-08-04 08:00:01',
                        'end_time': '2020-08-04 08:06:01',
                        'status': 'succeeded'},
                       {'row': 'B',
                        'start_time': '2020-08-04 06:44:11',
                        'end_time': '2020-08-04 06:53:22',
                        'status': 'succeeded'},
                       {'row': 'B',
                        'start_time': '2020-08-04 07:01:58',
                        'end_time': '2020-08-04 07:07:48',
                        'status': 'succeeded'},
                       {'row': 'C',
                        'start_time': '2020-08-04 06:38:56',
                        'end_time': '2020-08-04 06:44:59',
                        'status': 'succeeded'},
                       {'row': 'C',
                        'start_time': '2020-08-04 06:59:00',
                        'end_time': '2020-08-04 07:05:00',
                        'status': 'failed'}])
    print(df)
    fig = px.timeline(df, x_start="start_time", x_end="end_time", y="row", color="status")
    fig.show()


def read_list(name, name2):
    # READ IN ALL OF DEPOT ONE's INFO
    list = []
    f = open(name)
    line = f.readline()
    line = f.readline()
    line = f.readline()
    line = f.readline()
    line = f.readline()
    line = f.readline()
    while line:
        try:
            list.append(line)
        except ValueError:
            print('Error in line :' + line)
        line = f.readline()
    df = pd.DataFrame([sub.split() for sub in list])
    logFrame = pd.DataFrame([sub.split("-") for sub in df[3]])
    df2 = logFrame[pd.to_numeric(logFrame[2], errors='coerce').notnull()]
    df2.columns = ['id', 'vehicle', 'start', 'end', 'ACTIVITY']
    df3 = df2.copy()
    df3['start'] = df2['start'].astype(float)
    df3['end'] = df2['end'].astype(float)
    # ---------

    # READ IN ALL OF DEPOT 2's info
    list = []
    f = open(name2)
    line = f.readline()
    line = f.readline()
    line = f.readline()
    line = f.readline()
    line = f.readline()
    line = f.readline()
    while line:
        try:
            list.append(line)
        except ValueError:
            print('Error in line :' + line)
        line = f.readline()
    df = pd.DataFrame([sub.split() for sub in list])
    logFrame = pd.DataFrame([sub.split("-") for sub in df[3]])
    df2 = logFrame[pd.to_numeric(logFrame[2], errors='coerce').notnull()]
    df2.columns = ['id', 'vehicle', 'start', 'end', 'ACTIVITY']
    df4 = df2.copy()
    df4['start'] = df2['start'].astype(float)
    df4['end'] = df2['end'].astype(float)
    # ----------------------
    # print(df3)
    # print(df4)
    deltaFrame = pd.concat([df3, df4])
    # SUBTRACT THE EARLIEST MOOSTIME ENTRY TO MAKE EVERYTHING REFERENCED FROM ZERO
    deltaFrame['start1'] = deltaFrame['start'] - 33296827658
    deltaFrame['end1'] = deltaFrame['end'] - 33296827658

    # TRIM AND RESET INDEX OF CONCATENATED LIST
    nf = deltaFrame[['vehicle', 'start1', 'end1', 'ACTIVITY']].copy()
    nf.reset_index(drop=True, inplace=True)
    # Convert our start and end times to datetimes so that they can be correctly read by plotly
    nf['end1'] = pd.to_datetime(nf['end1'], unit='s')
    nf['start1'] = pd.to_datetime(nf['start1'], unit='s')
    fig = px.timeline(
        nf, x_start="start1", x_end="end1", y="vehicle",

        color='ACTIVITY'

    )
    # fig.show()
    return fig


# def generateFuelTimeHistogram():
#     directory="/Users/ericyoung/moos-ivp-younge/missions/ufld_saxis/testLogFolder"
#     logLines = []
#     for filename in os.listdir(directory):
#         name = os.path.join(directory, filename)
#
#         # checking if it is a file
#         if not filename.startswith('.') and os.path.isfile(name):
#             #chomp the first header lines of the log file
#             f = open(name)
#             line = f.readline()
#             line = f.readline()
#             line = f.readline()
#             line = f.readline()
#             line = f.readline()
#             while line:
#                 try:
#                     line=f.readline()
#                     #print(line)
#                     logLines.append(line)
#                 except ValueError:
#                     print('Error in line :' + line)
#     rawLogData = pd.DataFrame([sub.split() for sub in logLines])
#     #print(rawLogData[3])
#     splitWaitLogs=rawLogData[3].str.split(pat='-',expand =True)
#     finalWaitTimes=splitWaitLogs[pd.to_numeric(splitWaitLogs[1], errors='coerce').notnull()]
#     finalWaitTimes.columns=['vehicle','time']
#     finalWaitTimes['time']=finalWaitTimes['time'].astype(float)
#     finalWaitTimes.sort_values(by=['time'],ascending=True)
#     fig = px.histogram(finalWaitTimes, x="time")
#     return fig
#     #fig.show()
#     #print(splitWaitLogs[1])
#     #print(finalWaitTimes)
#     #df2 = rawLogData[pd.to_numeric(rawLogData[3], errors='coerce').notnull()]

def generateFuelLineGraph():
    directory = "/Users/ericyoung/moos-ivp-younge/missions/ufld_saxis/testLogFolder2"
    vehicleFrames = []
    logLines = []
    fig = go.Figure()
    for filename in os.listdir(directory):
        name = os.path.join(directory, filename)

        # checking if it is a file
        if not filename.startswith('.') and os.path.isfile(name):
            # print(filename)
            f = open(name)
            line = f.readline()
            line = f.readline()
            line = f.readline()
            line = f.readline()
            line = f.readline()
            while line:
                try:
                    line = f.readline()
                    # print(line)
                    logLines.append(line)
                except ValueError:
                    print('Error in line :' + line)
            # create a list of dataframes that hold the "-" separated fuel log data for every vehicle
            vehicleFrames.append(pd.DataFrame([sub.split() for sub in logLines])[3])
            logLines = []
    # split the fuel log data by "-"
    tempProcessedFuelData = []
    for frame in vehicleFrames:
        tempProcessedFuelData.append(frame.str.split(pat='-', expand=True))
    # remove None rows from the fuel data
    dashSegmentedFuelData = []
    for frame in tempProcessedFuelData:
        dashSegmentedFuelData.append(frame.replace(to_replace='None', value=np.nan).dropna())
    finalSegmentedFuelData = []
    fig = go.Figure()
    for frame in dashSegmentedFuelData:
        # convert fuel and time values to numbers
        frame[1] = frame[1].astype(float)
        frame[2] = frame[2].astype(float)
        # name each column in the dataframe
        frame.columns = ['vehicle', 'level', 'time']
        # remove all starting values with values greater than the fuel tank size (this is for testing only)
        tempData = frame[(frame['level'] <= 200)]
        # reset the dataframe index to make up for the values just removed
        tempData.reset_index(inplace=True)
        # zero out the time data based on the earliest value and convert to a datetime object

        tempData['time'] = tempData['time'] - tempData['time'][0]
        tempData['time'] = pd.to_datetime(tempData['time'], unit='s')
        finalSegmentedFuelData.append(tempData)
        fig.add_trace(go.Scatter(x=tempData['time'], y=tempData['level'],
                                 mode='lines',
                                 name=tempData['vehicle'][0]))
    return fig
    # fig.show()
    # print(finalSegmentedFuelData[0])
    # print((dashSegmentedFuelData[0]['level']<=200))

    # print(dashSegmentedFuelData[0][(dashSegmentedFuelData[0]['level']<=200)])


def generateDepotTimelines(directory) -> object:
    depotData = pd.DataFrame(columns=["dname", 'status', 'st', "et"])

    # iterate over files in
    # that directory
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        # checking if it is a file
        if os.path.isfile(f):
            if "DEPOT" in filename:
                # read the depot pickle file and concat it to the depot data frame
                depotData = pd.concat([depotData, pd.read_pickle(f)])
    # set variable min time equal to the earliest start time in the depot data
    minTime = depotData['st'].min()
    # find the minimum in the start time column and subtract it from all values
    depotData['st'] = depotData['st'] - depotData['st'].min()
    # subtract mintTime from the end time column
    depotData['et'] = depotData['et'] - minTime
    # convert the start and end time columns to datetime objects
    depotData['st'] = pd.to_datetime(depotData['st'], unit='s')
    depotData['et'] = pd.to_datetime(depotData['et'], unit='s')
    fig = px.timeline(depotData, x_start="st", x_end="et", y="dname", color='status')
    return fig


# create a function that will generate a line graph of the fuel levels of all vehicles
def generateFuelLineGraph(directory,syncTime):
    vehicleFrames = []
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        # checking if it is a file
        if os.path.isfile(f):
            if "DEPOT" not in filename and "SHORESIDE" not in filename and "fuelTime" in filename:
                # read in the pickle file, create a dataframe, and append it to the list of vehicle dataframes
                vehicleFrame = pd.read_pickle(f)
                #syncTimes with all other moos logs
                vehicleFrame = vehicleFrame.loc[vehicleFrame['time'] > syncTime]
                #reset index
                vehicleFrame.reset_index(inplace=True)
                # subtract the earliest time from the time column
                vehicleFrame['time'] = vehicleFrame['time'] - vehicleFrame['time'].min()
                # convert the time column to a datetime object
                vehicleFrame['time'] = pd.to_datetime(vehicleFrame['time'], unit='s')
                vehicleFrames.append(vehicleFrame)
        # create a scatter plot of the fuel levels of all vehicles
    fig = go.Figure()
    for frame in vehicleFrames:
        fig.add_trace(go.Scatter(x=frame['time'], y=frame['level'], mode='lines', name=frame['vname'][0]))
    return fig

# create a function that will generate a line graph of the fuel levels of all vehicles
def generateEntropyLineGraph(directory,syncTime):
    entropyFrames = []
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        # checking if it is a file
        if os.path.isfile(f):
            if "SHORESIDE" in filename and "totalEntropy" in filename:
                # read in the pickle file, create a dataframe, and append it to the list of vehicle dataframes
                shoreFrame = pd.read_pickle(f)
                #syncTimes with all other moos logs
                shoreFrame = shoreFrame.loc[shoreFrame['time'] > syncTime]
                # subtract the earliest time from the time column
                shoreFrame['time'] = shoreFrame['time'] - shoreFrame['time'].min()
                # convert the time column to a datetime object
                shoreFrame['time'] = pd.to_datetime(shoreFrame['time'], unit='s')
                entropyFrames.append(shoreFrame)
        # create a scatter plot of the fuel levels of all vehicles
    fig = go.Figure()
    for frame in entropyFrames:
        fig.add_trace(go.Scatter(x=frame['time'], y=frame['entropy'], mode='lines', name="Shoreside Global Entropy"))
        #add a trace with a Moving Average filter
        fig.add_trace(go.Scatter(x=frame['time'], y=frame['entropy'].rolling(window=150).mean(), mode='lines', name="Shoreside Global Entropy (5m MA)"))
    return fig
def generateAreaStatsLineGraph(directory,syncTime):
    entropyFrames = []
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        # checking if it is a file
        if os.path.isfile(f):
            if "SHORESIDE" in filename and "areaStats" in filename:
                # read in the pickle file, create a dataframe, and append it to the list of vehicle dataframes
                shoreFrame = pd.read_pickle(f)
                #syncTimes with all other moos logs
                shoreFrame = shoreFrame.loc[shoreFrame['time'] > syncTime]
                # subtract the earliest time from the time column
                shoreFrame['time'] = shoreFrame['time'] - shoreFrame['time'].min()
                # convert the time column to a datetime object
                shoreFrame['time'] = pd.to_datetime(shoreFrame['time'], unit='s')
                entropyFrames.append(shoreFrame)
        # create a scatter plot of the fuel levels of all vehicles
    fig = go.Figure()
    for frame in entropyFrames:
        fig.add_trace(go.Scatter(x=frame['time'], y=frame['avgArea'], mode='lines', name="Average Area"))
        fig.add_trace(go.Scatter(x=frame['time'], y=frame['maxArea'], mode='lines', name="Maximum Area"))
        fig.add_trace(go.Scatter(x=frame['time'], y=frame['minArea'], mode='lines', name="Minimum Area"))
        fig.add_trace(go.Scatter(x=frame['time'], y=frame['stdDevArea'], mode='lines', name="StDev Area"))
    return fig
def generateLatencyAverageLineGraph(directory,syncTime):
    latencyFrames = []
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        # checking if it is a file
        if os.path.isfile(f):
            if "averageLatency" in filename:
                # read in the pickle file, create a dataframe, and append it to the list of vehicle dataframes
                latencyFrame = pd.read_pickle(f)

                #syncTimes with all other moos logs
                latencyFrame = latencyFrame.loc[latencyFrame['time'] > syncTime]
                # reset index
                latencyFrame.reset_index(inplace=True)

                # subtract the earliest time from the time column
                latencyFrame['time'] = latencyFrame['time'] - latencyFrame['time'].min()
                # convert the time column to a datetime object
                latencyFrame['time'] = pd.to_datetime(latencyFrame['time'], unit='s')
                latencyFrames.append(latencyFrame)
        # create a scatter plot of the fuel levels of all vehicles
    fig = go.Figure()
    for frame in latencyFrames:
        fig.add_trace(go.Scatter(x=frame['time'], y=frame['average'], mode='lines', name=frame['vname'][0]))

    return fig
def computeShoresideLatencyAverage(directory,syncTime):
    latencyFrames = []
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        # checking if it is a file
        if os.path.isfile(f):
            if "averageLatency" in filename:
                # read in the pickle file, create a dataframe, and append it to the list of vehicle dataframes
                latencyFrame = pd.read_pickle(f)

                #syncTimes with all other moos logs
                latencyFrame = latencyFrame.loc[latencyFrame['time'] > syncTime]
                # reset index
                latencyFrame.reset_index(inplace=True)

                # subtract the earliest time from the time column
                latencyFrame['time'] = latencyFrame['time'] - latencyFrame['time'].min()
                # convert the time column to a datetime object
                latencyFrame['time'] = pd.to_datetime(latencyFrame['time'], unit='s')
                latencyFrames.append(latencyFrame)

    for frame in latencyFrames:
        return frame['average'].mean()

    return None
def computeAreaAverage(directory,syncTime):
    areaFrames = []
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        # checking if it is a file
        if os.path.isfile(f):
            if "SHORESIDE" in filename and "areaStats" in filename:
                # read in the pickle file, create a dataframe, and append it to the list of vehicle dataframes
                shoreFrame = pd.read_pickle(f)
                #syncTimes with all other moos logs
                shoreFrame = shoreFrame.loc[shoreFrame['time'] > syncTime]
                # subtract the earliest time from the time column
                shoreFrame['time'] = shoreFrame['time'] - shoreFrame['time'].min()
                # convert the time column to a datetime object
                shoreFrame['time'] = pd.to_datetime(shoreFrame['time'], unit='s')
                areaFrames.append(shoreFrame)

    for frame in areaFrames:
        return frame['avgArea'].mean()

    return None
def generateVehicleFuelingCountLineGraph(directory,syncTime):
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        # checking if it is a file
        if os.path.isfile(f):
            if "vehicleFuelingCount" in filename:
                # read in the pickle file, create a dataframe
                frame = pd.read_pickle(f)
                #syncTimes with all other moos logs
                frame = frame.loc[frame['time'] > syncTime]

                # subtract the earliest time from the time column
                frame['time'] = frame['time'] - frame['time'].min()
                # convert the time column to a datetime object
                frame['time'] = pd.to_datetime(frame['time'], unit='s')
                frame.append(frame)
        # create a scatter plot of the count of all vehicles currently fueling
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=frame['time'], y=frame['count'], mode='lines', name="Number of Vehicles Fueling"))
    return fig


def generateFuelTimeHistogram(directory):
    vehicleFrames = []
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        # checking if it is a file
        if os.path.isfile(f) and "wait" in filename:
            # read in the pickle file, create a dataframe, and append it to the list of vehicle dataframes
            vehicleFrame = pd.read_pickle(f)
            vehicleFrames.append(vehicleFrame)
    # concatenate all the dataframes into one dataframe called allVehicleData
    allVehicleData = pd.concat(vehicleFrames)
    fig = px.histogram(allVehicleData, x="time")
    # fig.show()
    return fig
def findTimeVehiclesStabilized(directory,):
    threshold=7000
    stabilizedTimes= []
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        # checking if it is a file
        if os.path.isfile(f):
            if "DEPOT" not in filename and "SHORESIDE" not in filename and "fuelTime" in filename:
                # In this case, we just want to see the first time when a vehcles fuel level is below the threshold. This means that the simulation has stabilized and given the ships real fuel levels
                vehicleFrame = pd.read_pickle(f)
                vehicleFrame=vehicleFrame.loc[vehicleFrame['level'] < threshold]
                stabilizedTimes.append( vehicleFrame['time'].min())
    return max(stabilizedTimes)
def generateSyncTime(directory):
    #in this case, we need to read in all the pickled dataframes, find the minumum time in each frame, and then find the maximum of those minimum times
    minTimes=[]
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        # checking if it is a file
        if os.path.isfile(f):
            if "LOG" in filename:
                frame = pd.read_pickle(f)
                minTimes.append(frame['time'].min())
    return max(minTimes)

def generateAverageIdleTimeDataFrame(directory):
    #create a set of all tank sizes
    tankSizes = set()
    #create a set of all depot numbers
    depotNumbers = set()
    #determine all the unique tank sizes and depot numbers
    for run in os.listdir(directory):
        if "tank_" in run:
            #split the run name using "-" as the delimiter into number of depots and tank size
            depotNum=run.split("-")[0].split("_")[1]
            tankSize=run.split("-")[1].split("_")[1]
            #add the tank size and depot number to the sets
            tankSizes.add(tankSize)
            depotNumbers.add(depotNum)
    #sort the tank sizes and depot numbers
    tankSizes=sorted(tankSizes)
    depotNumbers=sorted(depotNumbers)
    #create a pandas datafrome using tank sizes as columhs and depot numbers as rows
    df = pd.DataFrame(index=depotNumbers, columns=tankSizes)
    #create column in the dataframe for the number of depots
    df.insert(0, "Number of Depots", depotNumbers)
    #iterate through the tank sizes and depot numbers in the dataframe and set the value to a random number between zero and 1
    for tankSize in tankSizes:
        for depotNum in depotNumbers:
            runDirectory=os.path.join(directory, "depots_"+str(depotNum)+"-tank_"+str(tankSize))
            syncTime=findTimeVehiclesStabilized(runDirectory)
            df[tankSize][depotNum]= round(computeShoresideLatencyAverage(runDirectory, syncTime),2)
    return df
def generateAverageAreaDataFrame(directory):
    #create a set of all tank sizes
    tankSizes = set()
    #create a set of all depot numbers
    depotNumbers = set()
    #determine all the unique tank sizes and depot numbers
    for run in os.listdir(directory):
        if "tank_" in run:
            #split the run name using "-" as the delimiter into number of depots and tank size
            depotNum=run.split("-")[0].split("_")[1]
            tankSize=run.split("-")[1].split("_")[1]
            #add the tank size and depot number to the sets
            tankSizes.add(tankSize)
            depotNumbers.add(depotNum)
    #sort the tank sizes and depot numbers
    tankSizes=sorted(tankSizes)
    depotNumbers=sorted(depotNumbers)
    #create a pandas datafrome using tank sizes as columhs and depot numbers as rows
    df = pd.DataFrame(index=depotNumbers, columns=tankSizes)
    #create column in the dataframe for the number of depots
    df.insert(0, "Number of Depots", depotNumbers)
    #iterate through the tank sizes and depot numbers in the dataframe and set the value to a random number between zero and 1
    for tankSize in tankSizes:
        for depotNum in depotNumbers:
            runDirectory=os.path.join(directory, "depots_"+str(depotNum)+"-tank_"+str(tankSize))
            syncTime=findTimeVehiclesStabilized(runDirectory)
            df[tankSize][depotNum]= round(computeAreaAverage(runDirectory, syncTime),2)
    return df
def generateSummaryDataFrame(directory):
    #create a set of all tank sizes
    tankSizes = set()
    #create a set of all depot numbers
    depotNumbers = set()
    #determine all the unique tank sizes and depot numbers
    for run in os.listdir(directory):
        if "tank_" in run:
            #split the run name using "-" as the delimiter into number of depots and tank size
            depotNum=run.split("-")[0].split("_")[1]
            tankSize=run.split("-")[1].split("_")[1]
            #add the tank size and depot number to the sets
            tankSizes.add(tankSize)
            depotNumbers.add(depotNum)
    #sort the tank sizes and depot numbers
    tankSizes=sorted(tankSizes)
    depotNumbers=sorted(depotNumbers)
    #create a pandas datafrome using tank sizes as columhs and depot numbers as rows
    df = pd.DataFrame(index=depotNumbers, columns=tankSizes)
    #create column in the dataframe for the number of depots
    df.insert(0, "Number of Depots", depotNumbers)
    areaDF=generateAverageAreaDataFrame(directory)
    idleDF=generateAverageIdleTimeDataFrame(directory)
    #iterate through the tank sizes and depot numbers in the dataframe and set the to the string concatination of the other area and idle time dataframes
    for tankSize in tankSizes:
        for depotNum in depotNumbers:
            runDirectory=os.path.join(directory, "depots_"+str(depotNum)+"-tank_"+str(tankSize))
            syncTime=findTimeVehiclesStabilized(runDirectory)
            df[tankSize][depotNum]= str(areaDF[tankSize][depotNum]) + " / " + str(idleDF[tankSize][depotNum])
    return df
def generateDashboard(depotDirectory, vehicleDirectory):
    # fig1=generateDepotTimelines(depotDirectory)
    #Generate default figures to be updated by the callback
    fig2= go.Figure()
    fig3= go.Figure()
    fig4= go.Figure()
    fig5= go.Figure()
    fig6= go.Figure()
    fig7= go.Figure()
    fig8= go.Figure()

    #syncTime=generateSyncTime(depotDirectory)

    df=generateSummaryDataFrame(vehicleDirectory)

    app = Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])
    colors = {
      'background': '#111111',
       'text': '#7FDBFF'
    }

        # f = os.path.join(vehicleDirectory, filename)
        # # checking if it is a file
        # if os.path.isfile(f):
        #     if "" in filename:
        #         frame = pd.read_pickle(f)
        #         minTimes.append(frame['time'].min())
    # syncTime=findTimeVehiclesStabilized(vehicleDirectory)
    # fig2=generateFuelLineGraph(vehicleDirectory,syncTime)
    # # fig3=generateFuelTimeHistogram(vehicleDirectory)
    # fig4 = generateIdleTimeHistogram(vehicleDirectory,syncTime)
    # fig5 = generateLatencyAverageLineGraph(vehicleDirectory,syncTime)
    # fig6 = generateEntropyLineGraph(vehicleDirectory,syncTime)
    # fig7 = generateAreaStatsLineGraph(vehicleDirectory,syncTime)
    # fig8 = generateVehicleFuelingCountLineGraph(vehicleDirectory,syncTime)
    #

    # fig2.update_layout(
    #     plot_bgcolor=colors['background'],
    #     paper_bgcolor=colors['background'],
    #     font_color=colors['text']
    # )
    # fig6.update_layout(
    #     plot_bgcolor=colors['background'],
    #     paper_bgcolor=colors['background'],
    #     font_color=colors['text']
    # )
    # fig7.update_layout(
    #     plot_bgcolor=colors['background'],
    #     paper_bgcolor=colors['background'],
    #     font_color=colors['text']
    # )
    # fig8.update_layout(
    #     plot_bgcolor=colors['background'],
    #     paper_bgcolor=colors['background'],
    #     font_color=colors['text']
    # )
    # # fig1.update_layout(
    # #     plot_bgcolor=colors['background'],
    # #     paper_bgcolor=colors['background'],
    # #     font_color=colors['text']
    # # )
    # # fig3.update_layout(
    # #     plot_bgcolor=colors['background'],
    # #     paper_bgcolor=colors['background'],
    # #     font_color=colors['text']
    # # )
    # fig4.update_layout(
    #     plot_bgcolor=colors['background'],
    #     paper_bgcolor=colors['background'],
    #     font_color=colors['text']
    # )
    # fig5.update_layout(
    #     plot_bgcolor=colors['background'],
    #     paper_bgcolor=colors['background'],
    #     font_color=colors['text']
    # )
    app.layout = html.Div(style={'backgroundColor': colors['background']}, children=[
        html.H1(children='Ship Supply Dashboard Visualization Tool', style={
            'textAlign': 'center',
            'color': colors['text']
        }),dbc.Container([
    dbc.Label('Click a cell in the table:', style={
            'textAlign': 'center',
            'color': colors['text']
        }),
    dash_table.DataTable(df.to_dict('records'),[{"name": i, "id": i} for i in df.columns], id='tbl',style_data={
        'backgroundColor': colors['background'],
        'color': colors['text']
    },style_header={
        'backgroundColor': 'rgb(30, 30, 30)',
        'color': colors['text']
    }),
    dbc.Alert(id='tbl_out')

]),
    #
    #     html.Div(children='''
    #         Run: 10/5/22 at 2100
    #     ''', style={
    #         'textAlign': 'center',
    #         'color': colors['text']
    #     }),
    #     #
    #     # dcc.Graph(
    #     #     id='Fueling Histogram',
    #     #     figure=fig1
    #     # ),
        dcc.Graph(
            id='fueling-timeline',
            figure=fig2
        ),
        #     dcc.Graph(
        #     id='entropy-timeline',
        #     figure=fig6
        # ),
        dcc.Graph(
            id='fueling-count-timeline',
            figure=fig8
        ),dcc.Graph(
            id='area-stats-timeline',
            figure=fig7
        # ),

        # , d
        # cc.Graph(
        #     id='Fuel History',
        #     figure=fig3
        ), dcc.Graph(
            id='idle-time-histogram',
            figure=fig4
        ),
        dcc.Graph(
            id='latency-average-timeline',
            figure=fig5
        )
    #
    ])

    @callback(Output('tbl_out', 'children'), Input('tbl', 'active_cell'))
    def update_graphs(active_cell):
        return str(active_cell) if active_cell else "Click the table"
    @callback(Output('fueling-timeline', 'figure'), Input('tbl', 'active_cell'))
    def update_graphs(active_cell):
        active_col_id = active_cell['column_id'] if active_cell else None
        active_row_id = active_cell['row']+1 if active_cell else None
        fig= go.Figure()
        if active_cell:
            runDirectory =  os.path.join(vehicleDirectory, "depots_"+str(active_row_id)+"-tank_"+str(active_col_id))
            syncTime=findTimeVehiclesStabilized(runDirectory)
            fig=generateFuelLineGraph(runDirectory,syncTime)

            fig.update_layout(
                plot_bgcolor=colors['background'],
                paper_bgcolor=colors['background'],
                font_color=colors['text']
            )
        return fig
    @callback(Output('idle-time-histogram', 'figure'), Input('tbl', 'active_cell'))
    def update_graphs(active_cell):
        active_col_id = active_cell['column_id'] if active_cell else None
        active_row_id = active_cell['row']+1 if active_cell else None
        fig= go.Figure()
        if active_cell:
            runDirectory =  os.path.join(vehicleDirectory, "depots_"+str(active_row_id)+"-tank_"+str(active_col_id))
            syncTime=findTimeVehiclesStabilized(runDirectory)

            fig = generateIdleTimeHistogram(runDirectory,syncTime)

            fig.update_layout(
                plot_bgcolor=colors['background'],
                paper_bgcolor=colors['background'],
                font_color=colors['text']
            )
        return fig
    @callback(Output('latency-average-timeline', 'figure'), Input('tbl', 'active_cell'))
    def update_graphs(active_cell):
        active_col_id = active_cell['column_id'] if active_cell else None
        active_row_id = active_cell['row']+1 if active_cell else None
        fig= go.Figure()
        if active_cell:
            runDirectory =  os.path.join(vehicleDirectory, "depots_"+str(active_row_id)+"-tank_"+str(active_col_id))
            syncTime=findTimeVehiclesStabilized(runDirectory)

            fig = generateLatencyAverageLineGraph(runDirectory,syncTime)

            fig.update_layout(
                plot_bgcolor=colors['background'],
                paper_bgcolor=colors['background'],
                font_color=colors['text']
            )
        return fig
    @callback(Output('area-stats-timeline', 'figure'), Input('tbl', 'active_cell'))
    def update_graphs(active_cell):
        active_col_id = active_cell['column_id'] if active_cell else None
        active_row_id = active_cell['row']+1 if active_cell else None
        fig= go.Figure()
        if active_cell:
            runDirectory =  os.path.join(vehicleDirectory, "depots_"+str(active_row_id)+"-tank_"+str(active_col_id))
            syncTime=findTimeVehiclesStabilized(runDirectory)

            fig = generateAreaStatsLineGraph(runDirectory,syncTime)

            fig.update_layout(
                plot_bgcolor=colors['background'],
                paper_bgcolor=colors['background'],
                font_color=colors['text']
            )
        return fig
    @callback(Output('fueling-count-timeline', 'figure'), Input('tbl', 'active_cell'))
    def update_graphs(active_cell):
        active_col_id = active_cell['column_id'] if active_cell else None
        active_row_id = active_cell['row']+1 if active_cell else None
        fig= go.Figure()
        if active_cell:
            runDirectory =  os.path.join(vehicleDirectory, "depots_"+str(active_row_id)+"-tank_"+str(active_col_id))
            syncTime=findTimeVehiclesStabilized(runDirectory)

            fig = generateVehicleFuelingCountLineGraph(runDirectory,syncTime)
            fig.update_layout(
                plot_bgcolor=colors['background'],
                paper_bgcolor=colors['background'],
                font_color=colors['text']
            )
        return fig

    app.run_server(debug=True)


def processLatencyCounts(directory):
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        # checking if it is a file
        if os.path.isfile(f) and "XLOG" in filename:
            # read in a pickle file and assign it to a dataframe called latencyData
            latencyData = pd.read_pickle(f)

    # create a histogram of the latency data
    fig = px.histogram(latencyData, x="counts", y="percent")
    return fig
def generateIdleTimeHistogram(directory,syncTime):
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        # checking if it is a file
        if os.path.isfile(f) and "averageLatency" in filename:
            # read in a pickle file and assign it to a dataframe called latencyData
            latencyData = pd.read_pickle(f)
            #remove all the data before the sync time
            latencyData=latencyData.loc[latencyData['time'] > syncTime]

    # create a histogram of the latency data
    fig = px.histogram(latencyData, x="average")
    return fig

if __name__ == '__main__':
    import sys
    import os
    import argparse

    # CONFIGURE ARGUMENT PARSER
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--examples", help="Display example uses", action="store_true")
    parser.add_argument("-s", "--depot", help="Source Data Directory", default="")
    # add a vhicle argument to the argument parser that will be used to specify which vehicle to generate a fuel line graph for
    parser.add_argument("-v", "--vehicle", help="Vehicle Name", default="")
    args = parser.parse_args()

    generateDashboard(args.depot, args.vehicle)
    # processLatencyCounts(args.depot)



# See PyCharm help at https://www.jetbrains.com/help/pycharm/
