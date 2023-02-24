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
from dash import Dash, html, dcc, Input, Output, callback, dash_table, State
import dash
import dash_bootstrap_components as dbc
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import datetime
from scipy import signal
import plotly.figure_factory as ff
from multiprocessing.pool import ThreadPool
import concurrent.futures
import multiprocessing
import pickle
import pyarrow



def load_numpickle(fpath):
    df = pd.DataFrame(np.load(fpath, allow_pickle=True))
    with open(fpath, "rb") as fin:
        meta = pickle.load(fin)
    # if no ‘types’ present, assuming all_numeric
    df.index, df.columns, dtypes = \
        meta['rownames'], meta['colnames'], meta.get('dtypes', None)
    if dtypes is not None:
        df = df.astype(dtypes)
    return df
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
    runDirectories = []
    returnThreadResults = []
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        # checking if it is a file
        if os.path.isfile(f):
            if "DEPOT" not in filename and "SHORESIDE" not in filename and "fuelTime" in filename:
                # # read in the pickle file, create a dataframe, and append it to the list of vehicle dataframes
                vehicleFrame = pd.read_pickle(f)
                #syncTimes with all other moos logs
                vehicleFrame = vehicleFrame.loc[vehicleFrame['time'] > syncTime]
                #reset index
                vehicleFrame.reset_index(inplace=True)
                # subtract the earliest time from the time column
                vehicleFrame['time'] = vehicleFrame['time'] - vehicleFrame['time'].min()
                #temporary catch to elminate spurios data logged after we quit the experiement
                vehicleFrame = vehicleFrame[(vehicleFrame['time'] < 85200)]
                # convert the time column to a datetime object
                vehicleFrame['time'] = pd.to_datetime(vehicleFrame['time'], unit='s')
                vehicleFrame['datetime']=vehicleFrame['time']
                #runDirectories.append(f)
                vehicleFrame = vehicleFrame[::4]
                vehicleFrames.append(vehicleFrame)
# using 6 processes to run the helper function, run fuel line graph helper in parallel over all the runDirectories using syncTime. Append the results as they arrive to the vehicle dataframe
#     with concurrent.futures.ProcessPoolExecutor(max_workers=6) as executor:
#         results = [executor.submit(fuelLineGraphHelper, f, syncTime) for f in runDirectories]
#         for f in concurrent.futures.as_completed(results):
#             vehicleFrames.append(f.result())
        # create a scatter plot of the fuel levels of all vehicles
    fig = go.Figure()
    for frame in vehicleFrames:
        fig.add_trace(go.Scatter(x=frame['time'], y=frame['level'], mode='lines', name=frame['vname'][0]))
    return fig
def fuelLineGraphHelper(f,syncTime):
    # read in the pickle file, create a dataframe, and append it to the list of vehicle dataframes
    vehicleFrame = pd.read_pickle(f)
    # syncTimes with all other moos logs
    vehicleFrame = vehicleFrame.loc[vehicleFrame['time'] > syncTime]
    # reset index
    vehicleFrame.reset_index(inplace=True)
    # subtract the earliest time from the time column
    vehicleFrame['time'] = vehicleFrame['time'] - vehicleFrame['time'].min()
    # # temporary catch to elminate spurios data logged after we quit the experiement
    # vehicleFrame = vehicleFrame[(vehicleFrame['time'] < 85200)]
    # convert the time column to a datetime object
    vehicleFrame['time'] = pd.to_datetime(vehicleFrame['time'], unit='s')
    #vehicleFrame.set_index('time', inplace=True)
    #convert dataframe to 1 minuute frequency
    vehicleFrame= vehicleFrame.asfreq('1Min')
    return vehicleFrame
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
                #eliminate all entries from frame whose count is less than zero


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

def generateAverageIdleTimeDataFrame(directory,shipNum):
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

            tankSizes.add(int(tankSize))
            depotNumbers.add(int(depotNum))
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
            runDirectory=os.path.join(directory, "depots_"+str(depotNum)+"-tank_"+str(tankSize)+"-ships_"+str(shipNum))
            syncTime=findTimeVehiclesStabilized(runDirectory)
            df[tankSize][depotNum]= round(computeShoresideLatencyAverage(runDirectory, syncTime),2)
    return df

def generateAverageAreaDataFrame(directory,shipNum):
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
            tankSizes.add(int(tankSize))
            depotNumbers.add(int(depotNum))
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
            runDirectory=os.path.join(directory, "depots_"+str(depotNum)+"-tank_"+str(tankSize)+"-ships_"+str(shipNum))
            syncTime=findTimeVehiclesStabilized(runDirectory)
            df[tankSize][depotNum]= round(computeAreaAverage(runDirectory, syncTime),2)
    return df


def generateDepotLatencySummaryFigure(idleSummaryDF):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=idleSummaryDF['Number of Depots'], y=idleSummaryDF[250], mode='lines', name="250"))
    fig.add_trace(go.Scatter(x=idleSummaryDF['Number of Depots'], y=idleSummaryDF[500], mode='lines', name="500"))
    fig.add_trace(go.Scatter(x=idleSummaryDF['Number of Depots'], y=idleSummaryDF[750], mode='lines', name="750"))
    return fig

def generateAverageAreaSummaryFigure(areaSummaryDF):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=areaSummaryDF['Number of Depots'], y=areaSummaryDF[250], mode='lines', name="250"))
    fig.add_trace(go.Scatter(x=areaSummaryDF['Number of Depots'], y=areaSummaryDF[500], mode='lines', name="500"))
    fig.add_trace(go.Scatter(x=areaSummaryDF['Number of Depots'], y=areaSummaryDF[750], mode='lines', name="750"))
    return fig
def generateAverageAreaSummaryFigureMap(directory, shipNumbers):
    #iterate through the ship numbers and generate a dataframe for each ship number make a figure for each ship number from the dataframe and add it to a figure map
    figMap = {}
    for shipNum in shipNumbers:
        areaSummaryDF = generateAverageAreaDataFrame(directory,shipNum)
        fig = generateAverageAreaSummaryFigure(areaSummaryDF)
        figMap[shipNum] = fig
    return figMap
def generateDepotLatencySummaryFigureMap(directory, shipNumbers):
    #iterate through the ship numbers and generate a dataframe for each ship number make a figure for each ship number from the dataframe and add it to a figure map
    figMap = {}
    for shipNum in shipNumbers:
        idleSummaryDF = generateAverageIdleTimeDataFrame(directory,shipNum)
        fig = generateDepotLatencySummaryFigure(idleSummaryDF)
        figMap[shipNum] = fig
    # generateAverageIdleTimeDataFrame
def generateSummaryDataFrame(directory,shipNumber):
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
            tankSizes.add(int(tankSize))
            depotNumbers.add(int(depotNum))
    #sort the tank sizes and depot numbers
    tankSizes=sorted(tankSizes)
    depotNumbers=sorted(depotNumbers)
    #create a pandas datafrome using tank sizes as columhs and depot numbers as rows
    df = pd.DataFrame(index=depotNumbers, columns=tankSizes)
    #create column in the dataframe for the number of depots
    df.insert(0, "Number of Depots", depotNumbers)
    areaDF=generateAverageAreaDataFrame(directory,shipNumber)
    idleDF=generateAverageIdleTimeDataFrame(directory,shipNumber)
    #iterate through the tank sizes and depot numbers in the dataframe and set the to the string concatination of the other area and idle time dataframes
    for tankSize in tankSizes:
        for depotNum in depotNumbers:
            # runDirectory=os.path.join(directory, "depots_"+str(depotNum)+"-tank_"+str(tankSize))
            # syncTime=findTimeVehiclesStabilized(runDirectory)
            df[tankSize][depotNum]= str(areaDF[tankSize][depotNum]) + " / " + str(idleDF[tankSize][depotNum])
    return df
def getShipQuantityInformationDF(directory):
    #iterate through the folders in the directory, for each folder, split the folder name by "_" and assign ships to the third element
    shipNumbers= set()
    for folder in os.listdir(directory):
        if "ships_" in folder:
            shipNumbers.add(int(folder.split("_")[3]))
    #sort the ship numbers
    shipNumbers=sorted(shipNumbers)
    #convert the set to a pandas dataframe with column name "Ship Number"
    df=pd.DataFrame(shipNumbers, columns=["Ship Number"])
    return df
def getShipQuantityInformation(directory):
    #iterate through the folders in the directory, for each folder, split the folder name by "_" and assign ships to the third element
    shipNumbers= set()
    for folder in os.listdir(directory):
        if "ships_" in folder:
            shipNumbers.add(int(folder.split("-")[2].split("_")[1]))
    #sort the ship numbers
    shipNumbers=sorted(shipNumbers)
    #convert the ship numbers to a list
    shipNumbers=list(shipNumbers)

    return shipNumbers
# def generateVariableShipButtons(directory):
#     #get the ship numbers
#     shipNumbers=getShipQuantityInformation(directory)
#     #create a list of buttons
#     buttons=[]
#     #iterate through the ship numbers and create a button for each ship number
#     for shipNumber in shipNumbers:
#         buttons.append(html.Button(str(shipNumber), id="ship_"+str(shipNumber), n_clicks=0))
#     buttons.append(html.Div(id='container-button-timestamp'))
#     return buttons

def generateFuelLineFigureMap(directory, shipNumbers):
    # this function finds the number of unique depots and tank sizes in the directory
    #the function then creates a double list of figures, one for each depot and tank size combination given a ship qty
    #the function then assigns the double list of figures to a map with the key being the ship qty and the value being the double list of figures
    #the function then returns the map
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
            tankSizes.add(int(tankSize))
            depotNumbers.add(int(depotNum))
    #sort the tank sizes and depot numbers
    tankSizes=sorted(tankSizes)
    depotNumbers=sorted(depotNumbers)
    #create a map of ship quantities to double lists of figures
    shipQtyMap={}
    #iterate through the ship quantities
    for shipQty in shipNumbers:
        #create a list to hold the figures
        figures=[]
        #iterate through the depot numbers
        for depotNum in depotNumbers:
            #iterate through the tank sizes
            tankList=[]
            for tankSize in tankSizes:
                print("depots_"+str(depotNum)+"-tank_"+str(tankSize)+ "-ships_"+str(shipQty))
                #create the file directory for the fuel line graph for the given ship qty, depot number, and tank size
                fileDirectory=os.path.join(directory, "depots_"+str(depotNum)+"-tank_"+str(tankSize)+ "-ships_"+str(shipQty))
                #generate the vehicle stabilization time from the file directory
                syncTime=findTimeVehiclesStabilized(fileDirectory)
                #generate the fuel line figure from the file directory and vehicle stabilization time
                fig=generateFuelLineGraph(fileDirectory, syncTime)
                #add the figure to the list
                tankList.append(fig)
            #add the tank list to the figures list
            figures.append(tankList)
        #add the figures list to the map with the ship qty as the key
        shipQtyMap[shipQty]=figures
    return shipQtyMap
def generateIdleTimeHistogramMap(directory, shipNumbers):
    # this function finds the number of unique depots and tank sizes in the directory
    #the function then creates a double list of figures, one for each depot and tank size combination given a ship qty
    #the function then assigns the double list of figures to a map with the key being the ship qty and the value being the double list of figures
    #the function then returns the map
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
            tankSizes.add(int(tankSize))
            depotNumbers.add(int(depotNum))
    #sort the tank sizes and depot numbers
    tankSizes=sorted(tankSizes)
    depotNumbers=sorted(depotNumbers)
    #create a map of ship quantities to double lists of figures
    shipQtyMap={}
    #iterate through the ship quantities
    for shipQty in shipNumbers:
        #create a list to hold the figures
        figures=[]
        #iterate through the depot numbers
        for depotNum in depotNumbers:
            #iterate through the tank sizes
            tankList=[]
            for tankSize in tankSizes:
                #create the file directory for the fuel line graph for the given ship qty, depot number, and tank size
                fileDirectory=os.path.join(directory, "depots_"+str(depotNum)+"-tank_"+str(tankSize)+"-ships_"+str(shipQty))
                #generate the vehicle stabilization time from the file directory
                syncTime=findTimeVehiclesStabilized(fileDirectory)
                #generate the fuel line figure from the file directory and vehicle stabilization time
                fig=generateAnnotatedIdleTimeHistogram(fileDirectory, syncTime)
                #add the figure to the list
                tankList.append(fig)
            #add the tank list to the figures list
            figures.append(tankList)
        #add the figures list to the map with the ship qty as the key
        shipQtyMap[shipQty]=figures
    return shipQtyMap
def generateLatencyAverageMap(directory, shipNumbers):
    # this function finds the number of unique depots and tank sizes in the directory
    #the function then creates a double list of figures, one for each depot and tank size combination given a ship qty
    #the function then assigns the double list of figures to a map with the key being the ship qty and the value being the double list of figures
    #the function then returns the map
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
            tankSizes.add(int(tankSize))
            depotNumbers.add(int(depotNum))
    #sort the tank sizes and depot numbers
    tankSizes=sorted(tankSizes)
    depotNumbers=sorted(depotNumbers)
    #create a map of ship quantities to double lists of figures
    shipQtyMap={}
    #iterate through the ship quantities
    for shipQty in shipNumbers:
        #create a list to hold the figures
        figures=[]
        #iterate through the depot numbers
        for depotNum in depotNumbers:
            #iterate through the tank sizes
            tankList=[]
            for tankSize in tankSizes:
                #create the file directory for the fuel line graph for the given ship qty, depot number, and tank size
                fileDirectory=os.path.join(directory, "depots_"+str(depotNum)+"-tank_"+str(tankSize)+ "-ships_"+str(shipQty))
                #generate the vehicle stabilization time from the file directory
                syncTime=findTimeVehiclesStabilized(fileDirectory)
                #generate the fuel line figure from the file directory and vehicle stabilization time
                fig=generateLatencyAverageLineGraph(fileDirectory, syncTime)
                #add the figure to the list
                tankList.append(fig)
            #add the tank list to the figures list
            figures.append(tankList)
        #add the figures list to the map with the ship qty as the key
        shipQtyMap[shipQty]=figures
    return shipQtyMap
def generateAreaStatsLineGraphMap(directory, shipNumbers):
    # this function finds the number of unique depots and tank sizes in the directory
    #the function then creates a double list of figures, one for each depot and tank size combination given a ship qty
    #the function then assigns the double list of figures to a map with the key being the ship qty and the value being the double list of figures
    #the function then returns the map
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
            tankSizes.add(int(tankSize))
            depotNumbers.add(int(depotNum))
    #sort the tank sizes and depot numbers
    tankSizes=sorted(tankSizes)
    depotNumbers=sorted(depotNumbers)
    #create a map of ship quantities to double lists of figures
    shipQtyMap={}
    #iterate through the ship quantities
    for shipQty in shipNumbers:
        #create a list to hold the figures
        figures=[]
        #iterate through the depot numbers
        for depotNum in depotNumbers:
            #iterate through the tank sizes
            tankList=[]
            for tankSize in tankSizes:
                #create the file directory for the fuel line graph for the given ship qty, depot number, and tank size
                fileDirectory=os.path.join(directory, "depots_"+str(depotNum)+"-tank_"+str(tankSize)+ "-ships_"+str(shipQty))
                #generate the vehicle stabilization time from the file directory
                syncTime=findTimeVehiclesStabilized(fileDirectory)
                #generate the fuel line figure from the file directory and vehicle stabilization time
                fig=generateAreaStatsLineGraph(fileDirectory, syncTime)
                #add the figure to the list
                tankList.append(fig)
            #add the tank list to the figures list
            figures.append(tankList)
        #add the figures list to the map with the ship qty as the key
        shipQtyMap[shipQty]=figures
    return shipQtyMap
def generateFuelingCountLineGraphMap(directory, shipNumbers):
    # this function finds the number of unique depots and tank sizes in the directory
    #the function then creates a double list of figures, one for each depot and tank size combination given a ship qty
    #the function then assigns the double list of figures to a map with the key being the ship qty and the value being the double list of figures
    #the function then returns the map
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
            tankSizes.add(int(tankSize))
            depotNumbers.add(int(depotNum))
    #sort the tank sizes and depot numbers
    tankSizes=sorted(tankSizes)
    depotNumbers=sorted(depotNumbers)
    #create a map of ship quantities to double lists of figures
    shipQtyMap={}
    #iterate through the ship quantities
    for shipQty in shipNumbers:
        #create a list to hold the figures
        figures=[]
        #iterate through the depot numbers
        for depotNum in depotNumbers:
            #iterate through the tank sizes
            tankList=[]
            for tankSize in tankSizes:
                #create the file directory for the fuel line graph for the given ship qty, depot number, and tank size
                fileDirectory=os.path.join(directory, "depots_"+str(depotNum)+"-tank_"+str(tankSize)+ "-ships_"+str(shipQty))
                #generate the vehicle stabilization time from the file directory
                syncTime=findTimeVehiclesStabilized(fileDirectory)
                #generate the fuel line figure from the file directory and vehicle stabilization time
                fig=generateVehicleFuelingCountLineGraph(fileDirectory, syncTime)
                #add the figure to the list
                tankList.append(fig)
            #add the tank list to the figures list
            figures.append(tankList)
        #add the figures list to the map with the ship qty as the key
        shipQtyMap[shipQty]=figures
    return shipQtyMap

def generateSummaryDataFrameMap(directory, shipNumbers):
    #iterate through the ship quantities, calculate the summary data frame for each ship quantity, and add the data frame to the map
    shipQtyMap={}
    for shipQty in shipNumbers:
        shipQtyMap[shipQty]=generateSummaryDataFrame(directory, shipQty)
    return shipQtyMap
def generateSummaryIdleTimeFigureMap(directory, shipNumbers):
    #iterate through the ship quantities, calculate the summary idle time figure for each ship quantity, and add the figure to the map
    shipQtyMap={}
    for shipQty in shipNumbers:
    #generate average idle time data frame
        df=generateAverageIdleTimeDataFrame(directory, shipQty)
        fig = generateAverageAreaSummaryFigure(df)
        shipQtyMap[shipQty]=fig
    return shipQtyMap
def generateSummaryAreaFigureMap(directory, shipNumbers):
    #iterate through the ship quantities, calculate the summary area figure for each ship quantity, and add the figure to the map
    shipQtyMap={}
    for shipQty in shipNumbers:
    #generate average area data frame
        df=generateAverageAreaDataFrame(directory, shipQty)
        fig = generateAverageAreaSummaryFigure(df)
        shipQtyMap[shipQty]=fig
    return shipQtyMap
def generateFuelPosFig(directory, syncTime):

    runDirectories = []
    tempFrames = []
    vehicleFrame = pd.DataFrame(columns=['time', 'x', 'y'])
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        # checking if it is a file
        if os.path.isfile(f):
            if "DEPOT" not in filename and "SHORESIDE" not in filename and "fuelPos" in filename:
                # # read in the pickle file, create a dataframe, and append it to the list of vehicle dataframes
                tempFrame = pd.read_pickle(f)

                tempFrame = tempFrame.loc[tempFrame['time'] > syncTime]

                tempFrames.append(tempFrame)
                # subtract the earliest time from the time column
    vehicleFrame = pd.concat(tempFrames)
    #sort the vehicle frame by time
    vehicleFrame.sort_values(by=['time'], inplace=True)
    vehicleFrame['time'] = vehicleFrame['time'] - vehicleFrame['time'].min()
    vehicleFrame.reset_index(inplace=True)
    # vehicleFrame['time'] = pd.to_datetime(vehicleFrame['time'], unit='s')
    # vehicleFrame['datetime']=vehicleFrame['time']
    fig = px.scatter(vehicleFrame, x="x", y="y", color="time")
    return fig

def generateFuelPosMap(directory,shipNumbers):
    tankSizes = set()
    # create a set of all depot numbers
    depotNumbers = set()
    # determine all the unique tank sizes and depot numbers
    for run in os.listdir(directory):
        if "tank_" in run:
            # split the run name using "-" as the delimiter into number of depots and tank size
            depotNum = run.split("-")[0].split("_")[1]

            tankSize = run.split("-")[1].split("_")[1]

            # add the tank size and depot number to the sets
            tankSizes.add(int(tankSize))
            depotNumbers.add(int(depotNum))
    # sort the tank sizes and depot numbers
    tankSizes = sorted(tankSizes)
    depotNumbers = sorted(depotNumbers)
    # create a map of ship quantities to double lists of figures
    shipQtyMap = {}
    # iterate through the ship quantities
    for shipQty in shipNumbers:
        # create a list to hold the figures
        figures = []
        # iterate through the depot numbers
        for depotNum in depotNumbers:
            # iterate through the tank sizes
            tankList = []
            for tankSize in tankSizes:
                # create the file directory for the fuel line graph for the given ship qty, depot number, and tank size
                fileDirectory = os.path.join(directory,
                                             "depots_" + str(depotNum) + "-tank_" + str(tankSize) + "-ships_" + str(
                                                 shipQty))
                # generate the vehicle stabilization time from the file directory
                syncTime = findTimeVehiclesStabilized(fileDirectory)
                # generate the fuel line figure from the file directory and vehicle stabilization time
                fig = generateFuelPosFig(fileDirectory, syncTime)
                # add the figure to the list
                tankList.append(fig)
            # add the tank list to the figures list
            figures.append(tankList)
        # add the figures list to the map with the ship qty as the key
        shipQtyMap[shipQty] = figures
    return shipQtyMap

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
    fig9= go.Figure()
    fig10= go.Figure()
    #syncTime=generateSyncTime(depotDirectory)
    shipQtyInfo=getShipQuantityInformation(depotDirectory)
    #we want to originally set our ship quantity to the first ship quantity in the list
    shipQty=shipQtyInfo[0]
    shipQtyDF=getShipQuantityInformationDF(depotDirectory)
    #generate the maps of figures for the given ship quantity
    currentWorkingDirectory=os.getcwd()
    featherDirectory=os.path.join(currentWorkingDirectory, "FeatherFiles")
    usedCacheFigures=True
    if not usedCacheFigures:
        shipQtyAreaMap=generateSummaryAreaFigureMap(vehicleDirectory, shipQtyInfo)

        with open(os.path.join(featherDirectory, "shipQtyAreaMap.pkl"), 'wb') as f:
            pickle.dump(shipQtyAreaMap, f)
        shipQtyIdleTimeMap=generateSummaryIdleTimeFigureMap(vehicleDirectory, shipQtyInfo)
        with open(os.path.join(featherDirectory, "shipQtyIdleTimeMap.pkl"), 'wb') as f:
            pickle.dump(shipQtyIdleTimeMap, f)
        shipQtySummaryMap=generateSummaryDataFrameMap(vehicleDirectory, shipQtyInfo)
        with open(os.path.join(featherDirectory, "shipQtySummaryMap.pkl"), 'wb') as f:
            pickle.dump(shipQtySummaryMap, f)
        shipQtyFuelLineMap=generateFuelLineFigureMap(vehicleDirectory, shipQtyInfo)
        with open(os.path.join(featherDirectory, "shipQtyFuelLineMap.pkl"), 'wb') as f:
            pickle.dump(shipQtyFuelLineMap, f)
        shipQtyFuelCountMap=generateFuelingCountLineGraphMap(vehicleDirectory, shipQtyInfo)
        with open(os.path.join(featherDirectory, "shipQtyFuelCountMap.pkl"), 'wb') as f:
            pickle.dump(shipQtyFuelCountMap, f)
    #write the summary figure maps
        shipAreaSummaryFigureMap=generateSummaryAreaFigureMap(vehicleDirectory, shipQtyInfo)
        with open(os.path.join(featherDirectory, "shipAreaSummaryFigureMap.pkl"), 'wb') as f:
            pickle.dump(shipAreaSummaryFigureMap, f)
        shipIdleTimeSummaryFigureMap=generateSummaryIdleTimeFigureMap(vehicleDirectory, shipQtyInfo)
        with open(os.path.join(featherDirectory, "shipIdleTimeSummaryFigureMap.pkl"), 'wb') as f:
            pickle.dump(shipIdleTimeSummaryFigureMap, f)
        idleTimeHistogramMap=generateIdleTimeHistogramMap(vehicleDirectory, shipQtyInfo)
        with open(os.path.join(featherDirectory, "idleTimeHistogramMap.pkl"), 'wb') as f:
            pickle.dump(idleTimeHistogramMap, f)
        latencyAverageMap=generateLatencyAverageMap(vehicleDirectory, shipQtyInfo)
        with open(os.path.join(featherDirectory, "latencyAverageMap.pkl"), 'wb') as f:
            pickle.dump(latencyAverageMap, f)
        fuelPosMap=generateFuelPosMap(vehicleDirectory, shipQtyInfo)
        with open(os.path.join(featherDirectory, "fuelPosMap.pkl"), 'wb') as f:
            pickle.dump(fuelPosMap, f)
        areaStatesLineGraphMap=generateAreaStatsLineGraphMap(vehicleDirectory, shipQtyInfo)
        with open(os.path.join(featherDirectory, "areaStatesLineGraphMap.pkl"), 'wb') as f:
            pickle.dump(areaStatesLineGraphMap, f)

    else:
        #read the maps from the pickle files
        with open(os.path.join(featherDirectory, "shipQtyAreaMap.pkl"), 'rb') as f:
            shipQtyAreaMap = pickle.load(f)
        with open(os.path.join(featherDirectory, "shipQtyIdleTimeMap.pkl"), 'rb') as f:
            shipQtyIdleTimeMap = pickle.load(f)
        with open(os.path.join(featherDirectory, "shipQtySummaryMap.pkl"), 'rb') as f:
            shipQtySummaryMap = pickle.load(f)
        with open(os.path.join(featherDirectory, "shipQtyFuelLineMap.pkl"), 'rb') as f:
            shipQtyFuelLineMap = pickle.load(f)
        with open(os.path.join(featherDirectory, "shipQtyFuelCountMap.pkl"), 'rb') as f:
            shipQtyFuelCountMap = pickle.load(f)
        with open(os.path.join(featherDirectory, "shipAreaSummaryFigureMap.pkl"), 'rb') as f:
            shipAreaSummaryFigureMap = pickle.load(f)
        with open(os.path.join(featherDirectory, "shipIdleTimeSummaryFigureMap.pkl"), 'rb') as f:
            shipIdleTimeSummaryFigureMap = pickle.load(f)
        with open(os.path.join(featherDirectory, "idleTimeHistogramMap.pkl"), 'rb') as f:
            idleTimeHistogramMap = pickle.load(f)
        with open(os.path.join(featherDirectory, "fuelPosMap.pkl"), 'rb') as f:
            fuelPosMap = pickle.load(f)
        with open(os.path.join(featherDirectory, "latencyAverageMap.pkl"), 'rb') as f:
            latencyAverageMap = pickle.load(f)
        with open(os.path.join(featherDirectory, "areaStatesLineGraphMap.pkl"), 'rb') as f:
            areaStatesLineGraphMap = pickle.load(f)






    # if(True):
    #     #dump all maps to pickle files
    #     with open(os.path.join(featherDirectory, "shipQtyAreaMap.pkl"), 'wb') as f:
    #         pickle.dump(shipQtyAreaMap, f)
    #     with open(os.path.join(featherDirectory, "shipQtyIdleTimeMap.pkl"), 'wb') as f:
    #         pickle.dump(shipQtyIdleTimeMap, f)
    #     with open(os.path.join(featherDirectory, "shipQtySummaryMap.pkl"), 'wb') as f:
    #         pickle.dump(shipQtySummaryMap, f)
    #     with open(os.path.join(featherDirectory, "shipQtyFuelLineMap.pkl"), 'wb') as f:
    #         pickle.dump(shipQtyFuelLineMap, f)
    #     with open(os.path.join(featherDirectory, "shipQtyFuelCountMap.pkl"), 'wb') as f:
    #         pickle.dump(shipQtyFuelCountMap, f)



    # df=generateSummaryDataFrame(vehicleDirectory, shipQty)
    # fig3=generateDepotLatencySummaryFigure(generateAverageIdleTimeDataFrame(vehicleDirectory,shipQty))
    # fig9=generateAverageAreaSummaryFigure(generateAverageAreaDataFrame(vehicleDirectory,shipQty))
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
    fig9.update_layout(
        plot_bgcolor=colors['background'],
        paper_bgcolor=colors['background'],
        font_color=colors['text']
    )
    fig3.update_layout(
        plot_bgcolor=colors['background'],
        paper_bgcolor=colors['background'],
        font_color=colors['text']
    )
    # fig4.update_layout(
    #     plot_bgcolor=colors['background'],
    #     paper_bgcolor=colors['background'],
    #     font_color=colors['text']
    # )
    # fig10.update_layout(
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
        }), dash_table.DataTable(shipQtyDF.to_dict('records'),[{"name": str(i), "id": str(i)} for i in shipQtyDF.columns], id='stbl',style_data={
        'backgroundColor': colors['background'],'textAlign': 'center',
        'color': colors['text']
    },style_header={
        'backgroundColor': 'rgb(30, 30, 30)','textAlign': 'center',
        'color': colors['text']
    }),
    dash_table.DataTable(shipQtySummaryMap[shipQty].to_dict('records'),[{"name": str(i), "id": str(i)} for i in shipQtySummaryMap[shipQty].columns], id='tbl',style_data={
        'backgroundColor': colors['background'],
        'color': colors['text']
    },style_header={
        'backgroundColor': 'rgb(30, 30, 30)',
        'color': colors['text']
    }),
    dbc.Alert(id='tbl_out')
            ,
            dbc.Alert(id='stbl_out')

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
        ), dcc.Graph(
            id='area-stats-timeline',
            figure=fig7


        ), dcc.Graph(
            id='idle-time-histogram',
            figure=fig4
        ),
        dcc.Graph(
            id='latency-average-timeline',
            figure=fig5
        ),        dcc.Graph(
            id='refuel-pos-graph',
            figure=fig10
        ),dcc.Graph(
            id='area-average-graph',
            figure=fig9
        ),
        dcc.Graph(
            id='depot-latency-graph',
            figure=fig3
        ),dcc.Store(id="shipQty", data=shipQty)
        #
    ])

    # @callback(Output('tbl_out', 'children'), Input('tbl', 'active_cell'))
    # def update_graphs(active_cell):
    #     return str(active_cell) if active_cell else "Click the table"
    # @callback(Output('fueling-timeline', 'figure'), Input('tbl', 'active_cell'))
    # def update_graphs(active_cell):
    #     active_col_id = active_cell['column_id'] if active_cell else None
    #     active_row_id = active_cell['row']+1 if active_cell else None
    #     fig= go.Figure()
    #     if active_cell:
    #         runDirectory =  os.path.join(vehicleDirectory, "depots_"+str(active_row_id)+"-tank_"+str(active_col_id))
    #         syncTime=findTimeVehiclesStabilized(runDirectory)
    #         fig=generateFuelLineGraph(runDirectory,syncTime)
    #
    #         fig.update_layout(
    #             plot_bgcolor=colors['background'],
    #             paper_bgcolor=colors['background'],
    #             font_color=colors['text']
    #         )
    #     return fig
    @callback(Output('tbl_out', 'children'), Input('tbl', 'active_cell'))
    def update_graphs(active_cell):
        return str(active_cell['row'])+"-"+str(active_cell['column']) if active_cell else "click the table"
    @callback(Output('shipQty', 'data'), Input('stbl', 'active_cell'),Input('stbl', "derived_virtual_row_ids"))
    def update_graphs(active_cell,derived_virtual_row_ids):
        shipQty=shipQtyInfo[active_cell['row']] if active_cell else 5
        return shipQty
    @callback(Output('tbl', 'data'), Input('stbl', 'active_cell'))
    def update_graphs(active_cell):
        shipQty=shipQtyInfo[active_cell['row']] if active_cell else 5

        # return dash_table.DataTable(shipQtySummaryMap[shipQty].to_dict('records'),[{"name": str(i), "id": str(i)} for i in shipQtySummaryMap[shipQty].columns], id='tbl',style_data={

    #     'backgroundColor': colors['background'],
    #     'color': colors['text']
    # })
        return shipQtySummaryMap[shipQty].to_dict('records')
    @callback(Output('fueling-timeline', 'figure'), Input('tbl', 'active_cell'),Input('shipQty', 'data'))
    def update_graphs(active_cell,data):
        active_col_id = active_cell['column']-1 if active_cell else None
        active_row_id = active_cell['row'] if active_cell else None
        shipQty=data
        fig= go.Figure()
        if active_cell:
            fig=shipQtyFuelLineMap[shipQty][active_row_id][active_col_id]
            fig.update_layout(
                plot_bgcolor=colors['background'],
                paper_bgcolor=colors['background'],
                font_color=colors['text']
            )
        return fig
    @callback(Output('idle-time-histogram', 'figure'), Input('tbl', 'active_cell'),Input('shipQty', 'data'))
    def update_graphs(active_cell,data):
        active_col_id = active_cell['column'] - 1 if active_cell else None
        active_row_id = active_cell['row'] if active_cell else None
        shipQty = data
        fig = go.Figure()
        if active_cell:
            fig = idleTimeHistogramMap[shipQty][active_row_id][active_col_id]
            fig.update_layout(
                plot_bgcolor=colors['background'],
                paper_bgcolor=colors['background'],
                font_color=colors['text']
            )
        return fig
    @callback(Output('latency-average-timeline', 'figure'), Input('tbl', 'active_cell'),Input('shipQty', 'data'))
    def update_graphs(active_cell,data):
        active_col_id = active_cell['column'] - 1 if active_cell else None
        active_row_id = active_cell['row'] if active_cell else None
        shipQty = data
        fig = go.Figure()
        if active_cell:
            fig = latencyAverageMap[shipQty][active_row_id][active_col_id]
            fig.update_layout(
                plot_bgcolor=colors['background'],
                paper_bgcolor=colors['background'],
                font_color=colors['text']
            )
        return fig
    @callback(Output('area-stats-timeline', 'figure'), Input('tbl', 'active_cell'),Input('shipQty', 'data'))
    def update_graphs(active_cell,data):
        active_col_id = active_cell['column'] - 1 if active_cell else None
        active_row_id = active_cell['row'] if active_cell else None
        shipQty = data
        fig = go.Figure()
        if active_cell:
            fig = areaStatesLineGraphMap[shipQty][active_row_id][active_col_id]
            fig.update_layout(
                plot_bgcolor=colors['background'],
                paper_bgcolor=colors['background'],
                font_color=colors['text']
            )
        return fig
    @callback(Output('fueling-count-timeline', 'figure'), Input('tbl', 'active_cell'),Input('shipQty', 'data'))
    def update_graphs(active_cell,data):
        active_col_id = active_cell['column'] - 1 if active_cell else None
        active_row_id = active_cell['row'] if active_cell else None
        shipQty = data
        fig = go.Figure()
        if active_cell:
            fig = shipQtyFuelCountMap[shipQty][active_row_id][active_col_id]
            fig.update_layout(
                plot_bgcolor=colors['background'],
                paper_bgcolor=colors['background'],
                font_color=colors['text']
            )
        return fig
    @callback(Output('depot-latency-graph', 'figure'), Input('tbl', 'active_cell'),Input('shipQty', 'data'))
    def update_graphs(active_cell,data):
        active_col_id = active_cell['column'] - 1 if active_cell else None
        active_row_id = active_cell['row'] if active_cell else None
        shipQty = data
        fig = go.Figure()
        if active_cell:
            fig = shipIdleTimeSummaryFigureMap[shipQty]
            fig.update_layout(
                plot_bgcolor=colors['background'],
                paper_bgcolor=colors['background'],
                font_color=colors['text']
            )
        return fig
    @callback(Output('area-average-graph', 'figure'), Input('tbl', 'active_cell'),Input('shipQty', 'data'))
    def update_graphs(active_cell,data):
        active_col_id = active_cell['column'] - 1 if active_cell else None
        active_row_id = active_cell['row'] if active_cell else None
        shipQty = data
        fig = go.Figure()
        if active_cell:
            fig = shipAreaSummaryFigureMap[shipQty]
            fig.update_layout(
                plot_bgcolor=colors['background'],
                paper_bgcolor=colors['background'],
                font_color=colors['text']
            )
        return fig
    @callback(Output('refuel-pos-graph', 'figure'), Input('tbl', 'active_cell'),Input('shipQty', 'data'))
    def update_graphs(active_cell,data):
        active_col_id = active_cell['column'] - 1 if active_cell else None
        active_row_id = active_cell['row'] if active_cell else None
        shipQty = data
        fig = go.Figure()
        if active_cell:
            fig = fuelPosMap[shipQty][active_row_id][active_col_id]
            fig.update_layout(
                plot_bgcolor=colors['background'],
                paper_bgcolor=colors['background'],
                font_color=colors['text']
            )
        return fig






    # @callback(Output('latency-average-timeline', 'figure'), Input('tbl', 'active_cell'))
    # def update_graphs(active_cell):
    #     active_col_id = active_cell['column_id'] if active_cell else None
    #     active_row_id = active_cell['row']+1 if active_cell else None
    #     fig= go.Figure()
    #     if active_cell:
    #         runDirectory =  os.path.join(vehicleDirectory, "depots_"+str(active_row_id)+"-tank_"+str(active_col_id))
    #         syncTime=findTimeVehiclesStabilized(runDirectory)
    #
    #         fig = generateLatencyAverageLineGraph(runDirectory,syncTime)
    #
    #         fig.update_layout(
    #             plot_bgcolor=colors['background'],
    #             paper_bgcolor=colors['background'],
    #             font_color=colors['text']
    #         )
    #     return fig
    # @callback(Output('area-stats-timeline', 'figure'), Input('tbl', 'active_cell'))
    # def update_graphs(active_cell):
    #     active_col_id = active_cell['column_id'] if active_cell else None
    #     active_row_id = active_cell['row']+1 if active_cell else None
    #     fig= go.Figure()
    #     if active_cell:
    #         runDirectory =  os.path.join(vehicleDirectory, "depots_"+str(active_row_id)+"-tank_"+str(active_col_id))
    #         syncTime=findTimeVehiclesStabilized(runDirectory)
    #
    #         fig = generateAreaStatsLineGraph(runDirectory,syncTime)
    #
    #         fig.update_layout(
    #             plot_bgcolor=colors['background'],
    #             paper_bgcolor=colors['background'],
    #             font_color=colors['text']
    #         )
    #     return fig
    # @callback(Output('fueling-count-timeline', 'figure'), Input('tbl', 'active_cell'))
    # def update_graphs(active_cell):
    #     active_col_id = active_cell['column_id'] if active_cell else None
    #     active_row_id = active_cell['row']+1 if active_cell else None
    #     fig= go.Figure()
    #     if active_cell:
    #         runDirectory =  os.path.join(vehicleDirectory, "depots_"+str(active_row_id)+"-tank_"+str(active_col_id))
    #         syncTime=findTimeVehiclesStabilized(runDirectory)
    #
    #         fig = generateVehicleFuelingCountLineGraph(runDirectory,syncTime)
    #         fig.update_layout(
    #             plot_bgcolor=colors['background'],
    #             paper_bgcolor=colors['background'],
    #             font_color=colors['text']
    #         )
    #     return fig


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
def generateAnnotatedIdleTimeHistogram(directory,syncTime):
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        # checking if it is a file
        if os.path.isfile(f) and "averageLatency" in filename:
            # read in a pickle file and assign it to a dataframe called latencyData
            latencyData = pd.read_pickle(f)
            #remove all the data before the sync time
            latencyData=latencyData.loc[latencyData['time'] > syncTime]
    #create a distplot using the average column of the latencyData dataframe
    fig = ff.create_distplot([latencyData['average']], ["average"])


    #fig = ff.create_distplot(latencyData, "average")
    mean = latencyData['average'].mean()
    stdev_pluss = mean + latencyData['average'].std()
    stdev_minus = mean + latencyData['average'].std() * -1

    fig.add_shape(type="line", x0=mean, x1=mean, y0=0, y1=0.02, xref='x', yref='y',
                  line=dict(color='orange', dash='dash'))
    fig.add_shape(type="line", x0=stdev_pluss, x1=stdev_pluss, y0=0, y1=0.02, xref='x', yref='y',
                  line=dict(color='red', dash='dash'))
    fig.add_shape(type="line", x0=stdev_minus, x1=stdev_minus, y0=0, y1=0.02, xref='x', yref='y',
                  line=dict(color='red', dash='dash'))

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
