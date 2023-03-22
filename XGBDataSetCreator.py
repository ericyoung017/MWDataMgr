import mxnet as mx
from mxnet import gluon
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
from itertools import islice
from pathlib import Path
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.common import ListDataset
import plotly.express as px

from gluonts.itertools import Map

from gluonts.dataset.pandas import PandasDataset


from datasets import Dataset, Features, Value, Sequence




def generateIdleTabularDataFrameList(directory):
    shipMap = {}
    tankMap = {}
    depotMap = {}
    #create a set of all tank sizes
    tankSizes = set()
    #create a set of all depot numbers
    depotNumbers = set()

    shipNumbers= set()
    for folder in os.listdir(directory):
        if "ships_" in folder:
            shipNumbers.add(int(folder.split("-")[2].split("_")[1]))
    #sort the ship numbers
    shipNumbers=sorted(shipNumbers)

    #convert the ship numbers to a list
    shipNumbers=list(shipNumbers)
    for i in range(len(shipNumbers)):
        shipMap[shipNumbers[i]]=i
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
    for i in range(len(tankSizes)):
        tankMap[tankSizes[i]]=i
    depotNumbers=sorted(depotNumbers)
    for i in range(len(depotNumbers)):
        depotMap[depotNumbers[i]]=i
    #create a list of dataframes corresponding to the timeSeries data for the gluonTS model
    dfList=[]

    #iterate through the tank sizes and depot numbers in the dataframe and set the value to a random number between zero and 1
    for shipNum in shipNumbers:
        for tankSize in tankSizes:
            for depotNum in depotNumbers:
                runDirectory=os.path.join(directory, "depots_"+str(depotNum)+"-tank_"+str(tankSize)+"-ships_"+str(shipNum))
                syncTime=findTimeVehiclesStabilized(runDirectory)
                dfList.append(generateSingeTabularDataset(runDirectory, syncTime, depotNum, tankSize, shipNum))
    return dfList

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

def generateSingeTabularDataset(directory,syncTime,depotNum, tankSize, shipNum):
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        # checking if it is a file
        if os.path.isfile(f):
            if "averageLatency" in filename:
                # read in the pickle file, create a dataframe, and append it to the list of vehicle dataframes
                latencyFrame = pd.read_pickle(f)

                #syncTimes with all other moos logs
                latencyFrame = latencyFrame.loc[latencyFrame['time'] > syncTime]
                #remove all average latency values outside of the 5th and 95th percentile
                latencyFrame = latencyFrame.loc[latencyFrame['average'] < latencyFrame['average'].quantile(0.95)]
                #drop the time column
                latencyFrame.drop(columns=['time'], inplace=True)
                # reset index
                latencyFrame.reset_index(inplace=True)
                
                #add the first static feature as depot
                latencyFrame['depot'] = depotNum
                #add the second static feature as tank
                latencyFrame['tank'] = tankSize
                #add the third static feature as ship
                latencyFrame['ship'] = shipNum
                #rename the average column to values
                latencyFrame.rename(columns={'average': 'values'}, inplace=True)
                #sample the data to 1000 points
                latencyFrame = latencyFrame.sample(n=100)
                #create a histogram plot of the average latency values
                # fig = px.histogram(latencyFrame, x="values")
                # fig.show()
                return latencyFrame
def getRunCombinations(directory):
        #create a set of all tank sizes
    tankSizes = set()
    #create a set of all depot numbers
    depotNumbers = set()

    shipNumbers= set()
    for folder in os.listdir(directory):
        if "ships_" in folder:
            shipNumbers.add(int(folder.split("-")[2].split("_")[1]))
    #sort the ship numbers
    shipNumbers=sorted(shipNumbers)
    #convert the ship numbers to a list
    shipNumbers=list(shipNumbers)
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
    return shipNumbers, tankSizes, depotNumbers

