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


from gluonts.dataset.pandas import PandasDataset
def generateIdleTimeSeriesDataFrameList(directory):
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
    #create a list of dataframes corresponding to the timeSeries data for the gluonTS model
    dfList=[]

    #iterate through the tank sizes and depot numbers in the dataframe and set the value to a random number between zero and 1
    for shipNum in shipNumbers:
        for tankSize in tankSizes:
            for depotNum in depotNumbers:
                runDirectory=os.path.join(directory, "depots_"+str(depotNum)+"-tank_"+str(tankSize)+"-ships_"+str(shipNum))
                syncTime=findTimeVehiclesStabilized(runDirectory)
                dfList.append(generateSingleIdleTimeSeriesDataset(runDirectory, syncTime, depotNum, tankSize, shipNum))
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

def generateSingleIdleTimeSeriesDataset(directory,syncTime,depotNum, tankSize, shipNum):
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
                seriesLength= 1140 # max samples we want to look at corresponding to 19 hours assumed sampling every minute
                latencyFrame=latencyFrame.iloc[:seriesLength]
                # convert the time column to a datetime object
                latencyFrame['time'] = pd.to_datetime(latencyFrame['time'], unit='s')
                # floor all the times to the nearest minute
               # latencyFrame['time'] = latencyFrame['time'].dt.floor('min')
                #check there is a time for every minute if not fill in the missing times with the previous value
                latencyFrame = latencyFrame.set_index('time').resample('min').ffill().reset_index()
                #add the first static feature as stat_cat_1
                latencyFrame['stat_cat_1'] = depotNum
                #add the second static feature as stat_cat_2
                latencyFrame['stat_cat_2'] = tankSize
                #add the third static feature as stat_cat_3
                latencyFrame['stat_cat_3'] = shipNum
                #rename the average column to values
                latencyFrame.rename(columns={'average': 'values'}, inplace=True)
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
#define a function to create a gluoTS dataset from a list of dataframes
def createGluonTSDataSet(dfList,directory):
#split the list into training and testing dataframes with 70 percent of the data for training and 30 percent for testing
    trainList=dfList[:int(len(dfList)*0.7)]
    testList=dfList[int(len(dfList)*0.7):]
    #create a gluonTSDataSet dictionary
    gluonTSDataSet={}
    #create a dataset dictionary for the training data
    multiple_train = {}
    #add each trainList dataframe to the dictionary each getting its own key
    for i in range(len(trainList)):
        multiple_train[i]=trainList[i]
    #create a dataset dictionary for the testing data
    multiple_test = {}
    #add each testList dataframe to the dictionary each getting its own key
    for i in range(len(testList)):
        multiple_test[i]=testList[i]

    #create a static feature dictionary for the gluonTS model
    # test=pd.Categorical(np.random.choice(["red", "green", "blue"], size=10))
    depotCategoryList= []
    #iterate through the trainList dataframes and add the depot number to the list
    for df in trainList:
        depotCategoryList.append(df['stat_cat_1'].values[0])
    #create a categorical feature for the depots
    depots = pd.Categorical(depotCategoryList)
    #iterate through the trainList dataframes and add the tank size to the list
    tankCategoryList= []
    for df in trainList:
        tankCategoryList.append(df['stat_cat_2'].values[0])
    #create a categorical feature for the tank sizes
    tanks = pd.Categorical(tankCategoryList)
    #iterate through the trainList dataframes and add the ship number to the list
    shipCategoryList= []
    for df in trainList:
        shipCategoryList.append(df['stat_cat_3'].values[0])
    #create a categorical feature for the ship numbers
    ships = pd.Categorical(shipCategoryList)
    static_features = pd.DataFrame({'stat_cat_1': depots,"stat_cat_2": tanks,"stat_cat_3": ships},index=list(multiple_train.keys()))

    
    
    gluonTSDataSet['train']=PandasDataset(multiple_train, freq='1min', static_features=static_features,target='values')





    depotCategoryList= []
    #iterate through the testList dataframes and add the depot number to the list
    for df in testList:
        depotCategoryList.append(df['stat_cat_1'].values[0])
    #create a categorical feature for the depots
    depots = pd.Categorical(depotCategoryList)
    #iterate through the testList dataframes and add the tank size to the list
    tankCategoryList= []
    for df in testList:
        tankCategoryList.append(df['stat_cat_2'].values[0])
    #create a categorical feature for the tank sizes
    tanks = pd.Categorical(tankCategoryList)
    #iterate through the testList dataframes and add the ship number to the list
    shipCategoryList= []
    for df in testList:
        shipCategoryList.append(df['stat_cat_3'].values[0])
    #create a categorical feature for the ship numbers
    ships = pd.Categorical(shipCategoryList)
    static_features = pd.DataFrame({'stat_cat_1': depots,"stat_cat_2": tanks,"stat_cat_3": ships},index=list(multiple_test.keys()))










    gluonTSDataSet['test']=PandasDataset(multiple_test, freq='1min', static_features=static_features,target='values')
    # #create a list of dictionaries for the gluonTS model
    # gluonTSList=[]
    # #iterate through the dataframes in the list and create a dictionary for each one
    # for df in dfList:
    #     #create a dictionary for the gluonTS model
    #     gluonTSList.append({'target': df['values'].values, 'start': df['time'].min(), 'feat_static_cat': [df['stat_cat_1'].values[0], df['stat_cat_2'].values[0], df['stat_cat_3'].values[0]]})
    # #create a gluonTS dataset from the list of dictionaries with 70 percent of the data for training and 30 percent for testing
    # gluonTSDataSet=ListDataset(gluonTSList, freq='1min')
    
    return gluonTSDataSet



# N = 10  # number of time series
# T = 100  # number of timesteps
# freq = "1H"
# custom_dataset = np.random.normal(size=(N, T))
# prediction_length = 24
# start = pd.Timestamp("01-01-2019", freq=freq)
# train_ds = ListDataset([{'target': x, 'start': start}
#                         for x in custom_dataset[:, :-prediction_length]],
#                        freq=freq)
# c = train_ds
