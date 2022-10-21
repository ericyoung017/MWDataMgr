# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import os
import numpy as np
from dash import Dash, html, dcc
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import datetime

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

def read_list(name,name2):
  #READ IN ALL OF DEPOT ONE's INFO
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
    df2=logFrame[pd.to_numeric(logFrame[2], errors='coerce').notnull()]
    df2.columns = ['id','vehicle','start', 'end', 'ACTIVITY']
    df3 = df2.copy()
    df3['start'] = df2['start'].astype(float)
    df3['end'] = df2['end'].astype(float)
  # ---------

#READ IN ALL OF DEPOT 2's info
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
    df2=logFrame[pd.to_numeric(logFrame[2], errors='coerce').notnull()]
    df2.columns = ['id','vehicle','start', 'end', 'ACTIVITY']
    df4 = df2.copy()
    df4['start'] = df2['start'].astype(float)
    df4['end'] = df2['end'].astype(float)
#----------------------
    #print(df3)
    #print(df4)
    deltaFrame =pd.concat([df3, df4])
  #SUBTRACT THE EARLIEST MOOSTIME ENTRY TO MAKE EVERYTHING REFERENCED FROM ZERO
    deltaFrame['start1']=deltaFrame['start']-33296827658
    deltaFrame['end1']=deltaFrame['end'] - 33296827658


#TRIM AND RESET INDEX OF CONCATENATED LIST
    nf=deltaFrame[['vehicle','start1','end1','ACTIVITY']].copy()
    nf.reset_index(drop=True, inplace=True)
#Convert our start and end times to datetimes so that they can be correctly read by plotly
    nf['end1'] = pd.to_datetime(nf['end1'],unit='s')
    nf['start1'] = pd.to_datetime(nf['start1'], unit='s')
    fig = px.timeline(
        nf, x_start="start1", x_end="end1", y="vehicle",

        color='ACTIVITY'

    )
    #fig.show()
    return fig

def generateFuelTimeHistogram():
    directory="/Users/ericyoung/moos-ivp-younge/missions/ufld_saxis/testLogFolder"
    logLines = []
    for filename in os.listdir(directory):
        name = os.path.join(directory, filename)

        # checking if it is a file
        if not filename.startswith('.') and os.path.isfile(name):
            #chomp the first header lines of the log file
            f = open(name)
            line = f.readline()
            line = f.readline()
            line = f.readline()
            line = f.readline()
            line = f.readline()
            while line:
                try:
                    line=f.readline()
                    #print(line)
                    logLines.append(line)
                except ValueError:
                    print('Error in line :' + line)
    rawLogData = pd.DataFrame([sub.split() for sub in logLines])
    #print(rawLogData[3])
    splitWaitLogs=rawLogData[3].str.split(pat='-',expand =True)
    finalWaitTimes=splitWaitLogs[pd.to_numeric(splitWaitLogs[1], errors='coerce').notnull()]
    finalWaitTimes.columns=['vehicle','time']
    finalWaitTimes['time']=finalWaitTimes['time'].astype(float)
    finalWaitTimes.sort_values(by=['time'],ascending=True)
    fig = px.histogram(finalWaitTimes, x="time")
    return fig
    #fig.show()
    #print(splitWaitLogs[1])
    #print(finalWaitTimes)
    #df2 = rawLogData[pd.to_numeric(rawLogData[3], errors='coerce').notnull()]

def generateFuelLineGraph():
    directory="/Users/ericyoung/moos-ivp-younge/missions/ufld_saxis/testLogFolder2"
    vehicleFrames=[]
    logLines = []
    fig = go.Figure()
    for filename in os.listdir(directory):
        name = os.path.join(directory, filename)

        # checking if it is a file
        if not filename.startswith('.') and os.path.isfile(name):
            #print(filename)
            f = open(name)
            line = f.readline()
            line = f.readline()
            line = f.readline()
            line = f.readline()
            line = f.readline()
            while line:
                try:
                    line=f.readline()
                    #print(line)
                    logLines.append(line)
                except ValueError:
                    print('Error in line :' + line)
            #create a list of dataframes that hold the "-" separated fuel log data for every vehicle
            vehicleFrames.append(pd.DataFrame([sub.split() for sub in logLines])[3])
            logLines=[]
    #split the fuel log data by "-"
    tempProcessedFuelData=[]
    for frame in vehicleFrames:
        tempProcessedFuelData.append(frame.str.split(pat='-',expand =True))
    #remove None rows from the fuel data
    dashSegmentedFuelData=[]
    for frame in tempProcessedFuelData:
        dashSegmentedFuelData.append(frame.replace(to_replace='None', value=np.nan).dropna())
    finalSegmentedFuelData=[]
    fig=go.Figure()
    for frame in dashSegmentedFuelData:
        #convert fuel and time values to numbers
        frame[1]=frame[1].astype(float)
        frame[2]=frame[2].astype(float)
        #name each column in the dataframe
        frame.columns=['vehicle','level','time']
        #remove all starting values with values greater than the fuel tank size (this is for testing only)
        tempData=frame[(frame['level'] <= 200)]
        #reset the dataframe index to make up for the values just removed
        tempData.reset_index(inplace=True)
        #zero out the time data based on the earliest value and convert to a datetime object
        tempData['time'] = tempData['time'] - tempData['time'][0]
        tempData['time'] = pd.to_datetime(tempData['time'],unit='s')
        finalSegmentedFuelData.append(tempData)
        fig.add_trace(go.Scatter(x=tempData['time'], y=tempData['level'],
                                 mode='lines',
                                 name=tempData['vehicle'][0]))
    return fig
    #fig.show()
    #print(finalSegmentedFuelData[0])
    #print((dashSegmentedFuelData[0]['level']<=200))

    #print(dashSegmentedFuelData[0][(dashSegmentedFuelData[0]['level']<=200)])
def generateDepotTimelines(self,data):
        print("foo")

if __name__ == '__main__':
    import sys
    import os
    import argparse

    # CONFIGURE ARGUMENT PARSER
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--examples", help="Display example uses", action="store_true")
    parser.add_argument("-s", "--source", help="Source Data Directory", default="")

    args = parser.parse_args()

    # assign directory
    directory = args.source
    depotData=pd.DataFrame(columns=["dname",'status', 'st', "et"])

    # iterate over files in
    # that directory
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        # checking if it is a file
        if os.path.isfile(f):
            if "DEPOT" in filename:
                print("Depot")




    # fig1 = generateFuelTimeHistogram()
    # fig2= read_list("/Users/ericyoung/moos-ivp-younge/missions/ufld_saxis/LOG_DEPOT_ONE_3_10_2022_____19_56_21/LOG_DEPOT_ONE_3_10_2022_____19_56_21.alog","/Users/ericyoung/moos-ivp-younge/missions/ufld_saxis/LOG_DEPOT_TWO_3_10_2022_____19_56_21/LOG_DEPOT_TWO_3_10_2022_____19_56_21.alog")
    # fig3=generateFuelLineGraph()
    # app=Dash(__name__)
    # colors = {
    #     'background': '#111111',
    #     'text': '#7FDBFF'
    # }
    # fig2.update_layout(
    #     plot_bgcolor=colors['background'],
    #     paper_bgcolor=colors['background'],
    #     font_color=colors['text']
    # )
    # fig1.update_layout(
    #     plot_bgcolor=colors['background'],
    #     paper_bgcolor=colors['background'],
    #     font_color=colors['text']
    # )
    # fig3.update_layout(
    #     plot_bgcolor=colors['background'],
    #     paper_bgcolor=colors['background'],
    #     font_color=colors['text']
    # )
    # app.layout = html.Div(style={'backgroundColor': colors['background']},children=[
    #     html.H1(children='Ship Supply Dashboard Visualization Tool',style={
    #         'textAlign': 'center',
    #         'color': colors['text']
    #     }),
    #
    #     html.Div(children='''
    #         Run: 10/5/22 at 2100
    #     ''',style={
    #     'textAlign': 'center',
    #     'color': colors['text']
    # }),
    #
    #     dcc.Graph(
    #         id='Fueling Histogram',
    #         figure=fig1
    #     ),
    #     dcc.Graph(
    #         id='Fueling Timeline',
    #         figure=fig2
    #     ),dcc.Graph(
    #         id='Fuel History',
    #         figure=fig3
    #     )
    #
    # ])
    # app.run_server(debug=True)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/


