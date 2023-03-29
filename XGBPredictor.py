import random
import string
import XGBDataSetCreator
import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import dash
import sympy as sp
# import random 
import plotly.express as px
from xgboost.sklearn import XGBRegressor
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from dash import Dash, html, dcc, Input, Output, callback, dash_table, State
# log cosh quantile is a regularized quantile loss function
def log_cosh_quantile(alpha):
    def _log_cosh_quantile(y_true, y_pred):
        err = y_pred - y_true
        err = np.where(err < 0, alpha * err, (1 - alpha) * err)
        grad = np.tanh(err)
        hess = 1 / np.cosh(err)**2
        return grad, hess
    return _log_cosh_quantile
def generateRMSEPlot(dfList):
    #create a dataframe with columns "Train_Size","RMSE_UP","RMSE_AVG","RMSE_median","RMSE_OP"
    plotDFRMSE=pd.DataFrame(columns=["Train_Size","RMSE_UP","RMSE_AVG","RMSE_median","RMSE_OP"])
    for trainSize in np.arange(0.3,0.9,0.05):
        #vary the size of the training set to determine how many data points are needed to train the model
        #plot the RMSE for the test set as a function of the size of the training set
        dfTestOriginal=[]
        dfTrain=[]



        #USE THIS CODE FOR RANDOM TESTING AND TRAINING DATA#################
        #shuffle the list of dataframes

        for df in dfList:
            if np.random.rand() < (1-trainSize):
                dfTestOriginal.append(df)
            else:
                dfTrain.append(df)
        print("Training on ",len(dfTrain)," data points")
        #create a depotPredict list containing the values of the depot column in each dataframe in dfTestOriginal
        depotPredict=[df.iloc[0]["depot"] for df in dfTestOriginal]
        #create a tankPredict list containing the values of the tank column in each dataframe in dfTestOriginal
        tankPredict=[df.iloc[0]["tank"] for df in dfTestOriginal]
        #create a shipPredict list containing the values of the ship column in each dataframe in dfTestOriginal
        shipPredict=[df.iloc[0]["ship"] for df in dfTestOriginal]

            #concatenate the dataframes in the list into a single dataframe
        df=pd.concat(dfTrain)
        #drop the columns "vname" as it is not needed for the model
        df=df.drop(columns=["vname"])
        #reset the index of the dataframe
        dtf=df.reset_index(drop=True)
        #drop the index column
        dtf=dtf.drop(columns=["index"])
        #convert all columns to int64
        #dtf=dtf.astype('int64')
        scaleFactor=500
        #divide the values column by 100
        dtf['values']=dtf['values']/scaleFactor

        alpha = 0.95
        n_est = 300
        to_predict = 'values'



        values = np.zeros(len(tankPredict))
        #create a test dataframe for model evaluation
        dfTestPredicted=pd.DataFrame({'depot':depotPredict,'tank':tankPredict,'ship':shipPredict,'values':values})
        #train dataset
        X=dtf
        y=dtf[to_predict]
        X.drop([to_predict], axis=1, inplace=True)

        #test dataset
        X_test=dfTestPredicted
        y_test=dfTestPredicted[to_predict]
        X_test.drop([to_predict], axis=1, inplace=True)

            # over predict
        modelOP = XGBRegressor(objective=log_cosh_quantile(alpha),
                            n_estimators=n_est,
                            max_depth=10,
                            n_jobs=24,
                            learning_rate=.01)
        modelOP.fit(X, y)
            # under predict
        modelUP = XGBRegressor(objective=log_cosh_quantile(1-alpha),
                            n_estimators=n_est,
                            max_depth=10,
                            n_jobs=24,
                            learning_rate=.01)
        modelUP.fit(X, y)
            # single prediction
        modelAVG = XGBRegressor(n_estimators=n_est, max_depth=10, eta=0.01, subsample=1, colsample_bytree=1)
        modelAVG.fit(X, y)

        y_OP = modelOP.predict(X_test)
        #scale the values back up by scaleFactor
        y_OP=y_OP*scaleFactor
        y_UP = modelUP.predict(X_test)
        #scale the values back up by scaleFactor
        y_UP=y_UP*scaleFactor
        y_AVG = modelAVG.predict(X_test)
        #scale the values back up by scaleFactor
        y_AVG=y_AVG*scaleFactor
        y_actual_AVG = [original['values'].mean() for original in dfTestOriginal]
        y_actual_median = [original['values'].median() for original in dfTestOriginal]
        y_actual_OP = [original['values'].quantile(alpha) for original in dfTestOriginal]
        y_actual_UP = [original['values'].quantile(1-alpha) for original in dfTestOriginal]
        #calculate the RMSE for the predicted values of OP, UP, AVG, and median
        RMSE_OP = mean_squared_error(y_actual_OP, y_OP,squared=False)
        RMSE_UP = mean_squared_error(y_actual_UP, y_UP,squared=False)
        RMSE_AVG = mean_squared_error(y_actual_AVG, y_AVG,squared=False)
        RMSE_median = mean_squared_error(y_actual_median, y_AVG,squared=False)
        #create a dataframe with the RMSE values
        dfRMSE = pd.DataFrame({'Train_Size':[len(dfTrain)],'RMSE_UP':[RMSE_UP],'RMSE_AVG':[RMSE_AVG],'RMSE_median':[RMSE_median],'RMSE_OP':[RMSE_OP]})
        #concat the dataframe to the plotDFRMSE dataframe
        plotDFRMSE=pd.concat([plotDFRMSE,dfRMSE])
    #sort the plotDFRMSE dataframe by the train size
    plotDFRMSE=plotDFRMSE.sort_values(by=['Train_Size'])
    #generate a random string for the filename
    filename = ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))
    plotDFRMSE.to_latex(buf=os.path.join(latexDirectory,filename+"RMSE"+str(n_est)+".tex"), index=False)
    #generate CSV for the plotDFRMSE dataframe
    plotDFRMSE.to_csv(os.path.join(csvDirectory,filename+"RMSE.csv"), index=False)
    #create a plotly figure from the plotDFRMSE dataframe
    fig = px.line(plotDFRMSE, x="Train_Size", y=["RMSE_UP","RMSE_AVG","RMSE_median","RMSE_OP"], title='RMSE vs. Training Size')
    return fig 




#get the current working directory and join it with the directory "output_files" containing the data
directory=os.path.join(os.getcwd(), "output_files")
#make an image directory with the current working directory and the name "images"
imageDirectory=os.path.join(os.getcwd(), "images")
#make a directory for pandas latex tables
latexDirectory=os.path.join(os.getcwd(), "latex")
#make a directory for the csvs
csvDirectory=os.path.join(os.getcwd(), "csv")
#generate the list of dataframes
dfList=XGBDataSetCreator.generateIdleTabularDataFrameList(directory)
#drop index column for all dataframes
for df in dfList:
    df=df.drop(columns=["index"])
    
fig=generateRMSEPlot(dfList)
fig.show()

dfTestOriginal=[]
dfTrain=[]


#USE THIS CODE FOR RANDOM TESTING AND TRAINING DATA#################
#shuffle the list of dataframes

#randomly select 25% of the dataframes in dfList and put them in dfTestOriginal
#randomly select 75% of the dataframes in dfList and put them in dfTrain
for df in dfList:
    if np.random.rand() < 0.25:
        dfTestOriginal.append(df)
    else:
        dfTrain.append(df)
#create a depotPredict list containing the values of the depot column in each dataframe in dfTestOriginal
depotPredict=[df.iloc[0]["depot"] for df in dfTestOriginal]
#create a tankPredict list containing the values of the tank column in each dataframe in dfTestOriginal
tankPredict=[df.iloc[0]["tank"] for df in dfTestOriginal]
#create a shipPredict list containing the values of the ship column in each dataframe in dfTestOriginal
shipPredict=[df.iloc[0]["ship"] for df in dfTestOriginal]
#concatenate the dataframes in the list into a single dataframe
df=pd.concat(dfTrain)
#drop the columns "vname" as it is not needed for the model
df=df.drop(columns=["vname"])
#reset the index of the dataframe
dtf=df.reset_index(drop=True)
#drop the index column
dtf=dtf.drop(columns=["index"])
#convert all columns to int64
#dtf=dtf.astype('int64')
scaleFactor=500
#divide the values column by 100
dtf['values']=dtf['values']/scaleFactor

#####################################################3

#USE THIS CODE FOR SPECIFIC TESTING AND TRAINING DATA#################
#make arrays containing the values of the columns to predict
# depotPredict=[2,1,4]
# tankPredict=[720,360,1080]
# shipPredict=[10,15,30]
# #iterate through the dataframe list. separate dataframe with depots, tanks, and ships equal to specified values Put this dataframe in a separate list
# for df in dfList:
#     if df.iloc[0]["depot"]==2 and df.iloc[0]["tank"]==720 and df.iloc[0]["ship"]==10:
#         dfTestOriginal.append(df)
#     elif df.iloc[0]["depot"]==1 and df.iloc[0]["tank"]==360 and df.iloc[0]["ship"]==15:
#         dfTestOriginal.append(df)
#     elif df.iloc[0]["depot"]==4 and df.iloc[0]["tank"]==1080 and df.iloc[0]["ship"]==30:
#         dfTestOriginal.append(df)
#     else:
#         dfTrain.append(df)
##########################################################################


#concatenate the dataframes in the list into a single dataframe
df=pd.concat(dfTrain)
#drop the columns "vname" as it is not needed for the model
df=df.drop(columns=["vname"])
#reset the index of the dataframe
dtf=df.reset_index(drop=True)
#drop the index column
dtf=dtf.drop(columns=["index"])
#convert all columns to int64
#dtf=dtf.astype('int64')
scaleFactor=500
#divide the values column by 100
dtf['values']=dtf['values']/scaleFactor

# # Create an object to split input dataset into train and test datasets
# splitter = ShuffleSplit(n_splits=1, test_size=.25, random_state=0)


alpha = 0.95
n_est = 30000
to_predict = 'values'



values = np.zeros(len(tankPredict))
#create a test dataframe for model evaluation
dfTestPredicted=pd.DataFrame({'depot':depotPredict,'tank':tankPredict,'ship':shipPredict,'values':values})


#train dataset
X=dtf
y=dtf[to_predict]
X.drop([to_predict], axis=1, inplace=True)

#test dataset
X_test=dfTestPredicted
y_test=dfTestPredicted[to_predict]
X_test.drop([to_predict], axis=1, inplace=True)

    # over predict
modelOP = XGBRegressor(objective=log_cosh_quantile(alpha),
                       n_estimators=n_est,
                       max_depth=10,
                       n_jobs=24,
                       learning_rate=.01)
modelOP.fit(X, y)
    # under predict
modelUP = XGBRegressor(objective=log_cosh_quantile(1-alpha),
                       n_estimators=n_est,
                       max_depth=10,
                       n_jobs=24,
                       learning_rate=.01)
modelUP.fit(X, y)
    # single prediction
modelAVG = XGBRegressor(n_estimators=n_est, max_depth=10, eta=0.01, subsample=1, colsample_bytree=1)
modelAVG.fit(X, y)

y_OP = modelOP.predict(X_test)
#scale the values back up by scaleFactor
y_OP=y_OP*scaleFactor
y_UP = modelUP.predict(X_test)
#scale the values back up by scaleFactor
y_UP=y_UP*scaleFactor
y_AVG = modelAVG.predict(X_test)
#scale the values back up by scaleFactor
y_AVG=y_AVG*scaleFactor
y_actual_AVG = [original['values'].mean() for original in dfTestOriginal]
y_actual_median = [original['values'].median() for original in dfTestOriginal]
y_actual_OP = [original['values'].quantile(alpha) for original in dfTestOriginal]
y_actual_UP = [original['values'].quantile(1-alpha) for original in dfTestOriginal]


#make histogram plots in a dash from the "values" column in each dataframe in the dfTestOriginal list
app = dash.Dash()
children=[]
for i in range(len(dfTestOriginal)):
    df=dfTestOriginal[i]
    fig = go.Figure(data=[go.Histogram(x=df['values'])])
    fig.add_vline(x=y_actual_median[i], line_width=2, line_dash="solid", line_color="green")
    fig.add_vline(x=y_actual_AVG[i], line_width=2, line_dash="solid", line_color="yellow")
    fig.add_vline(x=y_actual_OP[i], line_width=2, line_dash="solid", line_color="blue")
    fig.add_vline(x=y_actual_UP[i], line_width=2, line_dash="solid", line_color="red")
    fig.add_vline(x=y_AVG[i], line_width=2, line_dash="dash", line_color="green")
    fig.add_vline(x=y_OP[i], line_width=2, line_dash="dash", line_color="blue")
    fig.add_vline(x=y_UP[i], line_width=2, line_dash="dash", line_color="red")
    #add a  centered title that shows the values of the depot, tank, and ship
    fig.update_layout(title={'text': 'Patrol Grid Average Idle Histogram for Run With '+str(df.iloc[0]["depot"])+' Depots, '+str(df.iloc[0]["tank"])+' sized Tank, and '+str(df.iloc[0]["ship"])+" Ships",
         'y':0.9, # new
         'x':0.5,
         'xanchor': 'center',
         'yanchor': 'top' # new
         })
    
    #add a legend describing the lines
    fig.update_layout(legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01
    ))
    #make x axis start at zero
    fig.update_xaxes(range=[0, max(df['values'])])
    #export the figure as a png to the images folder
    fig.write_image(os.path.join(imageDirectory,"histogram_"+str(df.iloc[0]["depot"])+"_"+str(df.iloc[0]["tank"])+"_"+str(df.iloc[0]["ship"])+".png"))
    children.append(dcc.Graph(figure=fig))
#create a list of percent differences between the actual and predicted values of OP, UP, and AVG
percentDiff_OP = [abs((y_actual_OP[i]-y_OP[i])/y_actual_OP[i])*100 for i in range(len(y_actual_OP))]
percentDiff_UP = [abs((y_actual_UP[i]-y_UP[i])/y_actual_UP[i])*100 for i in range(len(y_actual_UP))]
percentDiff_AVG = [abs((y_actual_AVG[i]-y_AVG[i])/y_actual_AVG[i])*100 for i in range(len(y_actual_AVG))]
percentDiff_median = [abs((y_actual_median[i]-y_AVG[i])/y_actual_median[i])*100 for i in range(len(y_actual_median))]
#create a pandas dataframe with the actual and predicted values of OP, UP, median and AVG and the percent differences as well as the values for the depot, tank, and ship
dfPercentDiff=pd.DataFrame({'depot':[df.iloc[0]["depot"] for df in dfTestOriginal], 'tank':[df.iloc[0]["tank"] for df in dfTestOriginal], 'ship':[df.iloc[0]["ship"] for df in dfTestOriginal],'predicted_UP':y_UP,'actual_UP':y_actual_UP,'percentDiff_UP':percentDiff_UP,
                            'predicted_Val':y_AVG,'actual_AVG':y_actual_AVG,'percentDiff_Val_AVG':percentDiff_AVG,
                            'actual_median':y_actual_median,'percentDiff_Val_Median':percentDiff_median,'predicted_OP':y_OP,'actual_OP':y_actual_OP,'percentDiff_OP':percentDiff_OP})

#write the dataframe to a latex table
dfPercentDiff.to_latex(buf=imageDirectory+"pDiff.tex", index=False)
#make a dash table from the dfPercentDiff dataframe. make the predicted value cells for UP and Op light red and light blue background respectively. make the actual value cells for OP and UP dark red and dark blue backgrounds respectively, and make the actual value cells for median and AVG dark green and dark yellow backgrounds respectively
children.append(dash_table.DataTable(
    id='table',
    columns=[{"name": i, "id": i} for i in dfPercentDiff.columns],
    data=dfPercentDiff.to_dict('records'),
    style_cell={
        
        'backgroundColor': 'rgb(50, 50, 50)',
        'color': 'white'
    },
    style_data_conditional=[
        {
            'if': {
                'column_id': 'predicted_OP'
            },
            'backgroundColor': 'rgb(0, 0, 50)',
            'color': 'white'
        },        {
            'if': {
                'column_id': 'actual_OP'
            },
            'backgroundColor': 'rgb(0, 0, 150)',
            'color': 'white'
        },
        {            
            'if': {
                'column_id': 'predicted_UP'
            },
            'backgroundColor': 'rgb(50, 0,0)',
            'color': 'white'
        },
        {
            'if': {
                'column_id': 'actual_UP'
            },
            'backgroundColor': 'rgb(150, 0, 0)',
            'color': 'white'
        },
        {
            'if': {
                'column_id': 'percentDiff_OP'
            },
            'backgroundColor': 'rgb(0, 0, 255)',
            'color': 'white'
        },
        {
            'if': {
                'column_id': 'percentDiff_UP'
            },
            'backgroundColor': 'rgb(255, 0, 0)',
            'color': 'white'
        },
        {
            'if': {
                'column_id': 'actual_AVG'
            },
            'backgroundColor': 'rgb(150, 150, 0)',
                        'color': 'black'
        },
                {
            'if': {
                'column_id': 'percentDiff_Val_AVG'
            },
            'backgroundColor': 'rgb(255, 255, 0)',
                        'color': 'black'
        },
        {
            'if': {
                'column_id': 'actual_median'
            },
            'backgroundColor': 'rgb(0, 150, 0)',
                                    'color': 'black'
        },
                {
            'if': {
                'column_id': 'percentDiff_Val_Median'
            },
            'backgroundColor': 'rgb(0, 255, 0)',
                                    'color': 'black'
        },
    ]
))
#calculate the RMSE for the predicted values of OP, UP, AVG, and median
RMSE_OP = mean_squared_error(y_actual_OP, y_OP,squared=False)
RMSE_UP = mean_squared_error(y_actual_UP, y_UP,squared=False)
RMSE_AVG = mean_squared_error(y_actual_AVG, y_AVG,squared=False)
RMSE_median = mean_squared_error(y_actual_median, y_AVG,squared=False)
#create a dataframe with the RMSE values
dfRMSE = pd.DataFrame({'RMSE_UP':[RMSE_UP],'RMSE_AVG':[RMSE_AVG],'RMSE_median':[RMSE_median],'RMSE_OP':[RMSE_OP]})
#write the dataframe to a latex table
dfRMSE.to_latex(buf=os.path.join(latexDirectory,"RMSE.tex"), index=False)
#make a dash table from the dfRMSE dataframe
children.append(dash_table.DataTable(
    id='table2',
    columns=[{"name": i, "id": i} for i in dfRMSE.columns],
    data=dfRMSE.to_dict('records')
))
# children.append(dash_table.DataTable(
#     id='table',
#     columns=[{"name": i, "id": i} for i in dfPercentDiff.columns],
#     data=dfPercentDiff.to_dict('records'), 
# ))

app.layout = html.Div(children=children)

app.run_server(debug=True)






# for train_index, test_index in splitter.split(dtf):
#     train = dtf.iloc[train_index]
#     test = dtf.iloc[test_index]

#     X = train
#     y = train[to_predict]
#     X.drop([to_predict], axis=1, inplace=True)

#     X_test = test
#     y_test = test[to_predict]
#     X_test.drop([to_predict], axis=1, inplace=True)

#     # over predict
#     model = XGBRegressor(objective=log_cosh_quantile(alpha),
#                        n_estimators=1250,
#                        max_depth=10,
#                        n_jobs=6,
#                        learning_rate=.05)

#     model.fit(X, y)
#     y_upper_smooth = model.predict(X_test)

#     # under predict
#     model = XGBRegressor(objective=log_cosh_quantile(1-alpha),
#                        n_estimators=1250,
#                        max_depth=10,
#                        n_jobs=6,
#                        learning_rate=.05)

#     model.fit(X, y)
#     y_lower_smooth = model.predict(X_test)
#     res = pd.DataFrame({'lower_bound' : y_lower_smooth, 'true_duration': y_test, 'upper_bound': y_upper_smooth})
#     res.to_csv('/tmp/duration_estimation.csv')

#     index = res['upper_bound'] < 0
#     print(res[res['upper_bound'] < 0])
#     print(X_test[index])

#     max_length = 150
#     fig = plt.figure()
#     plt.plot(list(y_test[:max_length]), 'gx', label=u'real value')
#     plt.plot(y_upper_smooth[:max_length], 'y_', label=u'Q up')
#     plt.plot(y_lower_smooth[:max_length], 'b_', label=u'Q low')
#     index = np.array(range(0, len(y_upper_smooth[:max_length])))
#     plt.fill(np.concatenate([index, index[::-1]]),
#              np.concatenate([y_upper_smooth[:max_length], y_lower_smooth[:max_length][::-1]]),
#              alpha=.5, fc='b', ec='None', label='90% prediction interval')
#     plt.xlabel('$index$')
#     plt.ylabel('$duration$')
#     plt.legend(loc='upper left')
#     plt.show()


#     count = res[(res.true_duration >= res.lower_bound) & (res.true_duration <= res.upper_bound)].shape[0]
#     total = res.shape[0]
#     print(f'pref = {count} / {total}')