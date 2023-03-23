import XGBDataSetCreator
import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import dash
import sympy as sp
from xgboost.sklearn import XGBRegressor
from sklearn.model_selection import ShuffleSplit
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
#get the current working directory and join it with the directory "output_files" containing the data
directory=os.path.join(os.getcwd(), "output_files")
#generate the list of dataframes
dfList=XGBDataSetCreator.generateIdleTabularDataFrameList(directory)
#drop index column for all dataframes
for df in dfList:
    df=df.drop(columns=["index"])
    
dfTestOriginal=[]
dfTrain=[]
#iterate through the dataframe list. separate dataframe with depots, tanks, and ships equal to specified values Put this dataframe in a separate list
for df in dfList:
    if df.iloc[0]["depot"]==2 and df.iloc[0]["tank"]==720 and df.iloc[0]["ship"]==10:
        dfTestOriginal.append(df)
    elif df.iloc[0]["depot"]==1 and df.iloc[0]["tank"]==360 and df.iloc[0]["ship"]==15:
        dfTestOriginal.append(df)
    elif df.iloc[0]["depot"]==4 and df.iloc[0]["tank"]==1080 and df.iloc[0]["ship"]==30:
        dfTestOriginal.append(df)
    else:
        dfTrain.append(df)

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
n_est = 30
to_predict = 'values'


#make arrays containing the values of the columns to predict
depotPredict=[2,1,4]
tankPredict=[720,360,1080]
shipPredict=[10,15,30]
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
    #make x axis start at zero
    fig.update_xaxes(range=[0, max(df['values'])])
    children.append(dcc.Graph(figure=fig))
#create a list of percent differences between the actual and predicted values of OP, UP, and AVG
percentDiff_OP = [abs((y_actual_OP[i]-y_OP[i])/y_actual_OP[i])*100 for i in range(len(y_actual_OP))]
percentDiff_UP = [abs((y_actual_UP[i]-y_UP[i])/y_actual_UP[i])*100 for i in range(len(y_actual_UP))]
percentDiff_AVG = [abs((y_actual_AVG[i]-y_AVG[i])/y_actual_AVG[i])*100 for i in range(len(y_actual_AVG))]
percentDiff_median = [abs((y_actual_median[i]-y_AVG[i])/y_actual_median[i])*100 for i in range(len(y_actual_median))]
#create a pandas dataframe with the actual and predicted values of OP, UP, median and AVG and the percent differences
dfPercentDiff=pd.DataFrame({'predicted_UP':y_UP,'actual_UP':y_actual_UP,'percentDiff_UP':percentDiff_UP,
                            'predicted_Val':y_AVG,'actual_AVG':y_actual_AVG,'percentDiff_Val_AVG':percentDiff_AVG,
                            'actual_median':y_actual_median,'percentDiff_Val_Median':percentDiff_median,'predicted_OP':y_OP,'actual_OP':y_actual_OP,'percentDiff_OP':percentDiff_OP})


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
            'backgroundColor': 'rgb(0, 0, 150)',
            'color': 'white'
        },        {
            'if': {
                'column_id': 'actual_OP'
            },
            'backgroundColor': 'rgb(0, 0, 255)',
            'color': 'white'
        },
        {            
            'if': {
                'column_id': 'predicted_UP'
            },
            'backgroundColor': 'rgb(150, 0,0)',
            'color': 'white'
        },
        {
            'if': {
                'column_id': 'actual_UP'
            },
            'backgroundColor': 'rgb(255, 0, 0)',
            'color': 'white'
        },
        {
            'if': {
                'column_id': 'percentDiff_OP'
            },
            'backgroundColor': 'rgb(0, 0, 50)',
            'color': 'white'
        },
        {
            'if': {
                'column_id': 'percentDiff_UP'
            },
            'backgroundColor': 'rgb(50, 0, 0)',
            'color': 'white'
        },
        {
            'if': {
                'column_id': 'actual_AVG'
            },
            'backgroundColor': 'rgb(255, 255, 0)',
                        'color': 'black'
        },
                {
            'if': {
                'column_id': 'percentDiff_Val_AVG'
            },
            'backgroundColor': 'rgb(150, 150, 0)',
                        'color': 'black'
        },
        {
            'if': {
                'column_id': 'actual_median'
            },
            'backgroundColor': 'rgb(0, 255, 0)',
                                    'color': 'black'
        },
                {
            'if': {
                'column_id': 'percentDiff_Val_Median'
            },
            'backgroundColor': 'rgb(0, 150, 0)',
                                    'color': 'black'
        },
    ]
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