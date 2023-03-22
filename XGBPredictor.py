import XGBDataSetCreator
import os
import pandas as pd
import numpy as np
import sympy as sp
from xgboost.sklearn import XGBRegressor
from sklearn.model_selection import ShuffleSplit
import matplotlib.pyplot as plt
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
#concatenate the dataframes in the list into a single dataframe
df=pd.concat(dfList)
#drop the columns "vname" as it is not needed for the model
df=df.drop(columns=["vname"])
#reset the index of the dataframe
dtf=df.reset_index(drop=True)
#drop the index column
dtf=dtf.drop(columns=["index"])
#convert all columns to int64
#dtf=dtf.astype('int64')
#divide the values column by 100
dtf['values']=dtf['values']/1000

# Create an object to split input dataset into train and test datasets
splitter = ShuffleSplit(n_splits=1, test_size=.25, random_state=0)


alpha = 0.95
to_predict = 'values'


for train_index, test_index in splitter.split(dtf):
    train = dtf.iloc[train_index]
    test = dtf.iloc[test_index]

    X = train
    y = train[to_predict]
    X.drop([to_predict], axis=1, inplace=True)

    X_test = test
    y_test = test[to_predict]
    X_test.drop([to_predict], axis=1, inplace=True)

    # over predict
    model = XGBRegressor(objective=log_cosh_quantile(alpha),
                       n_estimators=125,
                       max_depth=5,
                       n_jobs=6,
                       learning_rate=.05)

    model.fit(X, y)
    y_upper_smooth = model.predict(X_test)

    # under predict
    model = XGBRegressor(objective=log_cosh_quantile(1-alpha),
                       n_estimators=125,
                       max_depth=5,
                       n_jobs=6,
                       learning_rate=.05)

    model.fit(X, y)
    y_lower_smooth = model.predict(X_test)
    res = pd.DataFrame({'lower_bound' : y_lower_smooth, 'true_duration': y_test, 'upper_bound': y_upper_smooth})
    res.to_csv('/tmp/duration_estimation.csv')

    index = res['upper_bound'] < 0
    print(res[res['upper_bound'] < 0])
    print(X_test[index])

    max_length = 150
    fig = plt.figure()
    plt.plot(list(y_test[:max_length]), 'gx', label=u'real value')
    plt.plot(y_upper_smooth[:max_length], 'y_', label=u'Q up')
    plt.plot(y_lower_smooth[:max_length], 'b_', label=u'Q low')
    index = np.array(range(0, len(y_upper_smooth[:max_length])))
    plt.fill(np.concatenate([index, index[::-1]]),
             np.concatenate([y_upper_smooth[:max_length], y_lower_smooth[:max_length][::-1]]),
             alpha=.5, fc='b', ec='None', label='90% prediction interval')
    plt.xlabel('$index$')
    plt.ylabel('$duration$')
    plt.legend(loc='upper left')
    plt.show()


    count = res[(res.true_duration >= res.lower_bound) & (res.true_duration <= res.upper_bound)].shape[0]
    total = res.shape[0]
    print(f'pref = {count} / {total}')