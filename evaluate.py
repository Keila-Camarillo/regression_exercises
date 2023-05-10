import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error

import wrangle as w


import env
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import train_test_split
import sklearn.preprocessing
from sklearn.linear_model import LinearRegression

from scipy import stats
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score

from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import f_regression 
from math import sqrt
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')



def compare_sse(df, x, y):
    # create baseline
    df['yhat_baseline'] = df[y].mean()
    
    # creating simplae model 
    lr = LinearRegression()
    ols_model = lr.fit(df[[x]], df[y])
    df['yhat'] = ols_model.predict(df[[x]])
    
    # compute SSE
    SSE = mean_squared_error(df[y], df.yhat)*len(df)
    SSE_baseline = mean_squared_error(df[y], df.yhat_baseline)*len(df)
    
    # compute MSE
    MSE = mean_squared_error(df[y], df.yhat)
    MSE_baseline = mean_squared_error(df[y], df.yhat_baseline)
    
    # compute RMSE
    RMSE = sqrt(mean_squared_error(df[y], df.yhat))
    RMSE_baseline = sqrt(mean_squared_error(df[y], df.yhat_baseline))
    
    # compute ESS
    ESS = sum((df.yhat - df[y].mean())**2)
    
    # create dataframe
    df_eval = pd.DataFrame(np.array(['SSE','MSE','RMSE', 'SSE_baseline','MSE_baseline','RMSE_baseline']), columns=['metric'])

    df_eval['model_error'] = np.array([SSE, MSE, RMSE, SSE_baseline, MSE_baseline, RMSE_baseline])
    
    return pd.DataFrame(df_eval)

def compare_models(y, yhat, y_baseline):
    # Calculate the sum of squared errors for the model and the baseline
    model_sse = np.sum((y - yhat) ** 2)
    baseline_sse = np.sum((y - y_baseline) ** 2)

    # Compare the SSE values and output the result
    if model_sse < baseline_sse:
        print("Your model performs better than the baseline model.")
    elif model_sse > baseline_sse:
        print("Your model does not perform better than the baseline model.")
    else:
        print("Your model performs equally to the baseline model.")

def plot_residuals(y, yhat):
    # Calculate residuals
    residuals = y - yhat

    # Create a scatter plot of residuals
    plt.scatter(yhat, residuals)
    plt.axhline(y=0, color='r', linestyle='--')  # Add a horizontal line at y=0

    # Set plot labels and title
    plt.xlabel('Predicted values')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')

    # Show the plot
    plt.show()

def regression_errors(y, yhat):
    # Calculate the squared errors
    squared_errors = (y - yhat) ** 2

    # Calculate the sum of squared errors (SSE)
    sse = np.sum(squared_errors)

    # Calculate the explained sum of squares (ESS)
    ess = np.sum((yhat - np.mean(y)) ** 2)

    # Calculate the total sum of squares (TSS)
    tss = np.sum((y - np.mean(y)) ** 2)

    # Calculate the mean squared error (MSE)
    mse = np.mean(squared_errors)

    # Calculate the root mean squared error (RMSE)
    rmse = np.sqrt(mse)
    
    print("Sum of Squared Errors (SSE):", sse)
    print("Explained Sum of Squares (ESS):", ess)
    print("Total Sum of Squares (TSS):", tss)
    print("Mean Squared Error (MSE):", mse)
    print("Root Mean Squared Error (RMSE):", rmse)

    return sse, ess, tss, mse, rmse


def baseline_mean_errors(y):
    # Calculate the mean of y
    y_mean = np.mean(y)

    # Calculate the squared errors
    squared_errors = (y - y_mean) ** 2

    # Calculate the sum of squared errors (SSE)
    sse = np.sum(squared_errors)

    # Calculate the mean squared error (MSE)
    mse = np.mean(squared_errors)

    # Calculate the root mean squared error (RMSE)
    rmse = np.sqrt(mse)

    return sse, mse, rmse

def better_than_baseline(y, yhat):
    # Calculate the mean of y
    y_mean = np.mean(y)

    # Calculate the sum of squared errors for the model and the baseline
    model_sse = np.sum((y - yhat) ** 2)
    baseline_sse = np.sum((y - y_mean) ** 2)

    # Check if the model's SSE is lower than the baseline's SSE
    if model_sse < baseline_sse:
        return True
    else:
        return False