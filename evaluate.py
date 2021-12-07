import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import f_regression 
from math import sqrt
import seaborn as sns

def plot_residuals(y, yhat):
    baseline = y.mean()
    baseline_residual = y - baseline
    plt.figure(figsize = (11,5))

    plt.subplot(121)
    plt.scatter(y, baseline_residual)
    plt.axhline(y = 0, ls = ':')
    plt.xlabel('x_variable')
    plt.ylabel('Residual')
    plt.title('Baseline Residuals')

    plt.subplot(122)
    plt.scatter(y, (y - yhat))
    plt.axhline(y = 0, ls = ':')
    plt.xlabel('x_variable')
    plt.ylabel('Residual')
    plt.title('OLS model residuals');

def regression_errors(y,yhat):
    R = y - yhat
    R2 = R**2
    SSE = R2.sum()
    print('SSE = ', "{:.1f}".format(SSE))

    baseline = y.mean()
    baseline_residual = y - baseline
    baseline_R2 = baseline_residual**2
    SSE_baseline = baseline_R2.sum()
    print('SSE Baseline = ', "{:.1f}".format(SSE_baseline))

    ESS = SSE_baseline - SSE
    print('ESS = ', "{:.1f}".format(ESS))

    TSS = SSE_baseline
    print('TSS = ', "{:.1f}".format(TSS))

    MSE = SSE/len(y)
    MSE_baseline = SSE_baseline/len(y)
    print('MSE = ', "{:.1f}".format(MSE))
    print('MSE Baseline = ', "{:.1f}".format(MSE_baseline))

    from math import sqrt
    RMSE = sqrt(MSE)
    RMSE_baseline = sqrt(MSE_baseline)
    print('RMSE = ', "{:.1f}".format(RMSE))
    print('RMSE Baseline = ', "{:.1f}".format(RMSE_baseline))

def better_than_baseline(y,yhat):
    R = y - yhat
    R2 = R**2
    SSE = R2.sum()
    baseline = y.mean()
    baseline_residual = y - baseline
    baseline_R2 = baseline_residual**2
    SSE_baseline = baseline_R2.sum()
    ESS = SSE_baseline - SSE
    TSS = SSE_baseline
    MSE = SSE/len(y)
    MSE_baseline = SSE_baseline/len(y)
    from math import sqrt
    RMSE = sqrt(MSE)
    RMSE_baseline = sqrt(MSE_baseline)
    if RMSE < RMSE_baseline:
        print('The regression model outperforms baseline')
    else:
        print('The baseline model outperforms regression')