#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 13:34:10 2021

@author: Andy
"""

import numpy as np
import pandas as pd
import scipy.stats as stats # use to calculate correlation
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from pyfinance import ols
from pyfinance.ols import PandasRollingOLS



s1 = pd.Series(range(6)) + 1
s2 = pd.Series(range(5)) + 2
s3 = pd.Series([6,5,8,30,12,1])

def CORR(df1, df2, n):
# use pearsonr to calculate correlation, the first term of results is correlation, the second is significance level
# the results should a number related to the latest date
    corrnum = df1.rolling(window=n,min_periods=n).corr(df2)
    return corrnum

def COVIANCE(df1, df2, n):
    cov = df1.rolling(window=n,min_periods=n).cov(df2)
    return cov

def STD(df, n):
    return df.rolling(window=n, min_periods=n).std()

def RANK(df):
# the results should be an df
    return df.rolling(window=len(df),min_periods=1).apply(lambda x: x.rank(pct = True).iloc[-1])

def BANCHMARK_INDEX(df):
    return df.rolling(window=len(df),min_periods=1).mean()

def LOG(df):
# the results should be an df  
    return np.log(df)

def DELTA(df, n):
    # if total lenth is p, the result should be a len(p-n) df 
    return df.diff(n)

def DELAY(df, n):
# the results should be a len(p-n) df
    return df.shift(n)

def MAX(df, n):
# to compare a df and a singal number "n"
    return df.where(df > n, n)

def MAX_two_dfs(df1, df2):
# to compare two dfs, rember to add a replace if df has Nan
    return np.maximum(df1, df2)

def MIN(df, n):
# to compare a df and a singal number "n"
    return df.where(df > n, n)

def MIN_two_dfs(df1, df2):
# to compare two dfs, rember to add a replace if df has Nan
    return np.minimum(df1, df2)

def SIGN(df):
# the results should an df only contains -1, 0, 1
    return np.sign(df)

def ABS(df):
# return the absolute value of df
    return df.abs()

def TSRANK(df, n):
# the result is the index of latest dates among the nth latest dates
    return df.rolling(window=n,min_periods=n).apply(lambda x: stats.rankdata(x)[-1]/n)

def TSMAX(df, n):
# the result should be an df, each item is max(df[i], n)
    return df.rolling(window=n,min_periods=n).max()

def TSMIN(df, n):
# the result should be an df, each item is min(df[i], n)
    return df.rolling(window=n,min_periods=n).min()

def SUM(df, n):
# The sum of the last n days of data for df 
    return df.rolling(window=n,min_periods=n).sum()

def SUMIF(df, n, condition):
# The sum of the last n days of data (satisfying the condition) for df,if not, treated it as zeros.
    sr = pd.Series(np.zeros(len(df)))
    sr[condition] = df[condition]
    return sr.rolling(window=n,min_periods=n).sum()

def SUMAC(df):
    return np.cumsum(df)

def SUMAC_MAX(df):
    return df.rolling(window = len(df), min_periods = 1).max()

def SUMAC_MIN(df):
    return df.rolling(window = len(df), min_periods = 1).min()

def MEAN(df, n):
# return the average of the last n days of data for df
    return df.rolling(window=n,min_periods=n).mean()

def DECAYLINEAR(df, n):
    w = np.array(range(1, n+1))
    Sum = w.sum()
    w = w / Sum
    return df.rolling(window=n, min_periods=n).apply(lambda x: np.dot(x, w))

def WMA(df, n):
    w = np.array(range(1, n+1)) * 0.9
    return df.rolling(window=n, min_periods=n).apply(lambda x: np.dot(x, w))

def SMA(df, n, m, periods = 0):
    return df.ewm(adjust=False, alpha=float(m)/n, min_periods=periods, ignore_na=False).mean()

def SEQUENCE(n):
    # return a sequence, e.g. if n = 5, the return should be [1,2,3,4,5]
    return pd.Series(list(range(1, n+1)))

def COUNT(condition, n):
    #in the past n data, the number of data which satisfies the condition
    return condition.rolling(window=n, min_periods=n).sum()
# print(DECAYLINEAR(s,2))
# print(COUNT(s>2,3))

def write(beg, end):
    for i in range(beg, end):
        exec('a' + str(i)  + '= alpha' + str(i) + '(Data)')
        exec('factors[' + str(i) + '] = a' + str(i))

def LOWDAY(df, n):
    return df.rolling(window = n, min_periods = n).apply(lambda x: n - 1 - x.argmin(axis=0))

def HIGHDAY(df, n):
    return df.rolling(window = n, min_periods = n).apply(lambda x: n - 1 - x.argmax(axis=0))

def REGRESI(df1, df2, n):
    return df1.rolling(window=n).apply(lambda x: ols.OLS(x, df2).alpha)

def REGBETA(df1, df2, n):
    return df1.rolling(window=n).apply(lambda x: ols.OLS(x, df2).beta)

def REGBETA_1(df1, df2, n):
    return PandasRollingOLS(df1, df2, window = n).beta

def REGRESI_1(df1, df2, n):
    return PandasRollingOLS(df1, df2, window = n).alpha

def PROD(df, n):
    return df.rolling(window=n, min_periods = n).apply(lambda x: np.cumprod(x,axis = 0).iloc[-1])



