# -*- coding: utf-8 -*-
"""
Created on Sat Feb 23 18:53:28 2019

@author: Rajas khokle
Code Title - Time Series Modelling
"""

import pandas as pd
from sqlalchemy import create_engine

import matplotlib.pyplot as plt

# Visualize top 3 drugs in insulin category.

# Create Connection to the database

engine = create_engine('postgres://postgres:DataAdmin@127.0.0.1:5432/Capstone')

# Get the Diabetes Database
diabetes_read="select * from diabetes where diabetes.tranbnfcode = '0601012D0BBAVBZ' "
df = pd.read_sql(diabetes_read,engine)
#df.period = df.period.astype('int64')

data = df.groupby(['period']).sum()

data['period'] = data.index

plt.plot(data.quantity)


# Get the Diabetes Database
diabetes_read="select * from diabetes where diabetes.tranbnfcode = '0601012W0BBABAB' "
df = pd.read_sql(diabetes_read,engine)
#df.period = df.period.astype('int64')

data = df.groupby(['period']).sum()

data['period'] = data.index

plt.plot(data.quantity)


# Get the Diabetes Database
diabetes_read="select * from diabetes where diabetes.tranbnfcode = '0601012V0BBAEAD' "
df = pd.read_sql(diabetes_read,engine)
#df.period = df.period.astype('int64')

data = df.groupby(['period']).sum()

data['period'] = data.index

plt.plot(data.quantity)


# Now Lets start building a model
 
# convert period to datetime object
from datetime import datetime

def period2dt (period):   # Period should contain year followed by month
    year = period[0:4]
    year = int(year)
    month = period[4:6]
    month = int(month)
    date = 1
    return (datetime(year,month,date))


data['dt'] = data.period.apply(period2dt)


# same thing can be accomplished in one line using pandas. This one is better 
# as it has capacity to throw errors, which is not present in our custome func.

data['dt2'] = pd.to_datetime(data.period,format = '%Y%m',errors = 'coerce')

QD = pd.Series(data.quantity)
QD.index = data.dt2

# Lets check the Stationarity

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import acf,pacf
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf

import numpy as np

def Station(df,col_name,rolling_window):
    rol_mean = df.rolling(window=rolling_window).mean()
    rol_std = df.rolling(window=rolling_window).std()
    # plot the time series
#     plt.figsize(14,6)
    plt.plot(df,color ='blue',label='Original')
    plt.plot(rol_mean,color ='red',label='Rolling_Mean')
    plt.plot(rol_std,color ='black',label='Rolling_STD')
    plt.legend(loc = 'best')
    plt.title('Rolling Mean and Std')
    plt.show()
    
    # Check Stationarity
    ## Plot a Histogram
    print('Data Visualization')  # Histogram should show normal distribution
    df.hist()
    plt.show()
    
        
    ## Run Dickey Fuller Test
       
    adftest = adfuller(df[col_name],autolag = 'AIC')
    adfoutput = pd.Series(adftest[0:4],index = ['adf','p-value','lags_used',
                          'number_of_obseravtions'])
    
    for key,value in adftest[4].items():
    	adfoutput['critical_value(%s)'%key] = value
    
    print(adfoutput)

''' The p value from Dicky Fuller test should be less than 0.05 and adf value 
should be less than the critical value for ensuring the stationarity '''
    
Station(QD,'quantity',12)

# Now considering that the model is not stationary at all, we need to find the 
# lag for which it will become stationary.

# For duture automation, we can take square roots and other functions and then
# Evaluate stationarity for different functions and different lags. For initial
# model, only log function with 1st order differencing is consideres


QD_log = np.sqrt(QD)              # np.sqrt and other functions may be used
Station(QD_log,'quantity',1)
diff1 = QD_log - QD_log.shift(1)
diff1.dropna(inplace = True)
Station(diff1,'quantity',12)

# exponential weighted mean        # May or may not produce great stationarity
expweightavg = QD_log.ewm(halflife = 12,adjust=True).mean()
diff2 = QD_log - expweightavg
Station(diff2,'quantity',12)
diff3 = diff2 - diff2.shift(1)
diff3.dropna(inplace = True)
Station(diff3,'quantity',1)


# Decompose the model

ss_decomposition = seasonal_decompose(QD_log)
fig = plt.figure()
fig.set_size_inches(12,10)
fig = ss_decomposition.plot()
plt.show

# Get the trend, seasonal and residual data
trend = ss_decomposition.trend
seasonal = ss_decomposition.seasonal
residual = ss_decomposition.resid
residual.dropna(inplace = True)
plt.plot(residual)
Station(residual,'quantity',1)

# Let do the ACF and PACF on the log diff data

lag_acf = acf(diff1,nlags = 10)
lag_pacf = pacf(diff1,nlags=10,method = 'ols')


# plot ACF and calculate the 95% confidence ineterval
CI = 1.9/np.sqrt(len(diff1))
plt.plot(lag_acf)
plt.axhline(y=0,linestyle='--',color = 'blue')
plt.axhline(y = -CI,linestyle='--',color = 'blue')
plt.axhline(y = CI,linestyle='--',color = 'blue')
plt.title('Autocorrelation Function')

# Plot PACF
plt.plot(lag_pacf)
plt.axhline(y=0,linestyle='--',color = 'blue')
plt.axhline(y = -CI,linestyle='--',color = 'blue')
plt.axhline(y = CI,linestyle='--',color = 'blue')
plt.title('Partial Autocorrelation Function')

# See where is the first 0 or 95% confidence crossing

pacf_zero_crossings = np.where(np.diff(np.sign(lag_pacf)))[0]

pacf_CI_crossings = np.where(np.diff(np.sign(lag_acf-CI)))[0][0]
p = pacf_CI_crossings

acf_zero_crossing = np.where(np.diff(np.sign(lag_pacf)))[0][0]
q =acf_zero_crossing

# Get the lag which is closest to the zero
P = np.argmin(abs(lag_pacf))
Q = np.argmin(abs(lag_acf))
# Construct ARIMA Model
I=1
order_tuple = (2,I,2)

model = ARIMA(QD_log,order_tuple,freq='MS')
results_AR = model.fit()
pred = results_AR.fittedvalues
sumabserror = np.sum(np.abs(pred-diff1))
plt.plot(diff1)
plt.plot(results_AR.fittedvalues)
plt.title('SAE %2f'% sumabserror)

# Convert cumulative sum and then exponential to get origianl values

pred_cumul = pred.cumsum()

pred_cumul_log = pd.Series(QD_log[0],index = QD_log.index)
pred_cumul_log = pred_cumul_log.add(pred_cumul,fill_value = 0)

pred_final = np.exp(pred_cumul_log)

plt.plot(QD)
plt.plot(pred_final)








    
