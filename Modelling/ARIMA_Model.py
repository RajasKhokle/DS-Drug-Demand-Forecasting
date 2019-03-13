# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 16:40:32 2019
@author: Rajas Khokle
Purpose: Investigate ARIMA Modelling techniques for drug demand forecasting on
         Glargine Pen data.
"""
# Import the Libraries

import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import matplotlib.pyplot as plt


from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import acf,pacf

# Create Connection to the database

engine = create_engine('postgres://postgres:DataAdmin@127.0.0.1:5432/Capstone')


# Drug load function
def load_drug(drug):
    
    sql_string = '''SELECT sum(quantity) as quantity,period FROM "Casptone_Tableau" WHERE TRANBNFCODE = '''+drug+ 'group by period '
    df = pd.read_sql(sql_string,engine)
    df['dt'] = pd.to_datetime(df.period, format = '%Y%m',errors = 'coerce')
    QD = pd.Series(df.quantity)
    QD.index = df.dt
    return(QD)

# Function to check the stationairty
def Stationary(ds,rolling_window):
    rol_mean = ds.rolling(window=rolling_window).mean()
    rol_std = ds.rolling(window=rolling_window).std()
    ##  plot the time series
    plt.plot(ds,color ='blue',label='Original')
    plt.plot(rol_mean,color ='red',label='Rolling_Mean')
    plt.plot(rol_std,color ='black',label='Rolling_STD')
    plt.legend(loc = 'best')
    plt.title('Rolling Mean and Std')
    plt.show()
    ## Run Dickey Fuller Test
    adftest = adfuller(ds,autolag = 'AIC')
    adfoutput = pd.Series(adftest[0:4],index = ['adf','p-value','lags_used',
                          'number_of_obseravtions'])
  
    for key,value in adftest[4].items():
    	adfoutput['critical_value(%s)'%key] = value
    
    return(adfoutput)

lan_pen = "'0601012V0BBAEAD'"
QD = load_drug(lan_pen) 
adf = Stationary(QD,12)     # Twevele becasue of Seasonality of 12 months. 
# Rolling means (or moving averages) are generally used to smooth out 
# short-term fluctuations in time series data and highlight long-term trends.

#Now differencing takes care of the trends. To find out optimal lag we go through
# different lags and evaluate the stationarity through dicky fuller test

p_val=[]
adf=[]
crit_diff = []
QD_log = np.log(QD)              # np.sqrt and other functions may be used
for i in range(12):
    
    diff = QD_log - QD_log.shift(i)
    diff.dropna(inplace = True)
    adfoutput = Stationary(diff,12)
    p_val.append(adfoutput['p-value'])
    adf.append(adfoutput['adf'])
    crit_diff.append(adfoutput['adf'] - adfoutput['critical_value(1%)'])
    
lag = np.argmin(p_val)        # here we use the best lag for getting difference

QD_log_diff = QD_log - QD_log.shift(lag+1)
QD_log_diff.dropna(inplace = True)

# Now to determine the P and Q values for ARIMA, ACF and PACF calculations are 
# used.

# Let do the ACF and PACF on the log diff data

lag_acf = acf(QD_log_diff,nlags = 12)
lag_pacf = pacf(QD_log_diff,nlags=12,method = 'ols')


# plot ACF and calculate the 95% confidence ineterval
CI = 1.9/np.sqrt(len(QD_log_diff))
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

# Construct ARIMA Model
I=1
order_tuple = (5,I,4)

model = ARIMA(QD_log_diff,order_tuple,freq='MS')
results_AR = model.fit()
pred = results_AR.fittedvalues

pred_cumsum=pred.cumsum()

rmse = np.sqrt(np.sum(np.abs(pred-QD_log_diff)**2))
plt.plot(pred)
plt.plot(QD_log_diff)
plt.title('RMSE %2f'% rmse)

# Convert  sum and then exponential to get origianl values
f, ax = plt.subplots()
ax.plot(QD.index, QD, label='Original Data')
ax.plot(QD.index[1:],
        np.exp((np.r_[np.log(QD.iloc[2]), 
                     pred].cumsum())),
       label='Fitted Data')
ax.set_title('Original data vs. UN-log-differenced data')
ax.legend(loc=0)










