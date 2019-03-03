# -*- coding: utf-8 -*-
"""
Created on Sat Feb 23 18:53:28 2019

@author: Rajas khokle
Code Title - Time Series Modelling
"""

import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import matplotlib.pyplot as plt


from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import acf,pacf
#from statsmodels.graphics.tsaplots import plot_acf,plot_pacf

# Create Connection to the database

engine = create_engine('postgres://postgres:DataAdmin@127.0.0.1:5432/Capstone')

# Get the Diabetes Database
diabetes_read="select * from diabetes where diabetes.tranbnfcode = '0601012V0BBAEAD' "
df = pd.read_sql(diabetes_read,engine)
data = df.groupby(['period']).sum()
data['period'] = data.index
plt.plot(data.quantity)


# Now Lets start building a model

data['dt'] = pd.to_datetime(data.period,format = '%Y%m',errors = 'coerce')
QD = pd.Series(data.quantity)
QD.index = data.dt
QD.to_csv('testdata.csv')
# Lets check the Stationarity

def Station(ds,rolling_window):
    rol_mean = ds.rolling(window=rolling_window).mean()
    rol_std = ds.rolling(window=rolling_window).std()
    # plot the time series
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

''' The p value from Dicky Fuller test should be less than 0.05 and adf value 
should be less than the critical value for ensuring the stationarity '''


# Check Stationarity of the data
    
Station(QD,12)

# Now considering that the model is not stationary at all, we need to find the 
# lag for which it will become stationary.

# For future automation, we can take square roots and other functions and then
# Evaluate stationarity for different functions and different lags. For initial
# model, only log function with 1st order differencing is consideres

# find optimal lag
# Lag more than 12 is not advisible due to seasonality


p_val=[]
adf=[]
crit_diff = []
QD_log = np.log(QD)              # np.sqrt and other functions may be used
for i in range(12):
    
    diff = QD_log - QD_log.shift(i+1)
    diff.dropna(inplace = True)
    adfoutput = Station(diff,12)
    p_val.append(adfoutput['p-value'])
    adf.append(adfoutput['adf'])
    crit_diff.append(adfoutput['adf'] - adfoutput['critical_value(1%)'])
    
lag = np.argmin(p_val)        # here we use the best lag for getting difference
QD_log_diff = QD_log - QD_log.shift(lag+1)
QD_log_diff.dropna(inplace = True)
adfoutput2=Station(QD_log_diff,3)
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
Station(residual,1)

# Let do the ACF and PACF on the log diff data

lag_acf = acf(QD_log_diff,nlags = 20)
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




# See where is the first 0 or 95% confidence crossing

pacf_zero_crossings = np.where(np.diff(np.sign(lag_pacf)))[0]

pacf_CI_crossings = np.where(np.diff(np.sign(lag_acf-CI)))[0][0]
p = pacf_CI_crossings

acf_zero_crossing = np.where(np.diff(np.sign(lag_pacf)))[0][0]
q =acf_zero_crossing

# Get the lag which is closest to the zero - It is the lag for which the data
# is most stationary.

P = np.argmin(abs(lag_pacf))
Q = np.argmin(abs(lag_acf))
# Construct ARIMA Model
I=2
order_tuple = (2,I,4)

model = ARIMA(QD_log_diff,order_tuple,freq='MS')
results_AR = model.fit()
pred = results_AR.fittedvalues

sumabserror = np.sum(np.abs(pred-QD_log_diff))
plt.plot(pred)
plt.plot(QD_log_diff)
plt.title('SAE %2f'% sumabserror)

# Convert  sum and then exponential to get origianl values

# Conversion
f, ax = plt.subplots()
ax.plot(QD.index, QD, label='Original Data')
ax.plot(QD.index[12:],
        np.exp(np.r_[np.log(QD.iloc[2]), 
                     pred].cumsum()),
       label='Fitted Data')
ax.set_title('Original data vs. UN-log-differenced data')
ax.legend(loc=0)


z = np.exp(np.r_[np.log(QD.iloc[2]),pred].cumsum())








pred_log = pd.Series(QD_log[0],index = QD_log.index)
predexp = np.exp(pred)
pred_log = pred_log.add(predexp,fill_value=0)

pred_final = np.exp(pred_log)

plt.plot(pred_final)
plt.plot(pred_log)
plt.plot(QD)
### Forcating is P in A

# Auto Arima
from pyramid.arima import auto_arima

stepwise_model = auto_arima(QD, start_p=1, start_q=1,
                           max_p=10, max_q=10, m=20,
                           start_P=0, seasonal=True,
                           d=1, D=1, trace=True,
                           error_action='ignore',  
                           suppress_warnings=True, 
                           stepwise=True)
stepwise_model.fit(QD)


future_forecast = stepwise_model.predict(n_periods=100)
future_forecast_ser = pd.Series(future_forecast,index = QD.index)
plt.plot(QD)
plt.plot(future_forecast_ser)

# FBprophet

diabetes_read="select * from diabetes where diabetes.tranbnfcode = '0601012V0BBAEAD' "
df = pd.read_sql(diabetes_read,engine)
data = df.groupby(['period']).sum()
data['period'] = data.index
plt.plot(data.quantity)

data['dt'] = pd.to_datetime(data.period,format = '%Y%m',errors = 'coerce')
QD = pd.Series(data.quantity)
QD.index = data.dt

ds=QD.index
y = data.quantity


x = pd.DataFrame(y,ds)
x['ds'] = x.index
x.reset_index(inplace =True,drop =True)
x.columns = ['y','ds']


from fbprophet import Prophet


model = Prophet()
model.fit(x)
future = model.make_future_dataframe(periods=12,freq='M')
forecast = model.predict(future)
fig1 = model.plot(forecast)






