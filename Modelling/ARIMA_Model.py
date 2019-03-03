# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 16:40:32 2019

@author: Rajas Khokle
"""
import pandas as pd
from sqlalchemy import create_engine
from fbprophet import Prophet
import matplotlib.pyplot as plt


# Create Connection to the database

engine = create_engine('postgres://postgres:DataAdmin@127.0.0.1:5432/Capstone')

# Read the data from database
Drug = "'0601011L0AAACAC'"
Drug_read='select * from diabetes where diabetes.tranbnfcode = ' + Drug 
df = pd.read_sql(Drug_read,engine)
data = df.groupby(['period']).sum()
data['period'] = data.index
plt.plot(data.quantity)
data['dt'] = pd.to_datetime(data.period,format = '%Y%m',errors = 'coerce')

QD = pd.Series(data.quantity) # QD is Quantity Data Series
QD.index = data.dt

CD = pd.Series(data.nic)
CD.index = data.dt

ds=QD.index                  # Column for datestamp in Prophet model
y = data.quantity            # Column for target in Prophet model  

data_dict = {'ds':ds,'y':y}
x = pd.DataFrame(data_dict)
x.reset_index(inplace =True,drop =True)


model = Prophet()
model.fit(x)
future = model.make_future_dataframe(periods=12,freq='M') # Forecast for 12 months
forecast = model.predict(future)
fig1 = model.plot(forecast)

Query = "select * from ukdrug_address where tranbnfcode = '0601012D0BBAVBZ'"
humulin =pd.read_sql(Query,engine)
humulin ['country'] = 'United Kingdom'
humulin.to_csv('humulin.csv')
